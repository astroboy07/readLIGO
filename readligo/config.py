"""Settings of the v2 cleaning pipeline, and the frequency-grid rules.

Frequency grid
--------------
Every band edge is a multiple of 1/1800 Hz, the bin width of the 1800 s SFTs
that Makefakedata_v5 (MFDv5) works on. rustyprune and pyloki require the same
for the synthetic data, and it makes the channelizer exact: with block
lengths that are multiples of 900 s, every band edge falls on an FFT bin and
the output rate 2B is reached with no resampling or rounding. Any bandwidth
B on this grid is allowed (0.1, 0.15, 0.37, 0.55, 1.0 Hz, ...), and the
output is sampled at exactly ``fs = 2B``. rustyprune additionally needs
``B = k/10`` Hz (k = 1..10); :class:`BandSpec` warns otherwise.
"""
from __future__ import annotations

import dataclasses
import json
import math
import warnings

import numpy as np
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from .conditioning import GateParams
from .inventory import DETECTORS

__all__ = [
    "DEFAULT_EPOCH",
    "FREQ_GRID_DEN",
    "BandSpec",
    "ChannelizerParams",
    "NoiseParams",
    "PipelineConfig",
    "VetoParams",
    "band_around",
    "default_guard",
    "to_grid",
]

#: Band edges are multiples of 1/FREQ_GRID_DEN Hz (the 1800 s SFT bin).
FREQ_GRID_DEN = 1800

#: Largest distance (Hz) from the grid that :func:`to_grid` snaps.
GRID_TOLERANCE = 1.2e-6

#: Default global epoch: output time grid origin and heterodyne reference.
#: This is rustyprune's and pyloki's hard-coded ``tstart`` (and the start of
#: GWOSC file H-H1_GWOSC_O3a_4KHZ_R1-1238319104-4096), so data exported from
#: this epoch line up sample for sample with their synthetic MFDv5 data.
DEFAULT_EPOCH = 1238319104


def to_grid(value: float | Fraction | str, name: str = "frequency") -> Fraction:
    """*value* as an exact multiple of 1/1800 Hz.

    Values within ``GRID_TOLERANCE`` of a grid point (about 1e-6 Hz, enough
    for the 6-decimal form of any grid value, e.g. ``1390.885556`` for
    2503594/1800) are snapped to it exactly.

    Raises
    ------
    ValueError
        If *value* is not a multiple of 1/1800 Hz. The message names the two
        nearest valid values.
    """
    if isinstance(value, Fraction):
        fr = value
    else:
        x = float(Fraction(value)) if isinstance(value, str) else float(value)
        k = round(x * FREQ_GRID_DEN)
        if abs(x * FREQ_GRID_DEN - k) > GRID_TOLERANCE * FREQ_GRID_DEN:
            lo = math.floor(x * FREQ_GRID_DEN)
            raise ValueError(
                f"{name} = {x!r} Hz is not a multiple of 1/{FREQ_GRID_DEN} Hz; nearest valid values "
                f"are {lo / FREQ_GRID_DEN!r} and {(lo + 1) / FREQ_GRID_DEN!r} Hz"
            )
        fr = Fraction(k, FREQ_GRID_DEN)
    if (fr * FREQ_GRID_DEN).denominator != 1:
        raise ValueError(f"{name} = {fr} Hz is not a multiple of 1/{FREQ_GRID_DEN} Hz")
    return fr


@dataclass(frozen=True)
class BandSpec:
    """One output band ``[fmin, fmin + band]`` (Hz), both on the 1/1800 Hz grid.

    Accepts floats (``BandSpec(100.68, 0.1)``) and keeps exact fractions
    internally, so ``fmax - fmin`` is exactly ``band``.
    """

    fmin: Fraction
    band: Fraction
    #: Source frequencies (Hz) whose whole detector-frame track stays inside the
    #: flat part of the band (from :func:`readligo.planning.plan_bands`), or None.
    usable: tuple[float, float] | None = field(default=None, compare=False)

    def __init__(
        self,
        fmin: float | Fraction | str,
        band: float | Fraction | str,
        usable: tuple[float, float] | None = None,
    ) -> None:
        object.__setattr__(self, "fmin", to_grid(fmin, "fmin"))
        object.__setattr__(self, "band", to_grid(band, "band"))
        if self.band <= 0:
            raise ValueError(f"band must be positive, got {float(self.band)} Hz")
        if self.fmin <= 0:
            raise ValueError(f"fmin must be positive, got {float(self.fmin)} Hz")
        if usable is not None:
            lo, hi = float(usable[0]), float(usable[1])
            if not (float(self.fmin) <= lo < hi <= float(self.fmax)):
                raise ValueError(f"usable range [{lo}, {hi}] Hz is not inside the band {self.label()}")
            usable = (lo, hi)
        object.__setattr__(self, "usable", usable)
        if (self.band * 10).denominator != 1 or not (1 <= self.band * 10 <= 10):
            warnings.warn(
                f"band {float(self.band)} Hz is valid here, but rustyprune and pyloki currently "
                "accept only band = k/10 Hz with k = 1..10",
                stacklevel=2,
            )

    def __repr__(self) -> str:
        u = "" if self.usable is None else f", usable={self.usable!r}"
        return f"BandSpec(fmin={float(self.fmin)!r}, band={float(self.band)!r}{u})"

    def to_list(self) -> list[str | float]:
        """``[fmin, band]`` as exact fraction strings, plus ``usable_lo, usable_hi`` if set."""
        out: list[str | float] = [str(self.fmin), str(self.band)]
        return out if self.usable is None else [*out, *self.usable]

    @property
    def fmax(self) -> Fraction:
        return self.fmin + self.band

    @property
    def fs(self) -> Fraction:
        """Output sample rate, exactly 2B."""
        return 2 * self.band

    def label(self) -> str:
        return f"{float(self.fmin):g}-{float(self.fmax):g} Hz"

    def as_tuple(self) -> tuple[float, float]:
        return float(self.fmin), float(self.band)


def default_guard() -> float:
    """Guard (Hz) of the default channelizer (L = 3600 s, 120 dB): 2.18 mHz."""
    from .channelize import optimal_beta, transition_halfwidth

    return 2 * transition_halfwidth(optimal_beta(120.0), 120.0) / 3600


def band_around(
    f_lo: float, f_hi: float | None = None, band: float | Fraction = 0.1, *, guard: float | None = None
) -> BandSpec:
    """Band of width *band* centred on ``[f_lo, f_hi]``, with fmin on the grid.

    This is the rule rustyprune uses for its injections (``centered_fmin``)
    and the one used to place the hardware-injection bands. The range must
    lie in the flat part ``[fmin + guard, fmax - guard]`` (default guard: that
    of the default channelizer, 2.18 mHz).

    Raises
    ------
    ValueError
        If ``f_hi < f_lo`` or ``[f_lo, f_hi]`` does not fit inside the flat part.
    """
    f_hi = f_lo if f_hi is None else f_hi
    if f_hi < f_lo:
        raise ValueError(f"f_hi = {f_hi} is below f_lo = {f_lo}")
    g = default_guard() if guard is None else float(guard)
    b = to_grid(band, "band")
    centre = 0.5 * (f_lo + f_hi)
    fmin = Fraction(round((centre - float(b) / 2) * FREQ_GRID_DEN), FREQ_GRID_DEN)
    spec = BandSpec(fmin, b)
    if not (float(spec.fmin) + g <= f_lo and f_hi <= float(spec.fmax) - g):
        raise ValueError(f"[{f_lo}, {f_hi}] Hz does not fit in the flat part of a {float(b)} Hz band (guard {g:.4g} Hz)")
    return spec


@dataclass(frozen=True)
class ChannelizerParams:
    """Settings of the FFT channelizer (:mod:`readligo.channelize`).

    ``kernel_length`` L sets the guard: the response is flat to 1e-6 on
    ``[fmin + g, fmax - g]`` with ``g = 7.84 / L`` Hz (2.18 mHz for
    L = 3600 s) and rolls off to 1e-6 at the band edges, so nothing outside
    the band can alias in (``edge="inside"``).
    """

    kernel_length: int = 3600
    hop: int = 3600
    atten_db: float = 120.0
    edge: str = "inside"
    #: In-band flattening (ISSUES.md A8): scale each band by a smooth gain so its
    #: average noise PSD is flat (the searches use one variance per band). The
    #: shape is a quadratic fit to log PSD, from a mean Welch PSD of the
    #: conditioned strain with ``flatten_psd_seg`` s segments; it is applied
    #: only where it changes the PSD by more than ``flatten_min`` across the band.
    flatten: bool = True
    flatten_psd_seg: int = 128
    flatten_min: float = 0.01
    #: no flattening where the smooth fit varies by more than this factor across
    #: the band (line skirts, not a noise floor)
    flatten_max_range: float = 10.0

    def __post_init__(self) -> None:
        if self.kernel_length <= 0 or self.kernel_length % FREQ_GRID_DEN:
            raise ValueError(f"kernel_length = {self.kernel_length} s must be a positive multiple of {FREQ_GRID_DEN} s")
        if self.hop <= 0 or self.hop % (FREQ_GRID_DEN // 2):
            raise ValueError(f"hop = {self.hop} s must be a positive multiple of {FREQ_GRID_DEN // 2} s")
        if self.edge not in ("inside", "center", "outside"):
            raise ValueError(f"edge must be inside, center or outside, not {self.edge!r}")
        if self.atten_db < 20:
            raise ValueError("atten_db must be at least 20 dB")


@dataclass(frozen=True)
class NoiseParams:
    """Settings of the noise estimate (:mod:`readligo.noise`)."""

    #: Length (s) of the segments with their own noise level.
    segment: int = 1800
    #: Segments with a smaller live fraction (mean squared window) get no noise estimate.
    min_duty: float = 0.5
    #: Kaiser beta of the periodogram taper (sidelobes below about -130 dB, so a
    #: strong line does not leak across the band; ISSUES.md A7).
    taper_beta: float = 16.0
    #: A line is a run of bins whose segment-averaged power exceeds the local
    #: baseline with false-alarm probability ``line_false_alarm`` per product
    #: (exact Gamma tail for the number of segments) somewhere, extended while
    #: it exceeds the ``line_extend_false_alarm`` level per bin.
    line_false_alarm: float = 1e-3
    line_extend_false_alarm: float = 0.05
    #: Width (bins) of the running median that defines the local baseline.
    baseline_bins: int = 31

    def __post_init__(self) -> None:
        if not (self.segment > 0 and float(self.segment).is_integer()):
            raise ValueError(f"NoiseParams.segment must be a positive whole number of seconds, got {self.segment}")
        if not (0 < self.min_duty <= 1):
            raise ValueError("NoiseParams.min_duty must be in (0, 1]")
        if not (0 < self.line_false_alarm < 1 and 0 < self.line_extend_false_alarm < 1):
            raise ValueError("NoiseParams false-alarm probabilities must be in (0, 1)")
        if self.taper_beta < 0 or self.baseline_bins < 3:
            raise ValueError("NoiseParams.taper_beta must be >= 0 and baseline_bins >= 3")


@dataclass(frozen=True)
class VetoParams:
    """Narrowband outlier veto on each band's output (:func:`noise.narrowband_veto`).

    Catches what broadband gating cannot see: transients concentrated in a
    narrow or low band (e.g. scattered light at 20-60 Hz) and the brief
    leakage of strong lines next to gates. With ``z = x^2 / s^2``, where s^2 is
    a robust variance of the segment's samples (median of x^2 / 0.4549, so a
    stationary line raises it and is never vetoed), runs of samples above
    ``extent`` that contain a sample above ``seed`` are zeroed, plus ``pad``
    samples on each side. For Gaussian noise a seed above 25 has probability
    5.7e-7 per sample, so false vetoes cost nothing; CW signals are far below
    the noise per sample and are never vetoed.
    """

    enabled: bool = True
    seed: float = 25.0
    extent: float = 9.0
    pad: int = 1
    #: Excess-power veto (ISSUES.md A9): windows of these lengths (s) whose mean
    #: z exceeds the chi-square level of false-alarm probability
    #: ``excess_false_alarm`` per band and window length are zeroed. It removes
    #: episodes of raised in-band noise lasting seconds to minutes (scattered
    #: light at 20-60 Hz) that no single sample shows. Empty: off.
    excess_scales: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0)
    excess_false_alarm: float = 1e-3

    def __post_init__(self) -> None:
        if not (0 < self.extent <= self.seed) or self.pad < 0 or int(self.pad) != self.pad:
            raise ValueError("VetoParams: need 0 < extent <= seed and a whole pad >= 0")
        object.__setattr__(self, "excess_scales", tuple(float(t) for t in self.excess_scales))
        if any(t <= 0 for t in self.excess_scales) or not (0 < self.excess_false_alarm < 1):
            raise ValueError("VetoParams: excess_scales must be positive and excess_false_alarm in (0, 1)")


#: Injection bits that are never allowed in ``dq_flags``: CW injections are on
#: almost all the time, so requiring their absence would remove the data.
FORBIDDEN_FLAGS = ("NO_CW_HW_INJ",)


@dataclass
class PipelineConfig:
    """Everything one pipeline run needs. Saved in every product."""

    data_dir: Path
    out_dir: Path
    ifo: str
    bands: list[BandSpec]
    gps_start: float | None = None
    gps_end: float | None = None
    epoch: int = DEFAULT_EPOCH
    #: DQ and injection bits that must all pass (ISSUES.md C9, A10). ``DATA``
    #: is the CW baseline; stochastic and detchar hardware injections add
    #: broadband noise or sinusoids for minutes, so their time is removed.
    #: Add ``"BURST_CAT2"``, ``"NO_BURST_HW_INJ"``, ``"NO_CBC_HW_INJ"`` as needed.
    dq_flags: tuple[str, ...] = ("DATA", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ")
    gating: GateParams = field(default_factory=GateParams)
    channelizer: ChannelizerParams = field(default_factory=ChannelizerParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    veto: VetoParams = field(default_factory=VetoParams)
    #: Seconds of neighbouring data each file's glitch search sees.
    gate_context: int = 64
    #: Scratch space: each run makes its own uniquely named folder in it
    #: (default ``out_dir/_work``) and deletes only that folder.
    work_dir: Path | None = None
    keep_work: bool = False
    overwrite: bool = False
    #: Skip bands whose product already exists (resume an interrupted job).
    skip_existing: bool = False
    #: Worker processes for conditioning, channelizing and the per-band steps.
    workers: int = 1
    #: Storage type of the conditioned strain in the scratch folder.
    work_dtype: str = "float64"
    #: Channelizer output held at once, GB (bands are processed in chunks).
    chunk_gb: float = 8.0
    #: DQ-good seconds with NaN/inf or constant strain: "raise" (default, ISSUES.md
    #: E5) or "drop" them (treated as DQ-bad, counted in the products).
    bad_strain: str = "raise"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
        if self.work_dir is not None:
            self.work_dir = Path(self.work_dir)
        if self.ifo not in DETECTORS:
            raise ValueError(f"detector {self.ifo!r} is not one of {DETECTORS}")
        if not self.bands:
            raise ValueError("at least one band is required")
        self.bands = [b if isinstance(b, BandSpec) else BandSpec(*b) for b in self.bands]
        if len(set(self.bands)) != len(self.bands):
            raise ValueError("duplicate bands")
        if int(self.epoch) != self.epoch:
            raise ValueError("epoch must be an integer GPS second")
        self.epoch = int(self.epoch)
        self.dq_flags = tuple(self.dq_flags)
        if "DATA" not in self.dq_flags:
            raise ValueError("dq_flags must include DATA (strain is NaN where DATA is off)")
        bad = [f for f in self.dq_flags if f in FORBIDDEN_FLAGS]
        if bad:
            raise ValueError(f"dq_flags must not include {bad}: CW hardware injections are on almost all the time")
        lowest = min(float(b.fmin) for b in self.bands)
        if lowest < self.gating.hp_pass:
            raise ValueError(
                f"fmin {lowest} Hz is below the high-pass passband edge {self.gating.hp_pass} Hz"
            )
        if not float(self.gate_context).is_integer():
            raise ValueError(f"gate_context must be whole seconds, got {self.gate_context}")
        self.gate_context = int(self.gate_context)
        min_ctx = int(np.ceil(max(16.0, 2 * self.gating.white_dur + self.gating.psd_seg)))
        if self.gate_context < min_ctx:
            raise ValueError(f"gate_context must be at least {min_ctx} s (whitening filter edges), got {self.gate_context}")
        T = self.channelizer.kernel_length + self.channelizer.hop
        for b in self.bands:
            if (b.fmin * T).denominator != 1:
                raise ValueError(
                    f"band {b.label()}: fmin x block length {T} s is not a whole number of cycles; "
                    "use hop and kernel_length that are multiples of 1800 s"
                )
            if (Fraction(self.noise.segment) * b.fs).denominator != 1:
                raise ValueError(f"noise segment {self.noise.segment} s is not a whole number of samples at {float(b.fs)} Hz")
        if int(self.workers) != self.workers or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        self.workers = int(self.workers)
        if self.work_dtype not in ("float32", "float64"):
            raise ValueError("work_dtype must be 'float32' or 'float64'")
        if not self.chunk_gb > 0:
            raise ValueError("chunk_gb must be positive")
        if self.bad_strain not in ("raise", "drop"):
            raise ValueError("bad_strain must be 'raise' or 'drop'")

    @property
    def work_path(self) -> Path:
        return self.work_dir if self.work_dir is not None else self.out_dir / "_work"

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form (bands as ``[fmin, band]`` strings of exact fractions)."""
        return {
            "data_dir": str(self.data_dir),
            "out_dir": str(self.out_dir),
            "ifo": self.ifo,
            "bands": [b.to_list() for b in self.bands],
            "gps_start": self.gps_start,
            "gps_end": self.gps_end,
            "epoch": self.epoch,
            "dq_flags": list(self.dq_flags),
            "gating": dataclasses.asdict(self.gating),
            "channelizer": dataclasses.asdict(self.channelizer),
            "noise": dataclasses.asdict(self.noise),
            "veto": dataclasses.asdict(self.veto),
            "gate_context": self.gate_context,
            "work_dtype": self.work_dtype,
            "bad_strain": self.bad_strain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        d = dict(d)
        d["bands"] = [_band_from_list(b) for b in d["bands"]]
        if "gating" in d:
            d["gating"] = GateParams(**d["gating"])
        if "channelizer" in d:
            d["channelizer"] = ChannelizerParams(**d["channelizer"])
        if "noise" in d:
            d["noise"] = NoiseParams(**d["noise"])
        if "veto" in d:
            d["veto"] = VetoParams(**d["veto"])
        return cls(**d)

    @classmethod
    def from_json(cls, path: str | Path) -> PipelineConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _band_from_list(b: Any) -> BandSpec:
    """BandSpec from ``[fmin, band]`` or ``[fmin, band, usable_lo, usable_hi]`` (or a BandSpec)."""
    if isinstance(b, BandSpec):
        return b
    b = list(b)
    if len(b) == 2:
        return BandSpec(b[0], b[1])
    if len(b) == 4:
        return BandSpec(b[0], b[1], (float(b[2]), float(b[3])))
    raise ValueError(f"a band is [fmin, band] or [fmin, band, usable_lo, usable_hi], got {b!r}")
