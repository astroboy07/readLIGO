"""Broadband conditioning of 4 kHz strain: DQ stretches, zero-phase high-pass,
edge tapers and transient gating (ISSUES.md C1, C4, C6).

Each contiguous run of GWOSC files is processed as one series, so file
boundaries are never data edges. Per run:

1. **DQ stretches.** Runs of seconds where every requested DQ bit passes.
   Stretches shorter than ``min_stretch`` are dropped. Everything outside the
   kept stretches is zero.
2. **Zero-phase high-pass**, per stretch: a linear-phase Kaiser FIR
   (stopband <= 10 Hz at 120 dB, passband >= 15 Hz, ripple ~1e-6) applied
   centred, with odd-reflection padding at the stretch ends so no step is
   filtered. GWOSC strain is dominated by a 7.93 Hz line and 8-12 Hz noise
   about 110 dB above the 100 Hz floor. Without this filter, every window
   (gap edge, gate) smears that content into the analysis bands.
3. **Edge taper**: a half-Hann ramp of ``edge_taper`` s at both ends of each
   stretch.
4. **Gating.** Each file is searched for loud transients with
   ``gate_context`` s of neighbouring data: median-Welch PSD, FIR whitening
   over 20-1700 Hz, robust sigma, and a dual threshold (seed at
   ``threshold`` sigma, extent at ``low_threshold`` sigma). Each transient is
   zeroed from ``gate_pad`` s before its first to ``gate_pad`` s after its
   last crossing, with ``gate_taper`` s Hann tapers outside (inverse Tukey).
   One gate list (GPS) then applies to every band.

The total multiplicative window of the strain is
``W(t) = DQ stretch x edge taper x gates``. The conditioned strain written per
run is ``high-passed strain x DQ stretch x edge taper``; gates are applied
when the channelizer reads it (:class:`ConditionedStrain`), so the gate list
can be stored and re-applied exactly.

Defaults are the values validated on the 56 local O3a H1 files (see
``docs/cleaning_report.md``): 274 gates at T = 20 (5.25 per hour), 0.23 % of
DQ-good time lost, no gate on coloured Gaussian noise, software CWs
recovered with amplitude 1.000 +- 0.01 through the gates.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import numpy.typing as npt
from scipy import signal

from .inventory import RawFile

__all__ = [
    "ConditionedRun",
    "ConditionedStrain",
    "Gate",
    "GateParams",
    "RunData",
    "condition_run",
    "design_highpass",
    "block_psd",
    "check_strain",
    "detect_gates",
    "detect_gates_run",
    "edge_ramp",
    "gate_window",
    "good_stretches",
    "highpass_zero_phase",
    "output_window",
    "whitening_filter",
]

logger = logging.getLogger(__name__)

_ROBUST_SIGMA = 1.482602218505602  # 1 / Phi^-1(3/4): sigma of a Gaussian from median |x|


@dataclass(frozen=True)
class GateParams:
    """Settings of the broadband conditioning. Defaults are the recommended ones."""

    # High-pass (zero phase). "fir": Kaiser FIR with stopband edge hp_stop,
    # passband edge hp_pass and attenuation hp_atten_db. "none": no filter.
    hp_type: Literal["fir", "none"] = "fir"
    hp_stop: float = 10.0
    hp_pass: float = 15.0
    hp_atten_db: float = 120.0
    # DQ stretches
    edge_taper: float = 8.0  # s, half-Hann ramp at each stretch end (ISSUES.md A5)
    min_stretch: float = 16.0  # s, shorter DQ-good stretches are dropped
    # PSD and whitening (detection only)
    psd_seg: float = 4.0  # s, Welch segment length
    white_flo: float = 20.0  # Hz
    white_fhi: float = 1700.0  # Hz (GWOSC anti-alias roll-off starts above)
    white_edge: float = 2.0  # Hz, cosine roll-on/off outside [flo, fhi]
    white_dur: float = 4.0  # s, whitening FIR length
    # Detection
    detect: bool = True  # False: high-pass and tapers only, no gates
    threshold: float = 20.0  # seed threshold, robust sigma
    low_threshold: float | None = 6.0  # extent threshold; None = single threshold
    cluster_window: float = 0.5  # s
    # Gate shape
    gate_pad: float = 0.25  # s zeroed beyond the first/last crossing
    # 4 s, not 0.25 s: a short taper spreads strong lines (violin modes) over
    # +-1 Hz at every gate (ISSUES.md A5); 4 s confines that to ~0.1 Hz.
    gate_taper: float = 4.0  # s, Hann taper outside the zeroed region
    # Whitening PSD of each file: the lower of its own median PSD and the
    # run's median PSD, so a file full of glitches (a "storm") is still
    # whitened by quiet-time noise and its glitches stand out (ISSUES.md A12).
    ref_psd: bool = True

    def __post_init__(self) -> None:
        def bad(msg: str) -> None:
            raise ValueError(f"GateParams: {msg}")

        if self.hp_type not in ("fir", "none"):
            bad(f"hp_type must be 'fir' or 'none', not {self.hp_type!r}")
        if self.hp_type == "fir" and not (0 < self.hp_stop < self.hp_pass):
            bad(f"need 0 < hp_stop < hp_pass, got {self.hp_stop}, {self.hp_pass}")
        if self.hp_atten_db <= 0:
            bad("hp_atten_db must be positive")
        if self.edge_taper < 0 or self.gate_taper < 0 or self.gate_pad < 0 or self.cluster_window < 0:
            bad("edge_taper, gate_taper, gate_pad and cluster_window must be >= 0")
        if self.min_stretch <= 0 or self.min_stretch < 2 * self.edge_taper:
            bad(f"min_stretch ({self.min_stretch} s) must be positive and >= 2 x edge_taper ({self.edge_taper} s)")
        if self.psd_seg <= 0 or self.white_dur <= 0 or self.white_edge <= 0:
            bad("psd_seg, white_dur and white_edge must be positive")
        if not (0 < self.white_flo < self.white_fhi):
            bad(f"need 0 < white_flo < white_fhi, got {self.white_flo}, {self.white_fhi}")
        if self.threshold <= 0:
            bad("threshold must be positive")
        if self.low_threshold is not None and not (0 < self.low_threshold <= self.threshold):
            bad("low_threshold must be None or in (0, threshold]")

    def as_attrs(self) -> dict[str, str | int | float | bool]:
        """Flat dict for HDF5 attrs (prefix ``gating_``; None is stored as "none")."""
        return {f"gating_{k}": ("none" if v is None else v) for k, v in dataclasses.asdict(self).items()}


@dataclass(frozen=True)
class Gate:
    """One gated transient. Sample indices count 1/fs steps from the epoch."""

    k_start: int  # first zeroed sample
    k_end: int  # one past the last zeroed sample
    k_peak: int  # sample of the maximum statistic
    peak: float  # maximum statistic, robust sigma
    fs: int
    epoch: int

    @property
    def gps_start(self) -> float:
        return self.epoch + self.k_start / self.fs

    @property
    def gps_end(self) -> float:
        return self.epoch + self.k_end / self.fs

    @property
    def gps_peak(self) -> float:
        return self.epoch + self.k_peak / self.fs


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def good_stretches(good_1hz: npt.NDArray[np.bool_], min_stretch: float = 0.0) -> list[tuple[int, int]]:
    """``[start, stop)`` second offsets of runs of True, dropping runs shorter than *min_stretch*."""
    d = np.diff(np.concatenate(([0], np.asarray(good_1hz, dtype=np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops) if (b - a) >= min_stretch]


def design_highpass(p: GateParams, fs: int) -> npt.NDArray[np.float64] | None:
    """Taps of the zero-phase high-pass (odd length, symmetric), or None."""
    if p.hp_type == "none":
        return None
    if p.hp_type != "fir":
        raise ValueError(f"unknown hp_type {p.hp_type!r}")
    nyq = fs / 2.0
    numtaps, beta = signal.kaiserord(p.hp_atten_db, (p.hp_pass - p.hp_stop) / nyq)
    numtaps |= 1  # odd length: type-I linear phase, integer delay
    cutoff = 0.5 * (p.hp_stop + p.hp_pass)
    return signal.firwin(numtaps, cutoff, window=("kaiser", beta), pass_zero=False, fs=fs)


def highpass_zero_phase(
    read: callable,  # read(i0, i1) -> samples [i0, i1) of the stretch
    n: int,
    taps: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
    *,
    chunk: int = 1 << 23,
) -> None:
    """Centred FIR of one stretch of *n* samples, in chunks, into *out* (length n).

    The stretch is extended at both ends by odd reflection (``2 x[0] - x[k]``),
    which is continuous in value and slope. The result equals
    ``oaconvolve(np.pad(x, half, "reflect", reflect_type="odd"), taps, "valid")``
    but needs only ``chunk + len(taps)`` samples in memory.
    """
    half = (len(taps) - 1) // 2
    if n <= half:
        raise ValueError(f"stretch of {n} samples is shorter than the filter half-length {half}")
    for i in range(0, n, chunk):
        j = min(n, i + chunk)
        lo, hi = max(0, i - half), min(n, j + half)
        seg = np.asarray(read(lo, hi), dtype=np.float64)
        parts = []
        if i - half < 0:  # odd reflection about x[0]
            k = half - i
            x0 = seg[0] if lo == 0 else read(0, 1)[0]
            first = seg if lo == 0 else read(0, k + 1)
            parts.append(2.0 * x0 - first[k:0:-1])
        parts.append(seg)
        if j + half > n:  # odd reflection about x[n-1]
            k = j + half - n
            last = seg if hi == n else read(n - k - 1, n)
            parts.append(2.0 * last[-1] - last[-2 : -k - 2 : -1])
        xp = np.concatenate(parts) if len(parts) > 1 else seg
        out[i:j] = signal.oaconvolve(xp, taps, mode="valid")


def edge_ramp(m: int) -> npt.NDArray[np.float64]:
    """Rising half-Hann ramp of *m* samples, ``sin^2(pi/2 (k + 1/2)/m)``."""
    return np.sin(0.5 * np.pi * (np.arange(m) + 0.5) / m) ** 2


def gate_window(n: int, zero_ranges: Sequence[tuple[int, int]], taper: int) -> npt.NDArray[np.float64]:
    """Product of inverse-Tukey windows over samples ``0..n-1``.

    Each ``(i0, i1)`` in *zero_ranges* (sample indices, may lie partly
    outside ``[0, n)``) is zeroed, with a falling Hann taper on
    ``[i0 - taper, i0)`` and a rising one on ``[i1, i1 + taper)``.
    """
    w = np.ones(n)
    ramp = edge_ramp(taper) if taper > 0 else np.empty(0)
    for i0, i1 in zero_ranges:
        a, b = max(0, i0), min(n, i1)
        if b > a:
            w[a:b] = 0.0
        if taper > 0:
            lo = max(0, i0 - taper)
            if i0 > lo and lo < n:
                hi = min(i0, n)
                w[lo:hi] *= ramp[::-1][lo - (i0 - taper) : hi - (i0 - taper)]
            hi = min(n, i1 + taper)
            if hi > max(i1, 0):
                lo = max(i1, 0)
                w[lo:hi] *= ramp[lo - i1 : hi - i1]
    return w


def _median_bias(n: int) -> float:
    """Median/mean of n chi2(2) samples (the scipy.signal.welch correction)."""
    ii_2 = 2 * np.arange(1.0, (n - 1) // 2 + 1)
    return float(1 + np.sum(1.0 / (ii_2 + 1) - 1.0 / ii_2))


def whitening_filter(
    freqs: npt.NDArray[np.float64], psd: npt.NDArray[np.float64], p: GateParams, fs: int
) -> npt.NDArray[np.float64]:
    """Symmetric, odd-length FIR with gain 1/sqrt(PSD) on [flo, fhi], zero outside.

    Cosine roll-on/off of ``white_edge`` Hz, truncated to ``white_dur`` s with
    a Hann window (inverse spectrum truncation). Unnormalised.
    """
    nw = int(p.white_dur * fs)
    f = np.fft.rfftfreq(nw, 1.0 / fs)
    s = np.interp(f, freqs, psd)
    band = np.zeros_like(f)
    band[(f >= p.white_flo) & (f <= p.white_fhi)] = 1.0
    lo = (f > p.white_flo - p.white_edge) & (f < p.white_flo)
    band[lo] = 0.5 * (1 - np.cos(np.pi * (f[lo] - (p.white_flo - p.white_edge)) / p.white_edge))
    hi = (f > p.white_fhi) & (f < p.white_fhi + p.white_edge)
    band[hi] = 0.5 * (1 + np.cos(np.pi * (f[hi] - p.white_fhi) / p.white_edge))
    amp = np.where(s > 0, band / np.sqrt(np.where(s > 0, s, 1.0)), 0.0)
    h = np.roll(np.fft.irfft(amp, n=nw), nw // 2)
    taps = np.concatenate((h, h[:1]))
    return taps * signal.windows.hann(len(taps), sym=True)


def block_psd(
    x: npt.NDArray[np.float64],
    stretches: Sequence[tuple[int, int]],
    p: GateParams,
    fs: int,
) -> tuple[npt.NDArray[np.float64], int] | None:
    """Median-Welch PSD of the stretch interiors of *x*, and the segment count.

    Each stretch loses ``edge_taper + 2`` s at both ends (tapers and filter
    edges). Stretches too short for one ``psd_seg`` segment after that are
    skipped; None if none is left.
    """
    nper = int(p.psd_seg * fs)
    guard = int(round((p.edge_taper + 2.0) * fs))
    cols = []
    for a, b in stretches:
        if b - a - 2 * guard < nper:  # also keeps a + guard < b - guard (ISSUES.md A3)
            continue
        _, _, sxx = signal.spectrogram(
            x[a + guard : b - guard], fs=fs, window="hann", nperseg=nper, noverlap=nper // 2,
            detrend=False, scaling="density", mode="psd",
        )
        cols.append(sxx)
    if not cols:
        return None
    allcols = np.concatenate(cols, axis=1)
    return np.median(allcols, axis=1) / _median_bias(allcols.shape[1]), int(allcols.shape[1])


def detect_gates(
    x: npt.NDArray[np.float64],
    stretches: Sequence[tuple[int, int]],
    p: GateParams,
    fs: int,
    *,
    psd: npt.NDArray[np.float64] | None = None,
    valid: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int, int, float]], dict[str, float]]:
    """Find transients in one block of conditioned strain.

    Parameters
    ----------
    x
        High-passed, edge-tapered strain (zero outside *stretches*).
    stretches
        ``[start, stop)`` sample ranges of the DQ-good stretches in *x*.
    psd
        Whitening PSD on ``rfftfreq(psd_seg * fs)``; default: :func:`block_psd` of *x*.
    valid
        Sample range ``[a, b)`` where detections may lie (default: all of *x*).

    Returns
    -------
    clusters
        ``(first, last, peak_index, peak)`` per transient, in samples of *x*.
    info
        ``sigma`` (robust sigma of the whitened series) and ``n_psd_segments``.
    """
    nper = int(p.psd_seg * fs)
    n_seg = 0
    if psd is None:
        est = block_psd(x, stretches, p, fs)
        if est is None:
            return [], {"sigma": float("nan"), "n_psd_segments": 0}
        psd, n_seg = est
    taps = whitening_filter(np.fft.rfftfreq(nper, 1.0 / fs), psd, p, fs)
    hw = signal.oaconvolve(x, taps, mode="same")

    allowed = np.zeros(len(x), dtype=bool)
    interior = np.zeros(len(x), dtype=bool)
    half_w = int(p.white_dur * fs / 2)
    g_edge = int(round(p.edge_taper * fs))
    for a, b in stretches:
        allowed[a:b] = True
        lo, hi = a + g_edge + half_w, b - g_edge - half_w
        if hi > lo:  # a negative bound would wrap around (ISSUES.md A3)
            interior[lo:hi] = True
    if valid is not None:
        allowed[: max(0, valid[0])] = False
        allowed[max(0, valid[1]) :] = False
        interior &= allowed
    if not interior.any():
        interior = allowed if allowed.any() else np.ones(len(x), dtype=bool)
    sigma = _ROBUST_SIGMA * float(np.median(np.abs(hw[interior])))
    if not sigma > 0:
        return [], {"sigma": sigma, "n_psd_segments": n_seg}
    stat = np.abs(hw) / sigma

    lo_thr = p.threshold if p.low_threshold is None else min(p.low_threshold, p.threshold)
    idx = np.flatnonzero((stat > lo_thr) & allowed)
    clusters: list[tuple[int, int, int, float]] = []
    if len(idx):
        split = np.flatnonzero(np.diff(idx) > p.cluster_window * fs) + 1
        for grp in np.split(idx, split):
            k = grp[np.argmax(stat[grp])]
            if stat[k] > p.threshold:
                clusters.append((int(grp[0]), int(grp[-1]), int(k), float(stat[k])))
    return clusters, {"sigma": sigma, "n_psd_segments": n_seg}


# ----------------------------------------------------------------------------
# Runs of contiguous files
# ----------------------------------------------------------------------------
class RunData:
    """Read access to one run of back-to-back GWOSC files as one series.

    Parameters
    ----------
    files
        Sample-continuous files (one element of :func:`inventory.contiguous_runs`).
    dq_flags
        Bits that must all pass for a second to be good: GWOSC DQ bits
        (``DATA``, ``BURST_CAT2``, ...) and/or hardware-injection bits
        (``NO_STOCH_HW_INJ``, ...; True where that injection type is off).
    """

    def __init__(self, files: Sequence[RawFile], dq_flags: Sequence[str]) -> None:
        from .io import read_gwosc

        if not files:
            raise ValueError("empty run")
        for a, b in zip(files, files[1:]):
            if a.gps_end != b.gps_start:
                raise ValueError(f"{a.path.name} and {b.path.name} are not contiguous")
        self.files = list(files)
        self.gps_start = files[0].gps_start
        self.gps_end = files[-1].gps_end
        good, cw_off, fs, channels = [], [], set(), set()
        for f in files:
            g = read_gwosc(f.path, read_strain=False)
            bits = {**g.injections, **g.dq}
            missing = [k for k in dq_flags if k not in bits]
            if missing:
                raise ValueError(f"{f.path.name}: DQ/injection bits {missing} not in file")
            good.append(np.logical_and.reduce([bits[k] for k in dq_flags]))
            cw_off.append(g.injections.get("NO_CW_HW_INJ", np.zeros(g.duration, bool)))
            fs.add(g.fs)
            channels.add(g.strain_channel)
        if len(fs) != 1:
            raise ValueError(f"files in one run have different sample rates {sorted(fs)}")
        self.fs: int = fs.pop()
        self.strain_channels = sorted(channels)
        self.good = np.concatenate(good)
        #: True where CW hardware injections were off (GWOSC NO_CW_HW_INJ bit).
        self.cw_injection_off = np.concatenate(cw_off)

    @property
    def duration(self) -> int:
        return self.gps_end - self.gps_start

    @property
    def n_samples(self) -> int:
        return self.duration * self.fs

    def read(self, i0: int, i1: int) -> npt.NDArray[np.float64]:
        """Strain samples ``[i0, i1)`` counted from the run start."""
        out = np.empty(i1 - i0)
        for f in self.files:
            f0 = (f.gps_start - self.gps_start) * self.fs
            f1 = f0 + f.duration * self.fs
            a, b = max(i0, f0), min(i1, f1)
            if b > a:
                with h5py.File(f.path, "r") as h:
                    out[a - i0 : b - i0] = h["strain/Strain"][a - f0 : b - f0]
        return out


def check_strain(path: str | Path, good_1hz: npt.NDArray[np.bool_], fs: int) -> dict[str, npt.NDArray[np.int64]]:
    """Second offsets in one file where *good_1hz* is True but the strain is unusable.

    Returns ``{"nonfinite": ..., "constant": ...}``: seconds with a NaN or
    infinite sample, and seconds whose samples are all equal (a dead or
    zero-filled channel, ISSUES.md A12).
    """
    with h5py.File(path, "r") as h:
        x = h["strain/Strain"][()]
    x = x.reshape(-1, fs)
    good = np.asarray(good_1hz, dtype=bool)
    finite = np.all(np.isfinite(x), axis=1)
    nonfinite = good & ~finite
    with np.errstate(invalid="ignore"):
        constant = good & finite & (np.max(x, axis=1) == np.min(x, axis=1))
    return {"nonfinite": np.flatnonzero(nonfinite), "constant": np.flatnonzero(constant)}


@dataclass
class ConditionedRun:
    """High-passed, edge-tapered strain of one run (zero outside the stretches)."""

    gps_start: int
    fs: int
    path: Path
    stretches: list[tuple[int, int]]  # GPS seconds [start, stop)
    dropped: list[tuple[int, int]]  # DQ-good stretches shorter than min_stretch
    files: list[RawFile]

    def strain(self) -> npt.NDArray[np.float64]:
        return np.load(self.path, mmap_mode="r")

    @property
    def gps_end(self) -> int:
        return self.gps_start + self.strain().shape[0] // self.fs


def condition_run(
    run: RunData, p: GateParams, out_path: str | Path, *, dtype: npt.DTypeLike = np.float64
) -> ConditionedRun:
    """Write the high-passed, edge-tapered strain of *run* to an ``.npy`` file.

    The filter runs in float64; *dtype* is the storage type (float32 halves
    the scratch space; its rounding, 6e-8 of the broadband rms, is ~1e-12 of
    the in-band noise power).

    Raises
    ------
    ValueError
        If a DQ-good sample is NaN or infinite (ISSUES.md E5).
    """
    fs = run.fs
    taps = design_highpass(p, fs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(run.n_samples,))
    kept = good_stretches(run.good, p.min_stretch)
    every = good_stretches(run.good)
    dropped = [s for s in every if s not in kept]
    m = int(round(p.edge_taper * fs))
    ramp = edge_ramp(m) if m > 0 else np.empty(0)
    for a_s, b_s in kept:
        a, b = a_s * fs, b_s * fs
        n = b - a

        def read(i0: int, i1: int, a: int = a) -> npt.NDArray[np.float64]:
            x = run.read(a + i0, a + i1)
            if not np.all(np.isfinite(x)):
                bad = np.flatnonzero(~np.isfinite(x))[0] + a + i0
                raise ValueError(
                    f"non-finite strain in a DQ-good second, GPS {run.gps_start + bad // fs}"
                )
            return x

        if taps is None:
            for i in range(0, n, 1 << 23):
                out[a + i : a + min(n, i + (1 << 23))] = read(i, min(n, i + (1 << 23)))
        else:
            highpass_zero_phase(read, n, taps, out[a:b])
        mm = min(m, n // 2)
        if mm > 0:
            r = ramp[:mm] if mm == m else edge_ramp(mm)
            out[a : a + mm] = out[a : a + mm] * r
            out[b - mm : b] = out[b - mm : b] * r[::-1]
    out.flush()
    del out
    to_gps = run.gps_start
    return ConditionedRun(
        gps_start=run.gps_start,
        fs=fs,
        path=out_path,
        stretches=[(a + to_gps, b + to_gps) for a, b in kept],
        dropped=[(a + to_gps, b + to_gps) for a, b in dropped],
        files=run.files,
    )


def detect_gates_run(
    cond: ConditionedRun, p: GateParams, epoch: int, *, context: int = 64
) -> tuple[list[Gate], list[dict[str, float]]]:
    """Gates of one conditioned run, searching file by file with *context* s on each side.

    Each file is searched in a block reaching *context* s into its
    neighbours. Within ``white_dur`` s of a block end that cuts through live
    data the whitened series is unreliable, so detections there are dropped;
    the neighbouring block sees that time in its interior. The transients
    found by all blocks are then merged into one interval list (overlapping
    or within ``cluster_window``), so a transient on a file boundary is gated
    once and in full, and one longer than the context is not cut off
    (ISSUES.md A12). With ``ref_psd`` each block is whitened by the lower of
    its own and the run's median PSD.

    Returns
    -------
    gates
        Sorted by start.
    blocks
        Per file: ``gps_start``, ``sigma``, ``n_psd_segments``, ``n_gates``.
    """
    fs = cond.fs
    x_all = cond.strain()
    if not p.detect:
        return [], []
    pad = int(round(p.gate_pad * fs))
    edge_zone = int(round(p.white_dur * fs))
    g0 = cond.gps_start
    run_end = cond.gps_end

    defs = []
    for f in cond.files:
        lo = max(g0, f.gps_start - context)
        hi = min(run_end, f.gps_end + context)
        st = [(max(a, lo), min(b, hi)) for a, b in cond.stretches if b > lo and a < hi]
        defs.append((f, lo, hi, [((a - lo) * fs, (b - lo) * fs) for a, b in st]))

    def block(lo: int, hi: int) -> npt.NDArray[np.float64]:
        return np.asarray(x_all[(lo - g0) * fs : (hi - g0) * fs], dtype=np.float64)

    psds: list[tuple[npt.NDArray[np.float64], int] | None] = []
    for _, lo, hi, st in defs:
        psds.append(block_psd(block(lo, hi), st, p, fs) if st else None)
    have = [q[0] for q in psds if q is not None]
    ref = np.median(np.stack(have), axis=0) if (p.ref_psd and len(have) >= 3) else None

    found: list[tuple[int, int, int, float]] = []  # [start, end) in epoch samples, peak sample, peak
    blocks: list[dict[str, float]] = []
    for (f, lo, hi, st), est in zip(defs, psds):
        info: dict[str, float] = {"gps_start": f.gps_start, "sigma": float("nan"), "n_psd_segments": 0, "n_gates": 0}
        blocks.append(info)
        if est is None:
            continue
        psd = est[0] if ref is None else np.minimum(est[0], ref)
        x = block(lo, hi)
        a_ok = edge_zone if lo > g0 else 0
        b_ok = len(x) - edge_zone if hi < run_end else len(x)
        clusters, det = detect_gates(x, st, p, fs, psd=psd, valid=(a_ok, b_ok))
        info.update(det)
        info["n_psd_segments"] = est[1]
        k_lo = (lo - epoch) * fs
        for first, last, k, peak in clusters:
            found.append((k_lo + first, k_lo + last + 1, k_lo + k, peak))

    found.sort()
    merged: list[list[float]] = []
    join = int(round(p.cluster_window * fs))
    for a, b, k, peak in found:
        if merged and a <= merged[-1][1] + join:
            m = merged[-1]
            m[1] = max(m[1], b)
            if peak > m[3]:
                m[2], m[3] = k, peak
        else:
            merged.append([a, b, k, peak])
    gates = [Gate(int(a) - pad, int(b) + pad, int(k), float(pk), fs, epoch) for a, b, k, pk in merged]
    for info, (f, *_rest) in zip(blocks, defs):
        info["n_gates"] = sum(1 for g in gates if f.gps_start <= g.gps_peak < f.gps_end)
    return gates, blocks


class ConditionedStrain:
    """Reader of the conditioned, gated strain of all runs, for the channelizer.

    ``reader(gps_start, n)`` returns *n* samples from integer GPS *gps_start*:
    the conditioned strain of whichever run covers each sample (zero
    elsewhere) times the gate window. Each sample's value depends only on its
    GPS time, so any split of the processing gives identical output.
    """

    def __init__(self, runs: Sequence[ConditionedRun], gates: Sequence[Gate], p: GateParams, epoch: int) -> None:
        if not runs:
            raise ValueError("no runs")
        fs = {r.fs for r in runs}
        if len(fs) != 1:
            raise ValueError("runs have different sample rates")
        self.fs: int = fs.pop()
        self.runs = list(runs)
        self.epoch = epoch
        self.taper = int(round(p.gate_taper * self.fs))
        self._starts = np.array([g.k_start for g in gates], dtype=np.int64)
        self._ends = np.array([g.k_end for g in gates], dtype=np.int64)
        self._maxlen = int((self._ends - self._starts).max()) if len(gates) else 0

    def __call__(self, gps_start: int, n: int) -> npt.NDArray[np.float64]:
        fs = self.fs
        out = np.zeros(n)
        t1 = gps_start + n / fs
        for r in self.runs:
            if r.gps_end <= gps_start or r.gps_start >= t1:
                continue
            x = r.strain()
            a = max(0, (r.gps_start - gps_start) * fs)
            src0 = max(0, (gps_start - r.gps_start) * fs)
            m = min(n - a, x.shape[0] - src0)
            if m > 0:
                out[a : a + m] = x[src0 : src0 + m]
        if len(self._starts):
            k0 = (gps_start - self.epoch) * fs
            lo = np.searchsorted(self._starts, k0 - self.taper - self._maxlen)
            hi = np.searchsorted(self._starts, k0 + n + self.taper)
            sel = [
                (int(s - k0), int(e - k0))
                for s, e in zip(self._starts[lo:hi], self._ends[lo:hi])
                if e + self.taper > k0 and s - self.taper < k0 + n
            ]
            # apply each gate only on its own span (a full-length window costs a block-sized array)
            for i0, i1 in sel:
                a, b = max(0, i0 - self.taper), min(n, i1 + self.taper)
                out[a:b] *= gate_window(b - a, [(i0 - a, i1 - a)], self.taper)
        return out


# ----------------------------------------------------------------------------
# Window on the output grid
# ----------------------------------------------------------------------------
def output_window(
    stretches: Sequence[tuple[int, int]],
    gates: Sequence[Gate],
    p: GateParams,
    *,
    epoch: int,
    fs_out: float,
    n0: int,
    n: int,
    oversample: int = 64,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Weight and mask of each output sample from the strain window W(t).

    Output sample k (time ``t = epoch + (n0 + k)/fs_out``) stands for the slot
    ``[t - 1/(2 fs_out), t + 1/(2 fs_out))``. Its weight is the RMS of
    ``W(t) = stretch x edge taper x gates`` over that slot, and its mask is
    True only if W = 1 over the whole slot (no gap, taper or gate in it).

    Only slots that touch a stretch edge or a gate are evaluated finely
    (*oversample* points), and each gate only on its own slots, so the cost
    grows linearly with the number of edges and gates.
    """
    dt = 1.0 / fs_out
    t_first = epoch + n0 * dt
    win2 = np.zeros(n)
    edge = p.edge_taper
    tap = p.gate_taper

    def slot_index(t: float) -> int:
        return int(np.floor((t - t_first) / dt + 0.5))

    touched: list[np.ndarray] = []
    for a, b in stretches:
        # slots fully inside [a + edge, b - edge): weight 1
        i_lo = max(0, int(np.ceil((a + edge - t_first) / dt + 0.5)))
        i_hi = min(n, int(np.floor((b - edge - t_first) / dt - 0.5)) + 1)
        if i_hi > i_lo:
            win2[i_lo:i_hi] = 1.0
        for t_a, t_b in ((a, a + edge), (b - edge, b)):
            s0, s1 = max(0, slot_index(t_a)), min(n, slot_index(t_b) + 1)
            if s1 > s0:
                touched.append(np.arange(s0, s1))
    gate_slots = []
    for g in gates:
        s0, s1 = max(0, slot_index(g.gps_start - tap)), min(n, slot_index(g.gps_end + tap) + 1)
        gate_slots.append((s0, s1))
        if s1 > s0:
            touched.append(np.arange(s0, s1))
    mask = win2 == 1.0
    if touched:
        idx = np.unique(np.concatenate(touched))
        sub = (np.arange(oversample) + 0.5) / oversample - 0.5
        t = t_first + (idx[:, None] + sub[None, :]) * dt
        starts = np.array([s[0] for s in stretches], dtype=np.float64)
        stops = np.array([s[1] for s in stretches], dtype=np.float64)
        j = np.searchsorted(starts, t, side="right") - 1
        inside = (j >= 0) & (t < stops[np.clip(j, 0, None)])
        jj = np.clip(j, 0, None)
        if edge > 0:
            r = np.minimum((t - starts[jj]) / edge, (stops[jj] - t) / edge)
            w = np.where(inside, np.where(r < 1.0, np.sin(0.5 * np.pi * np.clip(r, 0, 1)) ** 2, 1.0), 0.0)
        else:
            w = inside.astype(np.float64)
        for g, (s0, s1) in zip(gates, gate_slots):
            if s1 <= s0:
                continue
            r0, r1 = np.searchsorted(idx, s0), np.searchsorted(idx, s1)
            ts = t[r0:r1]
            f = np.ones_like(ts)
            f[(ts >= g.gps_start) & (ts < g.gps_end)] = 0.0
            if tap > 0:
                fall = (ts < g.gps_start) & (ts >= g.gps_start - tap)
                f[fall] = np.sin(0.5 * np.pi * (g.gps_start - ts[fall]) / tap) ** 2
                rise = (ts >= g.gps_end) & (ts < g.gps_end + tap)
                f[rise] = np.sin(0.5 * np.pi * (ts[rise] - g.gps_end) / tap) ** 2
            w[r0:r1] *= f
        win2[idx] = np.mean(w**2, axis=1)
        mask[idx] = False
    return np.sqrt(win2).astype(np.float32), mask
