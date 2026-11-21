"""FFT channelizer: 4 kHz strain to one real, heterodyned series per band at
exactly 2B (ISSUES.md C1, C2, C3, C7, C8).

For each band ``[fmin, fmin + B]`` the output is sampled on the global grid
``t_n = t0 + n/(2B)`` and follows the Makefakedata_v5 convention that
rustyprune and pyloki read::

    y(t_n) = 2 Re[ z(t_n) exp(-2 pi i fmin (t_n - t0)) ]

where ``z`` is the positive-frequency band-pass of the input. A strain term
``A cos(2 pi f t + phi)`` in the band appears as
``A cos(2 pi (f - fmin)(t - t0) + phi + 2 pi f t0)``: same amplitude, one
global phase epoch t0, no per-file restart.

Method (overlap-save fast convolution)
--------------------------------------
* Kernel: a zero-phase complex band-pass ``h(t) = w(t) h_ideal(t)`` of length
  L, with ``w`` a continuous Kaiser window. Its response is
  ``H(f) = C(f - a0) - C(f - b0)``, with C the running integral of the
  Kaiser transform (closed form, integrated numerically to ~1e-15). Beta is
  tuned for 120 dB (|H| <= 1e-6 out of band).
* Blocks of ``T = hop + L`` seconds start at ``s_b = t0 + b hop - L/2``, so the
  block grid is anchored to the epoch. Each block is FFTed once for all
  bands. Per band: take the bins around the band, multiply by H, fold the
  bins outside [fmin, fmax] onto the band exactly as sampling at 2B aliases
  them, inverse-FFT onto the 2B grid (``M = 2 B T`` points), apply the
  global phase ``exp(-2 pi i fmin (s_b - t0))`` (computed exactly with
  fractions), take 2 Re and keep the central ``hop`` seconds.
* Every output sample comes from exactly one block, so the result is
  bit-identical however the time range is split into calls.

Constraints (checked): fmin and B are multiples of 1/1800 Hz, hop and L/2 are
multiples of 900 s, t0 is an integer GPS second. Then ``fmin T`` and ``2 B T``
are integers and every block starts on both grids.

Band edges (``edge``): a real series at exactly 2B has no room for a
transition band. ``"inside"`` (default) puts the transition inside the band:
|H| <= 1e-6 everywhere outside [fmin, fmax] (nothing aliases in) and the
response is flat to 1e-6 on ``[fmin + g, fmax - g]``, ``g = 2 v_s / L``
(2.18 mHz for L = 3600 s). ``"center"`` and ``"outside"`` are kept for
comparison only; both let content near the edges alias in.

Measured (``docs/cleaning_report.md``): in-band gain flat to 7e-7 beyond
2.2 mHz from the edges; aliasing below -123.6 dB from anywhere outside the
band (-260 dB for the 7.93 Hz line); phase error < 1e-9 rad; MFDv5 frames
reproduced to 2e-6 relative rms with t0 = tstart.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import fft as sfft
from scipy import special

from .config import FREQ_GRID_DEN, to_grid

__all__ = [
    "GRID_SECONDS",
    "BandDesign",
    "FFTChannelizer",
    "Reader",
    "design_band",
    "kaiser_cum",
    "kaiser_ft",
    "optimal_beta",
    "transition_halfwidth",
]

#: hop and L/2 must be multiples of this (1/(2B) = 900/k s for B = k/1800 Hz).
GRID_SECONDS = FREQ_GRID_DEN // 2

_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)

Reader = Callable[[int, int], npt.NDArray[np.float64]]
"""``reader(gps_start, n)``: n input samples from integer GPS *gps_start*,
zero where there are no data (see :class:`conditioning.ConditionedStrain`)."""


# ---------------------------------------------------------------------------
# Kaiser window in continuous time
# ---------------------------------------------------------------------------
def kaiser_beta(atten_db: float) -> float:
    """Kaiser's empirical beta for a stopband attenuation of *atten_db*."""
    a = float(atten_db)
    if a > 50:
        return 0.1102 * (a - 8.7)
    if a >= 21:
        return 0.5842 * (a - 21) ** 0.4 + 0.07886 * (a - 21)
    return 0.0


def kaiser_ft(v: npt.ArrayLike, beta: float) -> npt.NDArray[np.float64]:
    """Fourier transform of the continuous Kaiser window (``w(0) = 1``), with ``v = L f``.

    ``F(beta^2 - pi^2 v^2) / I0(beta)`` with ``F(q) = sinh(sqrt q)/sqrt q`` for
    q > 0 and ``sin(sqrt -q)/sqrt -q`` for q < 0. Integrates to 1 over v.
    """
    v = np.asarray(v, dtype=np.float64)
    q = beta * beta - (np.pi * v) ** 2
    out = np.empty_like(q)
    pos, neg = q > 1e-8, q < -1e-8
    mid = ~(pos | neg)
    z = np.sqrt(q[pos])
    out[pos] = 0.5 * (np.exp(z - beta) - np.exp(-z - beta)) / z / special.i0e(beta)
    zn = np.sqrt(-q[neg])
    out[neg] = np.sin(zn) / zn * np.exp(-beta) / special.i0e(beta)
    out[mid] = (1.0 + q[mid] / 6.0) * np.exp(-beta) / special.i0e(beta)
    return out


def kaiser_cum(v: npt.ArrayLike, beta: float) -> npt.NDArray[np.float64]:
    """Running integral ``C(v)`` of :func:`kaiser_ft`: 0 at -inf, 1/2 at 0, 1 at +inf.

    16-point Gauss-Legendre on sub-intervals of at most 0.25 in v; accurate to
    ~1e-15.
    """
    v = np.asarray(v, dtype=np.float64)
    shape = v.shape
    vv = np.atleast_1d(v).ravel()
    av = np.abs(vv)
    order = np.argsort(av)
    pts = av[order]
    knots = np.union1d([0.0], np.union1d(np.arange(0.0, float(pts[-1]), 0.25), pts))
    a, b = knots[:-1], knots[1:]
    half, mid = 0.5 * (b - a), 0.5 * (b + a)
    nodes = mid[:, None] + half[:, None] * _GL_X[None, :]
    seg = (kaiser_ft(nodes, beta) * _GL_W[None, :]).sum(axis=1) * half
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.empty_like(av)
    s[order] = cum[np.searchsorted(knots, pts)]
    return (0.5 + np.sign(vv) * s).reshape(shape)


@functools.lru_cache(maxsize=None)
def transition_halfwidth(beta: float, atten_db: float) -> float:
    """Smallest ``v_s`` (units of 1/L) with ``|C(-v)| <= 10^(-A/20)`` for all ``v >= v_s``."""
    delta = 10.0 ** (-atten_db / 20.0)
    v = np.arange(0.0, 60.0 + beta, 1e-3)
    tail = np.abs(kaiser_cum(-v, beta))
    sup = np.maximum.accumulate(tail[::-1])[::-1]
    ok = np.flatnonzero(sup <= delta)
    if ok.size == 0:
        raise ValueError("attenuation not reached; increase beta")
    return float(v[ok[0]])


@functools.lru_cache(maxsize=None)
def optimal_beta(atten_db: float) -> float:
    """Kaiser beta with the narrowest transition at *atten_db* (12.3053 for 120 dB).

    Searched on a 0.01 grid from 0.3 below to 1.0 above Kaiser's formula.
    """
    kb = kaiser_beta(atten_db)
    best = (np.inf, kb)
    for b in np.arange(kb - 0.3, kb + 1.0, 0.01):
        try:
            v = transition_halfwidth(float(b), atten_db)
        except ValueError:
            continue
        if v < best[0]:
            best = (v, float(b))
    return round(best[1], 4)


# ---------------------------------------------------------------------------
# Per-band filter design
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BandDesign:
    """Response of one bandwidth's kernel on the block FFT grid.

    ``gain[i]`` is H at ``(kmin + i)/T`` Hz from fmin, times 1/N (amplitude
    preserving).
    """

    bandwidth: Fraction
    T: int
    L: int
    beta: float
    a0: float  # ideal edges relative to fmin (Hz)
    b0: float
    kmin: int
    M: int  # output points per block, 2 B T
    gain: npt.NDArray[np.float64] = field(repr=False)
    fold_index: npt.NDArray[np.int64] = field(repr=False)


def design_band(
    bandwidth: Fraction,
    T: int,
    L: int,
    atten_db: float = 120.0,
    edge: str = "inside",
    fold_frac: float = 0.5,
    fs_in: int = 4096,
) -> BandDesign:
    """Kernel response for a band of width *bandwidth* (exact, Hz).

    Bins up to ``fold_frac * B`` beyond each edge are kept and folded onto the
    band as sampling at 2B aliases them; content further out (|H| ~ 1e-8 for
    L = 3600 s, and falling) is dropped.
    """
    B = float(bandwidth)
    beta = optimal_beta(float(atten_db))
    half = transition_halfwidth(beta, float(atten_db)) / L
    if edge == "inside":
        a0, b0 = half, B - half
    elif edge == "center":
        a0, b0 = 0.0, B
    elif edge == "outside":
        a0, b0 = -half, B + half
    else:
        raise ValueError(f"edge must be inside, center or outside, not {edge!r}")
    if b0 - a0 <= 2 * half:
        raise ValueError(
            f"band {B} Hz is too narrow for a {L} s kernel: its transitions ({2 * half:.4g} Hz each) "
            "overlap; use a longer kernel"
        )
    BT = bandwidth * T
    if BT.denominator != 1:
        raise ValueError("B * T must be an integer")
    BT_i = int(BT)
    kfold = int(np.floor(fold_frac * BT_i))
    k = np.arange(-kfold, BT_i + kfold + 1)
    f = k / T
    H = kaiser_cum(L * (f - a0), beta) - kaiser_cum(L * (f - b0), beta)
    M = 2 * BT_i
    return BandDesign(
        bandwidth=bandwidth, T=T, L=L, beta=beta, a0=a0, b0=b0, kmin=-kfold, M=M,
        gain=H / (T * fs_in), fold_index=np.mod(k, M),
    )


# ---------------------------------------------------------------------------
# Channelizer
# ---------------------------------------------------------------------------
class FFTChannelizer:
    """Overlap-save FFT channelizer serving many bands from one FFT per block.

    Parameters
    ----------
    bands
        ``(fmin, band)`` pairs in Hz, on the 1/1800 Hz grid.
    t0
        Global epoch (integer GPS): output grid origin and heterodyne
        reference. Use the same t0 for every detector and band.
    L
        Kernel length (s), a multiple of 1800. Guard ``7.84 / L`` Hz at 120 dB.
    hop
        Output seconds per block, a multiple of 900. Block length ``T = hop + L``.
    atten_db, edge
        Stopband attenuation and transition placement (module docstring).
    fs_in
        Input sample rate (Hz).
    """

    def __init__(
        self,
        bands: Sequence[tuple[float | Fraction, float | Fraction]],
        t0: int,
        L: int = 3600,
        hop: int = 3600,
        atten_db: float = 120.0,
        edge: str = "inside",
        fs_in: int = 4096,
        fold_frac: float = 0.5,
        workers: int = 1,
    ) -> None:
        if int(t0) != t0:
            raise ValueError("t0 must be an integer GPS second")
        if L <= 0 or L % (2 * GRID_SECONDS):
            raise ValueError(f"L = {L} s must be a positive multiple of {2 * GRID_SECONDS} s")
        if hop <= 0 or hop % GRID_SECONDS:
            raise ValueError(f"hop = {hop} s must be a positive multiple of {GRID_SECONDS} s")
        self.t0 = int(t0)
        self.L = int(L)
        self.hop = int(hop)
        self.T = self.L + self.hop
        self.fs_in = int(fs_in)
        self.N = self.T * self.fs_in
        self.atten_db = float(atten_db)
        self.edge = edge
        self.fold_frac = float(fold_frac)
        self.workers = workers
        self.fmin: list[Fraction] = []
        self.B: list[Fraction] = []
        self.designs: dict[Fraction, BandDesign] = {}
        for fmin, band in bands:
            fm, B = to_grid(fmin, "fmin"), to_grid(band, "band")
            if B <= 0:
                raise ValueError(f"band ({fmin}, {band}) has non-positive width")
            if B not in self.designs:
                self.designs[B] = design_band(B, self.T, self.L, atten_db, edge, fold_frac, self.fs_in)
            d = self.designs[B]
            if (fm * self.T).denominator != 1:  # ISSUES.md A11: int() would truncate a half-integer
                raise ValueError(
                    f"fmin {float(fm)} Hz x block length {self.T} s is not a whole number of cycles; "
                    "use hop and kernel_length that are multiples of 1800 s"
                )
            lo = int(fm * self.T) + d.kmin
            if lo < 1 or lo + d.gain.size > self.N // 2:
                raise ValueError(f"band ({float(fm)}, {float(B)}) Hz is too close to 0 Hz or to Nyquist")
            self.fmin.append(fm)
            self.B.append(B)
        self.nbands = len(self.fmin)
        #: per band (fc, a1, a2, ...) of an extra smooth gain exp(-sum_k a_k u^k / 2), u = f - fc (Hz), or None
        self.shapes: list[tuple[float, ...] | None] = [None] * self.nbands

    def set_shape(self, i: int, shape: tuple[float, ...] | None) -> None:
        """Multiply band i's response by ``g(f) = exp(-sum_k a_k u^k / 2)``, ``u = f - fc``.

        With ``log S(f) = c + sum_k a_k u^k`` the band's smooth noise PSD, g
        flattens the noise in the band to ``S(fc)`` (in-band whitening; ISSUES.md
        A8). Every strain component at frequency f is scaled by g(f).
        """
        self.shapes[i] = None if shape is None else tuple(float(v) for v in shape)

    def shape_gain(self, i: int, f: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """The gain g of band i at absolute frequencies *f* (1 where no shape is set)."""
        f = np.asarray(f, dtype=np.float64)
        sh = self.shapes[i]
        if sh is None:
            return np.ones_like(f)
        u = f - sh[0]
        return np.exp(-0.5 * sum(a * u ** (k + 1) for k, a in enumerate(sh[1:])))

    # -- grid -----------------------------------------------------------------
    def fs_out(self, i: int) -> Fraction:
        return 2 * self.B[i]

    def samples_per_hop(self, i: int) -> int:
        return int(self.hop * self.fs_out(i))

    def guard(self, i: int) -> float:
        """Hz from each edge of band i beyond which the response is flat (edge="inside")."""
        d = self.designs[self.B[i]]
        return 2 * transition_halfwidth(d.beta, self.atten_db) / self.L

    def block_start(self, b: int) -> int:
        """GPS start of block b; its output covers ``[t0 + b hop, t0 + (b+1) hop)``."""
        return self.t0 + b * self.hop - self.L // 2

    def blocks_for(self, gps_start: float | Fraction, gps_end: float | Fraction) -> range:
        """Blocks whose output intersects ``[gps_start, gps_end)``."""
        b0 = math.floor((Fraction(gps_start) - self.t0) / self.hop)
        b1 = math.ceil((Fraction(gps_end) - self.t0) / self.hop)
        return range(b0, b1)

    def _epoch_phasor(self, i: int, b: int) -> complex:
        frac = (self.fmin[i] * (self.block_start(b) - self.t0)) % 1
        ang = -2.0 * np.pi * float(frac)
        return complex(np.cos(ang), np.sin(ang))

    # -- core -----------------------------------------------------------------
    def band_from_spectrum(self, X: npt.NDArray[np.complex128], i: int, b: int) -> npt.NDArray[np.float64]:
        """The ``hop * 2B`` output samples of band i from block b's spectrum X."""
        d = self.designs[self.B[i]]
        lo = int(self.fmin[i] * self.T) + d.kmin
        c = X[lo : lo + d.gain.size] * d.gain
        if self.shapes[i] is not None:
            c *= self.shape_gain(i, float(self.fmin[i]) + (d.kmin + np.arange(d.gain.size)) / self.T)
        c *= self._epoch_phasor(i, b)
        Y = np.bincount(d.fold_index, weights=c.real, minlength=d.M) + 1j * np.bincount(
            d.fold_index, weights=c.imag, minlength=d.M
        )
        y = 2.0 * (sfft.ifft(Y) * d.M).real
        n_lo = int(self.L // 2 * self.fs_out(i))
        return y[n_lo : n_lo + self.samples_per_hop(i)]

    def iter_blocks(
        self, reader: Reader, gps_start: float | Fraction, gps_end: float | Fraction
    ) -> Iterator[tuple[int, list[npt.NDArray[np.float64]]]]:
        """Yield ``(b, [band outputs])`` per block. All-zero blocks skip the FFT."""
        for b in self.blocks_for(gps_start, gps_end):
            block = np.asarray(reader(self.block_start(b), self.N), dtype=np.float64)
            if block.shape != (self.N,):
                raise ValueError(f"reader returned {block.shape} samples for block {b}, expected ({self.N},)")
            if not block.any():
                yield b, [np.zeros(self.samples_per_hop(i)) for i in range(self.nbands)]
                continue
            X = sfft.rfft(block, workers=self.workers)
            del block
            yield b, [self.band_from_spectrum(X, i, b) for i in range(self.nbands)]

    def run(
        self,
        reader: Reader,
        gps_start: float | Fraction,
        gps_end: float | Fraction,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[tuple[int, npt.NDArray[np.float64]]]:
        """Channelize ``[gps_start, gps_end)``. Returns ``(n_first, y)`` per band.

        ``y[j]`` is the sample at ``t0 + (n_first + j)/(2B)``.
        """
        blocks = self.blocks_for(gps_start, gps_end)
        pieces: list[list[npt.NDArray[np.float64]]] = [[] for _ in range(self.nbands)]
        for count, (_, ys) in enumerate(self.iter_blocks(reader, gps_start, gps_end), 1):
            for i, y in enumerate(ys):
                pieces[i].append(y)
            if progress is not None:
                progress(count, len(blocks))
        out = []
        for i in range(self.nbands):
            fs = self.fs_out(i)
            n_blk0 = blocks.start * self.samples_per_hop(i)
            y = np.concatenate(pieces[i]) if pieces[i] else np.zeros(0)
            n_a = math.ceil((Fraction(gps_start) - self.t0) * fs)
            n_b = math.ceil((Fraction(gps_end) - self.t0) * fs)
            out.append((n_a, y[n_a - n_blk0 : n_b - n_blk0].copy()))
        return out

    def describe(self) -> dict[str, Any]:
        """Settings to store with each product."""
        d0 = next(iter(self.designs.values()))
        return {
            "channelizer": "fft_overlap_save_kaiser",
            "channelizer_kernel_length_s": self.L,
            "channelizer_hop_s": self.hop,
            "channelizer_block_length_s": self.T,
            "channelizer_atten_db": self.atten_db,
            "channelizer_kaiser_beta": d0.beta,
            "channelizer_edge": self.edge,
            "channelizer_fold_frac": self.fold_frac,
            "channelizer_guard_hz": 2 * transition_halfwidth(d0.beta, self.atten_db) / self.L,
            "fs_in": self.fs_in,
        }
