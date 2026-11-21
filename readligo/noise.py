"""Noise level and lines of a band product (ISSUES.md O4, G1, G2, A7).

The searches normalise the data by one variance, ``Sx * B``, where Sx is the
one-sided strain PSD in the band. A plain sample variance of the output is
biased: the channelizer rolls off within ``g`` of each edge (about 2.5 % low
at B = 0.1 Hz), and a line in the band can inflate it by orders of magnitude.

Here Sx is measured per segment (default 1800 s, aligned to the epoch) from a
Kaiser-tapered periodogram of the band output (sidelobes below about -130 dB,
so even a very strong line stays within a few bins), using only the flat part
of the band ``[fmin + g, fmax - g]``. Lines are found on the spectrum
averaged over all segments (each normalised by its own Sx), relative to a
running-median baseline: a run of bins is a line if one bin exceeds the
baseline at a false-alarm probability of ``line_false_alarm`` per product
(exact Gamma tail for the number of segments), and the run extends while its
bins exceed a looser level. Sx is then the mean power of the remaining bins;
lines and Sx are iterated three times. Partly gated segments are normalised
by their window.

The product's ``sqrt_sx`` is the live-time average ``sqrt(<Sx>)``: with one
variance for all data (the searches today), the noise 2F has mean 4 exactly
when the variance is the average one. The harmonic mean (the effective noise
of a noise-weighted search) and the per-segment values are stored too.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import ndimage, signal, stats

from .config import NoiseParams, VetoParams

__all__ = [
    "NoiseEstimate",
    "estimate_noise",
    "excess_power_veto",
    "find_lines",
    "gwosc_antialias_gain",
    "inband_shape",
    "mean_psd",
    "narrowband_veto",
    "robust_scale",
    "veto_band",
]


@dataclass
class NoiseEstimate:
    """Noise of one band product."""

    sqrt_sx: float  # sqrt of the live-time mean of Sx (normalisation for unweighted searches)
    sqrt_sx_harmonic: float  # sqrt of the live-time harmonic mean of Sx
    sqrt_sx_median: float
    #: per segment: gps_start, duration, duty (taper weighted), live (plain mean w^2), sx (NaN if not estimated)
    segments: dict[str, npt.NDArray] = field(default_factory=dict)
    #: averaged spectrum: freq (Hz, absolute), ratio (power / Sx), flat (bool)
    spectrum: dict[str, npt.NDArray] = field(default_factory=dict)
    #: lines: freq (Hz, absolute), ratio (peak bin of each flagged run of bins)
    lines: dict[str, npt.NDArray] = field(default_factory=dict)
    #: var(data) over fully good samples / (sqrt_sx^2 B); 1 - ~2.5 mHz/B expected (edge roll-off)
    variance_ratio: float = float("nan")
    n_segments_used: int = 0
    #: bins excluded from Sx (lines and their skirts)
    line_bins: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))


def find_lines(
    ratio: npt.NDArray[np.float64], usable: npt.NDArray[np.bool_], n_segments: int, params: NoiseParams
) -> npt.NDArray[np.bool_]:
    """Bins of lines in a segment-averaged spectrum *ratio* (power / Sx).

    The baseline is a running median over ``baseline_bins`` usable bins (so a
    tilted or curved noise floor is not taken for lines). For n segments, a
    noise bin's ratio to the baseline is Gamma(n, 1/n) distributed. A run of
    bins above the extension level is a line if its peak passes the seed
    level (false alarm ``line_false_alarm`` over all usable bins).
    """
    out = np.zeros(ratio.size, dtype=bool)
    idx = np.flatnonzero(usable & np.isfinite(ratio))
    if idx.size < 8 or n_segments < 1:
        return out
    r = ratio[idx]
    # wider baseline for few segments, so its own scatter stays small next to the bins'
    k = min(params.baseline_bins * max(1, int(np.ceil(16 / n_segments))), idx.size) | 1
    base = _running_median(r, k) / _gamma_median(n_segments)
    rel = r / np.where(base > 0, base, np.inf)
    seed = stats.gamma.isf(params.line_false_alarm / idx.size, a=n_segments, scale=1.0 / n_segments)
    ext = stats.gamma.isf(params.line_extend_false_alarm, a=n_segments, scale=1.0 / n_segments)
    hot = rel > ext
    lab, nlab = ndimage.label(hot)
    if nlab:
        peak = ndimage.maximum(rel, lab, index=np.arange(1, nlab + 1))
        good = np.flatnonzero(peak > seed) + 1
        out[idx[np.isin(lab, good)]] = True
    return out


def _running_median(r: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.float64]:
    """Median of the up to *k* values centred on each element (the window shrinks at the ends)."""
    h = k // 2
    if r.size <= k:
        return np.full(r.size, float(np.median(r)))
    padded = np.concatenate((np.full(h, np.nan), r, np.full(h, np.nan)))
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * h + 1)
    return np.nanmedian(win, axis=1)


def _gamma_median(n: int) -> float:
    """Median of Gamma(n, 1/n) (the mean of n unit exponentials)."""
    return float(stats.gamma.median(a=n, scale=1.0 / n))


def estimate_noise(
    data: npt.NDArray[np.float64],
    window: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    *,
    fmin: float,
    band: float,
    epoch: float,
    n0: int,
    guard: float,
    params: NoiseParams,
) -> NoiseEstimate:
    """Estimate Sx per segment, the band's lines and the global ``sqrt_sx``.

    Parameters
    ----------
    data, window, mask
        The band product's arrays (sample k at ``epoch + (n0 + k)/(2 band)``).
    guard
        Width (Hz) of the channelizer's roll-off at each band edge.
    """
    fs = 2.0 * band
    seg_n = int(round(params.segment * fs))
    if abs(seg_n - params.segment * fs) > 1e-6 or seg_n < 8:
        raise ValueError(f"noise segment {params.segment} s is not a whole number of samples at {fs} Hz")
    freqs = np.fft.rfftfreq(seg_n, 1.0 / fs)
    df = freqs[1]
    flat = (freqs >= guard + df) & (freqs <= band - guard - df)
    if flat.sum() < 8:
        raise ValueError("band too narrow for the noise estimate")
    taper = signal.windows.kaiser(seg_n, params.taper_beta, sym=False)
    t2 = float(np.sum(taper**2))
    halfwidth = int(np.ceil(np.sqrt(1.0 + (params.taper_beta / np.pi) ** 2)))  # main-lobe half-width, bins

    m_first = int(np.ceil(n0 / seg_n))
    m_last = int(np.floor((n0 + data.size) / seg_n))  # exclusive
    nseg = max(0, m_last - m_first)
    a0 = m_first * seg_n - n0
    w2 = (np.asarray(window[a0 : a0 + nseg * seg_n], dtype=np.float64) ** 2).reshape(nseg, seg_n)
    duty = (w2 @ taper**2) / t2 if nseg else np.zeros(0)
    live = w2.mean(axis=1) if nseg else np.zeros(0)
    starts_a = epoch + (m_first + np.arange(nseg)) * seg_n / fs
    used = np.flatnonzero((live >= params.min_duty) & (duty > 0))
    sx = np.full(nseg, np.nan)
    ratio = np.full(freqs.size, np.nan)
    line_bins = np.zeros(freqs.size, dtype=bool)
    if used.size:
        X = data[a0 : a0 + nseg * seg_n].reshape(nseg, seg_n)[used] * taper
        P = 2.0 * np.abs(np.fft.rfft(X, axis=1)) ** 2 / (fs * t2 * duty[used, None])
        del X
        keep = flat.copy()
        sx_u = np.median(P[:, keep], axis=1) / np.log(2.0)  # first pass: median of exponential bins
        for _ in range(3):
            ratio = np.mean(P / sx_u[:, None], axis=0)
            line_bins = find_lines(ratio, flat, used.size, params)
            grown = ndimage.binary_dilation(line_bins, structure=np.ones(2 * halfwidth + 1, dtype=bool))
            keep_new = flat & ~grown
            if keep_new.sum() < 8:
                break
            keep = keep_new
            sx_u = np.mean(P[:, keep], axis=1)
        ratio = np.mean(P / sx_u[:, None], axis=0)
        sx[used] = sx_u
        line_bins = flat & ~keep
    ok = np.isfinite(sx)
    if ok.any():
        wts = duty[ok]
        mean_sx = float(np.sum(wts * sx[ok]) / np.sum(wts))
        harm_sx = float(np.sum(wts) / np.sum(wts / sx[ok]))
        med_sx = float(np.median(sx[ok]))
    else:
        mean_sx = harm_sx = med_sx = float("nan")
    good = mask & (np.asarray(window) > 0)
    var_ratio = float(np.var(data[good]) / (mean_sx * band)) if good.sum() > 10 and ok.any() else float("nan")
    seeds = find_lines(ratio, flat, max(int(used.size), 1), params) if used.size else np.zeros(freqs.size, bool)
    lab, nlab = ndimage.label(seeds)
    peaks = (np.array(ndimage.maximum_position(ratio, lab, index=np.arange(1, nlab + 1)), dtype=np.int64).ravel()
             if nlab else np.zeros(0, np.int64))
    return NoiseEstimate(
        sqrt_sx=float(np.sqrt(mean_sx)),
        sqrt_sx_harmonic=float(np.sqrt(harm_sx)),
        sqrt_sx_median=float(np.sqrt(med_sx)),
        segments={
            "gps_start": starts_a,
            "duration": np.full(nseg, float(params.segment)),
            "duty": duty,
            "live": live,
            "sx": sx,
        },
        spectrum={"freq": fmin + freqs, "ratio": ratio, "flat": flat, "excluded": line_bins},
        lines={"freq": fmin + freqs[peaks], "ratio": ratio[peaks] if peaks.size else np.zeros(0)},
        variance_ratio=var_ratio,
        n_segments_used=int(used.size),
        line_bins=line_bins,
    )


def gwosc_antialias_gain(f: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Amplitude gain of the GWOSC 16 -> 4 kHz decimation at frequency *f* (Hz).

    GWOSC made the 4 kHz files with ``scipy.signal.decimate(q=4)`` defaults
    (Chebyshev I, order 8, 0.05 dB ripple, applied forwards and backwards), so
    the strain is scaled by ``|H(f)|^2``: within 1.2 % of 1 below 1.5 kHz
    (passband ripple, 0.989-1.000), 0.78 at 1700 Hz, 0.0071 at 1991 Hz
    (measured on hardware injection P14). Signal and noise are scaled
    together, so SNR is unchanged, but absolute amplitudes must include this
    factor, above ~1.6 kHz especially.
    """
    b, a = signal.cheby1(8, 0.05, 0.8 / 4)
    _, h = signal.freqz(b, a, worN=np.atleast_1d(np.asarray(f, dtype=np.float64)), fs=16384)
    return np.abs(h) ** 2


_CHI2_1_MEDIAN = 0.45493642311957283  # median of a chi-square variable with 1 degree of freedom


def robust_scale(
    data: npt.NDArray[np.float64],
    window: npt.NDArray[np.floating],
    *,
    epoch: float,
    n0: int,
    fs: float,
    segment: float,
    min_live: float = 0.25,
) -> npt.NDArray[np.float64]:
    """Per-sample variance scale: ``median(x^2) / 0.4549`` over the live samples of its segment.

    For Gaussian noise this is the variance. A stationary line raises it as
    much as it raises x^2, so lines never look like outliers; a transient
    lasting a small part of the segment leaves it unchanged. Segments with
    fewer than ``min_live`` x segment live samples (window > 0.5) take the
    scale of the nearest segment that has enough (ISSUES.md A15).
    """
    data = np.asarray(data, dtype=np.float64)
    seg_n = int(round(segment * fs))
    n = data.size
    if n == 0:
        return np.zeros(0)
    m0 = int(np.floor(n0 / seg_n))
    m1 = int(np.floor((n0 + n - 1) / seg_n)) + 1
    pad_lo = n0 - m0 * seg_n
    total = (m1 - m0) * seg_n
    sq = np.full(total, np.nan)
    live = np.asarray(window) > 0.5
    sq[pad_lo : pad_lo + n] = np.where(live, data**2, np.nan)
    sq = sq.reshape(m1 - m0, seg_n)
    count = np.sum(np.isfinite(sq), axis=1)
    ok = count >= max(8, int(min_live * seg_n))
    scale = np.full(m1 - m0, np.nan)
    if ok.any():
        scale[ok] = np.nanmedian(sq[ok], axis=1) / _CHI2_1_MEDIAN
    ok &= np.isfinite(scale) & (scale > 0)
    if not ok.any():
        scale[:] = 1.0
    else:
        good = np.flatnonzero(ok)
        j = np.searchsorted(good, np.arange(scale.size))
        left = good[np.clip(j - 1, 0, good.size - 1)]
        right = good[np.clip(j, 0, good.size - 1)]
        pos = np.arange(scale.size)
        nearest = np.where(np.abs(pos - left) <= np.abs(right - pos), left, right)
        scale = scale[nearest]
    return np.repeat(scale, seg_n)[pad_lo : pad_lo + n]


def narrowband_veto(
    data: npt.NDArray[np.float64],
    window: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    scale: npt.NDArray[np.float64],
    params: VetoParams,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Zero runs of outlying samples in place (data, window, mask).

    ``z = data^2 / scale`` (see :func:`robust_scale`). A run is a maximal set
    of consecutive samples with ``z > params.extent``; runs whose peak exceeds
    ``params.seed`` are zeroed together with ``params.pad`` samples on each
    side. Padded runs that touch are merged, so each sample is counted once.

    Returns
    -------
    start, stop, peak_z
        Vetoed sample ranges ``[start, stop)`` (indices into *data*), disjoint
        and sorted, and the peak z of each.
    """
    if not params.enabled or data.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    live = np.asarray(window) > 0
    z = np.where(live, data**2 / scale, 0.0)
    above = z > params.extent
    d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    ranges: list[list[float]] = []
    for a, b in zip(starts, stops):
        peak = float(z[a:b].max())
        if peak <= params.seed:
            continue
        a2, b2 = max(0, a - params.pad), min(data.size, b + params.pad)
        if ranges and a2 <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], b2)
            ranges[-1][2] = max(ranges[-1][2], peak)
        else:
            ranges.append([a2, b2, peak])
    for a2, b2, _ in ranges:
        data[a2:b2] = 0.0
        window[a2:b2] = 0.0
        mask[a2:b2] = False
    return (np.array([r[0] for r in ranges], np.int64), np.array([r[1] for r in ranges], np.int64),
            np.array([r[2] for r in ranges]))


def excess_power_veto(
    data: npt.NDArray[np.float64],
    window: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    scale: npt.NDArray[np.float64],
    fs: float,
    params: VetoParams,
    *,
    min_live: float = 0.75,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Zero windows of raised in-band power in place (data, window, mask).

    For each length t in ``params.excess_scales``, shortest first, windows of
    ``W = max(2, t fs)`` samples step by W/4. A window with at least
    ``min_live`` W live samples whose mean ``z = data^2 / scale`` exceeds
    ``chi2.isf(p, W) / W`` is zeroed, with p the false-alarm probability per
    band and length divided by the number of windows. Samples zeroed at one
    length are not live at the next, so an episode is cut at the shortest
    length that sees it. For Gaussian noise the expected loss is about
    ``excess_false_alarm`` windows per length per band.

    Returns
    -------
    start, stop, mean_z
        Zeroed sample ranges (disjoint, sorted) and the largest window mean in each.
    """
    if not params.enabled or not params.excess_scales or data.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    live = np.asarray(window) > 0
    z = np.where(live, data**2 / scale, 0.0)
    hits: list[tuple[int, int, float]] = []
    for t in sorted(params.excess_scales):
        W = max(2, int(round(t * fs)))
        if W > data.size:
            continue
        cz = np.concatenate(([0.0], np.cumsum(z)))
        cl = np.concatenate(([0], np.cumsum(live)))
        starts = np.arange(0, data.size - W + 1, max(1, W // 4))
        k = cl[starts + W] - cl[starts]
        ok = k >= min_live * W
        if not ok.any():
            continue
        mean = (cz[starts + W] - cz[starts]) / np.maximum(k, 1)
        thr = stats.chi2.isf(params.excess_false_alarm / int(ok.sum()), W) / W
        sel = ok & (mean > thr)
        for a, m in zip(starts[sel], mean[sel]):
            hits.append((int(a), int(a) + W, float(m)))
            live[a : a + W] = False
            z[a : a + W] = 0.0
    hits.sort()
    ranges: list[list[float]] = []
    for a, b, m in hits:
        if ranges and a <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], b)
            ranges[-1][2] = max(ranges[-1][2], m)
        else:
            ranges.append([a, b, m])
    for a, b, _ in ranges:
        data[a:b] = 0.0
        window[a:b] = 0.0
        mask[a:b] = False
    return (np.array([r[0] for r in ranges], np.int64), np.array([r[1] for r in ranges], np.int64),
            np.array([r[2] for r in ranges]))


def mean_psd(
    strain: npt.NDArray[np.floating],
    gps_start: int,
    fs: int,
    stretches: Sequence[tuple[int, int]],
    gate_spans: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    *,
    seg: int,
    edge: float,
) -> tuple[npt.NDArray[np.float64], int]:
    """Sum of one-sided Hann periodograms (density) of *seg* s segments, and their count.

    Segments lie inside the stretches, *edge* s from their ends, and overlap
    no gate span (``(starts, ends)`` in GPS), so neither tapers nor gates
    smear lines into the estimate. *strain* starts at integer GPS *gps_start*.
    """
    nper = int(seg * fs)
    win = signal.windows.hann(nper, sym=False)
    norm = fs * float(np.sum(win**2))
    total = np.zeros(nper // 2 + 1)
    count = 0
    g0, g1 = gate_spans
    order = np.argsort(g0)
    g0, g1 = np.asarray(g0)[order], np.maximum.accumulate(np.asarray(g1)[order]) if len(g1) else np.asarray(g1)
    for a, b in stretches:
        t = a + edge
        while t + seg <= b - edge:
            k = np.searchsorted(g0, t + seg)
            if k == 0 or g1[k - 1] <= t:
                i0 = int(round((t - gps_start) * fs))
                x = np.asarray(strain[i0 : i0 + nper], dtype=np.float64)
                total += 2.0 * np.abs(np.fft.rfft(x * win)) ** 2 / norm
                count += 1
            t += seg
    return total, count


def inband_shape(
    freqs: npt.NDArray[np.float64],
    psd: npt.NDArray[np.float64],
    n_avg: int,
    fmin: float,
    band: float,
    *,
    min_change: float = 0.01,
    margin: float = 0.25,
    min_halfwidth: float = 0.25,
    max_range: float = 10.0,
    max_misfit: float = 0.05,
) -> tuple[float, ...] | None:
    """Smooth shape ``log S = c + a1 u + a2 u^2 (+ a3 u^3)`` (u = f - fc) of a mean PSD across one band.

    Fitted on ``fc +- max(B/2 + margin B, min_halfwidth)`` (a cubic when that
    holds 40 or more bins, else a quadratic), with lines (bins more than 3
    robust sigma above the fit) dropped iteratively. Returns
    ``(fc, a1, a2[, a3])``, or None if the fitted PSD changes by less than
    *min_change* across the band, the change is not significant (chi-square
    of the coefficients below 25), or there are too few bins. Also None when
    the shape is not a smooth floor: the fit changes S by more than
    *max_range* times across the band, or misses the kept bins by more than
    *max_misfit* rms beyond their statistical scatter (a band dominated by a
    strong line's skirt, where flattening would amplify the band edges).
    """
    fc = fmin + band / 2
    hw = max(band / 2 + margin * band, min_halfwidth)
    sel = (np.abs(freqs - fc) <= hw) & (psd > 0)
    if sel.sum() < 6:
        return None
    u = freqs[sel] - fc
    y = np.log(psd[sel])
    keep = np.ones(u.size, dtype=bool)
    deg = 3 if u.size >= 40 else (2 if u.size >= 8 else 1)
    coef = np.zeros(deg + 1)
    sig = 1.0
    for _ in range(8):
        if keep.sum() < deg + 4:
            return None
        coef = np.polyfit(u[keep], y[keep], deg)
        r = y - np.polyval(coef, u)
        sig = max(1.4826 * float(np.median(np.abs(r[keep] - np.median(r[keep])))), 1.0 / np.sqrt(max(n_avg, 1)))
        new = (r < 3.0 * sig) & (r > -6.0 * sig)
        if np.array_equal(new, keep):
            break
        keep = new
    a = coef[::-1][1:]  # a1, a2, ...
    A = np.stack([u[keep] ** k for k in range(1, deg + 1)], axis=1)
    A = A - A.mean(axis=0)
    chi2 = float(a @ (A.T @ A) @ a) / sig**2
    grid = np.linspace(-band / 2, band / 2, 65)
    rel = sum(ak * grid ** (k + 1) for k, ak in enumerate(a))
    if np.max(np.abs(rel)) < np.log1p(min_change) or chi2 < 25.0:
        return None
    misfit = np.sqrt(max(sig**2 - 1.0 / max(n_avg, 1), 0.0))
    if np.ptp(rel) > np.log(max_range) or misfit > max_misfit:
        return None
    return (float(fc), *[float(x) for x in a])


def veto_band(
    data: npt.NDArray[np.float64],
    window: npt.NDArray[np.floating],
    mask: npt.NDArray[np.bool_],
    *,
    epoch: float,
    n0: int,
    fs: float,
    segment: float,
    params: VetoParams,
    max_passes: int = 6,
) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray], npt.NDArray[np.float64], int]:
    """Narrowband and excess-power vetoes, repeated until nothing new is vetoed (in place).

    A burst that fills most of a noise segment (a line ringing up at lock loss)
    inflates the segment's robust variance, so one pass removes only its core.
    Each pass recomputes the variance without the samples vetoed so far
    (ISSUES.md A9 follow-up, found on O4a).

    Returns
    -------
    narrow, excess
        ``{"start", "stop", "peak"}`` sample ranges of each veto (disjoint, sorted).
    scale
        The final per-sample variance scale.
    passes
        Number of passes run.
    """
    acc: dict[str, list[tuple[int, int, float]]] = {"narrow": [], "excess": []}
    scale = robust_scale(data, window, epoch=epoch, n0=n0, fs=fs, segment=segment)
    passes = 0
    for passes in range(1, max_passes + 1):
        a, b, pk = narrowband_veto(data, window, mask, scale, params)
        acc["narrow"].extend(zip(a.tolist(), b.tolist(), pk.tolist()))
        xa, xb, xm = excess_power_veto(data, window, mask, scale, fs, params)
        acc["excess"].extend(zip(xa.tolist(), xb.tolist(), xm.tolist()))
        if a.size == 0 and xa.size == 0:
            break
        scale = robust_scale(data, window, epoch=epoch, n0=n0, fs=fs, segment=segment)

    def merged(items: list[tuple[int, int, float]]) -> dict[str, npt.NDArray]:
        items.sort()
        out: list[list[float]] = []
        for a, b, v in items:
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
                out[-1][2] = max(out[-1][2], v)
            else:
                out.append([a, b, v])
        return {"start": np.array([o[0] for o in out], np.int64), "stop": np.array([o[1] for o in out], np.int64),
                "peak": np.array([o[2] for o in out], np.float64)}

    return merged(acc["narrow"]), merged(acc["excess"]), scale, passes
