"""Band planning for all-sky CW searches (ISSUES.md A2).

A search band ``[fmin, fmin + B]`` is searched over *source* frequencies, but
the data hold a signal only while its *detector-frame* frequency stays inside
the flat part of the band, ``[fmin + g, fmax - g]`` (g is the channelizer
guard, 2.18 mHz for L = 3600 s). The detector frequency of a source with
frequency f0 at ``t_ref`` and spin-down ``fdot`` is::

    f(t) = (f0 + fdot (t - t_ref)) (1 + v(t).n / c),   |v.n| / c <= D

with D = 1.06e-4 a safe bound for Earth's orbital plus rotational velocity.
So only source frequencies in::

    [(fmin + g) / (1 - D) - d_lo,  (fmax - g) / (1 + D) - d_hi]

keep their whole track in the band, where ``d_lo <= 0 <= d_hi`` are the most
negative and most positive ``fdot (t - t_ref)`` over the observation and the
spin-down range. That is the band's *usable* range. At 1 kHz, D f alone is
0.106 Hz on each side, so a 0.1 Hz band holds no usable frequency at all.

:func:`plan_bands` tiles a search range with overlapping bands whose usable
ranges meet exactly, and records each range in its :class:`BandSpec`; the
pipeline stores it in the product and the export. The search must restrict
itself to the usable range (rustyprune today searches the whole band).

Command line::

    python -m readligo plan --fmin 20 --fmax 2000 --tspan-days 246.1 > bands.txt
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from .config import FREQ_GRID_DEN, BandSpec, default_guard

__all__ = ["DOPPLER_MAX", "RUSTYPRUNE_BANDWIDTHS", "plan_bands", "usable_range"]

#: Bound on |v.n|/c for a detector on Earth (orbit 1.0e-4 at perihelion plus rotation 1.6e-6, with margin).
DOPPLER_MAX = 1.06e-4

#: Bandwidths rustyprune accepts today (k/10 Hz, k = 1..10).
RUSTYPRUNE_BANDWIDTHS = tuple(Fraction(k, 10) for k in range(1, 11))


def _drift(tspan: float, fdot_range: tuple[float, float], t_ref_frac: float) -> tuple[float, float]:
    """Most negative and most positive ``fdot (t - t_ref)`` (Hz) over the span and spin-down range."""
    if not (0.0 <= t_ref_frac <= 1.0):
        raise ValueError("t_ref_frac must be in [0, 1]")
    lo_f, hi_f = sorted(float(f) for f in fdot_range)
    dts = (-t_ref_frac * tspan, (1.0 - t_ref_frac) * tspan)
    vals = [0.0] + [f * dt for f in (lo_f, hi_f) for dt in dts]
    return min(vals), max(vals)


def usable_range(
    fmin: float,
    band: float,
    *,
    tspan: float,
    fdot_range: tuple[float, float] = (-1e-9, 0.0),
    t_ref_frac: float = 0.0,
    doppler: float = DOPPLER_MAX,
    guard: float | None = None,
) -> tuple[float, float]:
    """Usable source-frequency range ``(lo, hi)`` of the band ``[fmin, fmin + band]``.

    ``lo >= hi`` means no source frequency keeps its whole track in the band.
    *tspan* is the observation length (s), *t_ref_frac* the position of the
    reference time in it (0 = start), *fdot_range* the searched spin-down range.
    """
    g = default_guard() if guard is None else float(guard)
    d_lo, d_hi = _drift(float(tspan), fdot_range, t_ref_frac)
    lo = (float(fmin) + g) / (1.0 - doppler) - d_lo
    hi = (float(fmin) + float(band) - g) / (1.0 + doppler) - d_hi
    return lo, hi


def plan_bands(
    f_lo: float,
    f_hi: float,
    *,
    tspan: float,
    bandwidth: float | str = "auto",
    fdot_range: tuple[float, float] = (-1e-9, 0.0),
    t_ref_frac: float = 0.0,
    doppler: float = DOPPLER_MAX,
    guard: float | None = None,
    min_efficiency: float = 0.5,
    bandwidths: Sequence[Fraction] = RUSTYPRUNE_BANDWIDTHS,
) -> list[BandSpec]:
    """Bands whose usable source ranges tile ``[f_lo, f_hi]`` exactly, in order.

    Parameters
    ----------
    bandwidth
        A fixed width (Hz, on the 1/1800 Hz grid) for every band, or
        ``"auto"``: at each step the narrowest of *bandwidths* whose usable
        width is at least ``min_efficiency * B``.
    tspan, fdot_range, t_ref_frac, doppler, guard
        See :func:`usable_range`.

    Returns
    -------
    list of BandSpec
        Each with ``usable = (lo, hi)``; the first ``lo`` is *f_lo*, each ``hi``
        is the next band's ``lo``, and the last ``hi`` is *f_hi*. Band edges
        are on the 1/1800 Hz grid.

    Raises
    ------
    ValueError
        If a fixed bandwidth (or the widest of *bandwidths*) leaves no usable
        range somewhere in ``[f_lo, f_hi]``.
    """
    if not (0 < f_lo < f_hi):
        raise ValueError("need 0 < f_lo < f_hi")
    if not (0 < min_efficiency <= 1):
        raise ValueError("min_efficiency must be in (0, 1]")
    g = default_guard() if guard is None else float(guard)
    d_lo, d_hi = _drift(float(tspan), fdot_range, t_ref_frac)
    if bandwidth == "auto":
        widths = sorted(Fraction(b) for b in bandwidths)
        need = min_efficiency
    else:
        widths = [Fraction(round(float(bandwidth) * FREQ_GRID_DEN), FREQ_GRID_DEN)]
        need = 0.0
    out: list[BandSpec] = []
    cur = float(f_lo)
    while cur < f_hi - 1e-9:
        # largest grid fmin whose usable range starts at or below cur
        fmin_max = (cur + d_lo) * (1.0 - doppler) - g
        fmin = Fraction(math.floor(fmin_max * FREQ_GRID_DEN + 1e-9), FREQ_GRID_DEN)
        chosen = None
        for B in widths:
            lo, hi = usable_range(float(fmin), float(B), tspan=tspan, fdot_range=fdot_range,
                                  t_ref_frac=t_ref_frac, doppler=doppler, guard=g)
            if lo <= cur + 1e-9 and hi - cur > max(need * float(B), 1e-6):
                chosen = (B, hi)
                break
        if chosen is None:
            B = widths[-1]
            lo, hi = usable_range(float(fmin), float(B), tspan=tspan, fdot_range=fdot_range,
                                  t_ref_frac=t_ref_frac, doppler=doppler, guard=g)
            raise ValueError(
                f"at {cur:.4f} Hz a {float(B)} Hz band leaves {max(hi - cur, 0):.4f} Hz usable "
                f"(Doppler {doppler * cur:.4f} Hz and spin-down drift [{d_lo:.4g}, {d_hi:.4g}] Hz each side); "
                "use a wider band"
            )
        B, hi = chosen
        top = min(hi, float(f_hi))
        out.append(BandSpec(fmin, B, (max(cur, float(fmin)), top)))
        cur = top
    return out
