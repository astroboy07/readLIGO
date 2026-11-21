"""Noise estimate, lines and the narrowband veto."""
from __future__ import annotations

import numpy as np

from readligo.config import NoiseParams, VetoParams
from readligo.noise import estimate_noise, gwosc_antialias_gain, narrowband_veto, robust_scale

B, FS = 0.1, 0.2
N = int(20 * 1800 * FS)


def _noise(seed: int = 0, sx: float = 4.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, np.sqrt(sx * B), N)


def test_white_noise_level_and_line() -> None:
    x = _noise()
    t = np.arange(N) / FS
    x_line = x + 3.0 * np.cos(2 * np.pi * 0.05 * t)  # a line at fmin + 0.05 Hz
    w = np.ones(N, dtype=np.float32)
    est = estimate_noise(x_line, w, w == 1, fmin=100.0, band=B, epoch=0, n0=0, guard=0.0022, params=NoiseParams())
    assert abs(est.sqrt_sx**2 / 4.0 - 1) < 0.05  # the line does not bias Sx
    assert np.any(np.abs(est.lines["freq"] - 100.05) < 1e-3)
    assert est.n_segments_used == 20


def test_veto_removes_transient_not_line_or_noise() -> None:
    x = _noise(1)
    t = np.arange(N) / FS
    x += 5.0 * np.cos(2 * np.pi * 0.03 * t)  # strong stationary line (amplitude ~8 sigma)
    x[1000:1003] += 40.0  # a short transient
    w = np.ones(N, dtype=np.float32)
    m = w == 1
    scale = robust_scale(x, w, epoch=0, n0=0, fs=FS, segment=1800)
    a, b, peak = narrowband_veto(x, w, m, scale, VetoParams())
    assert len(a) == 1 and a[0] <= 1000 and b[0] >= 1003 and np.all(x[a[0] : b[0]] == 0)
    clean = _noise(2)
    w2 = np.ones(N, dtype=np.float32)
    a2, _, _ = narrowband_veto(clean, w2, w2 == 1, robust_scale(clean, w2, epoch=0, n0=0, fs=FS, segment=1800),
                               VetoParams())
    assert len(a2) == 0


def test_gwosc_antialias_gain() -> None:
    g = gwosc_antialias_gain([100.0, 1700.0, 1991.28])
    assert abs(g[0] - 1) < 0.02 and 0.7 < g[1] < 0.85 and abs(g[2] / 0.00712 - 1) < 0.05


def test_excess_power_veto() -> None:
    """ISSUES.md A9: a 2-minute episode of 3x variance is removed; Gaussian noise loses nothing."""
    from readligo.config import VetoParams
    from readligo.noise import excess_power_veto

    fs = 2.0
    rng = np.random.default_rng(4)
    x = rng.normal(size=int(2 * 86400 * fs))
    x[100000:100240] *= np.sqrt(3.0)
    w = np.ones(x.size, np.float32)
    m = w == 1
    s = robust_scale(x, w, epoch=0, n0=0, fs=fs, segment=1800)
    a, b, _ = excess_power_veto(x, w, m, s, fs, VetoParams())
    assert a.size >= 1 and a[0] <= 100040 and b[-1] >= 100200 and np.sum(b - a) < 2.5 * 240
    clean = rng.normal(size=x.size)
    w2 = np.ones(x.size, np.float32)
    a2, _, _ = excess_power_veto(clean, w2, w2 == 1, robust_scale(clean, w2, epoch=0, n0=0, fs=fs, segment=1800),
                                 fs, VetoParams())
    assert a2.size == 0


def test_inband_shape_fit() -> None:
    """ISSUES.md A8: the shape of a tilted PSD is found through lines; a flat one gives None."""
    from readligo.noise import inband_shape

    rng = np.random.default_rng(8)
    n_avg = 2000
    freqs = np.arange(0, 60, 1 / 128)
    true = np.exp(-0.8 * (freqs - 25.5))  # PSD falls 55 % over 1 Hz
    psd = true * rng.gamma(n_avg, 1 / n_avg, freqs.size)
    psd[(np.abs(freqs - 25.2) < 0.02) | (np.abs(freqs - 25.7) < 0.01)] *= 50  # two lines
    sh = inband_shape(freqs, psd, n_avg, 25.0, 1.0)
    assert sh is not None and abs(sh[0] - 25.5) < 1e-12 and abs(sh[1] + 0.8) < 0.02
    flat = rng.gamma(n_avg, 1 / n_avg, freqs.size)
    assert inband_shape(freqs, flat, n_avg, 25.0, 1.0) is None
    assert inband_shape(freqs, flat, n_avg, 25.0, 0.1) is None


def test_veto_band_iterates_through_a_long_burst() -> None:
    """A burst filling most of a segment inflates its variance; repeated passes still remove its wings."""
    from readligo.config import VetoParams
    from readligo.noise import veto_band

    fs = 1.0
    rng = np.random.default_rng(3)
    x = rng.normal(size=86400)
    env = np.zeros(x.size)
    a0 = 18300
    env[a0 : a0 + 700] = 40.0
    env[a0 + 700 : a0 + 1600] = 40 * np.exp(-np.arange(900) / 150)
    x += env * rng.normal(size=x.size)
    w = np.ones(x.size, np.float32)
    m = w == 1
    _, _, _, passes = veto_band(x, w, m, epoch=0, n0=0, fs=fs, segment=1800, params=VetoParams())
    z = x**2 / robust_scale(x, w, epoch=0, n0=0, fs=fs, segment=1800)
    assert passes > 1 and np.sum(z[m] > 25) == 0 and np.sum(~m) < 1700


def test_inband_shape_skips_line_skirts() -> None:
    """A band centred on a strong Lorentzian line is not flattened (its edges would be amplified)."""
    from readligo.noise import inband_shape

    rng = np.random.default_rng(9)
    n_avg = 20000
    freqs = np.arange(500, 520, 1 / 128)
    psd = (1 + 300 / (1 + ((freqs - 510.0) / 0.01) ** 2)) * rng.gamma(n_avg, 1 / n_avg, freqs.size)
    assert inband_shape(freqs, psd, n_avg, 509.85, 0.3) is None
