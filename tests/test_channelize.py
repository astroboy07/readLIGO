"""Channelizer: exact rate for any B on the grid, flat gain, alias rejection,
global-epoch heterodyne, chunk independence (ISSUES.md C1, C2, C3, C7, C8)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from readligo.channelize import FFTChannelizer

T0 = 1238319104
FS_IN = 256  # low input rate keeps the tests fast; the method does not depend on it


def tone_reader(f: float, amp: float = 1.0, phase: float = 0.0, fs: int = FS_IN):
    """reader(gps, n) for A cos(2 pi f (t - T0) + phase), with the phase at gps computed exactly."""
    fr = Fraction(f)

    def read(gps: int, n: int) -> np.ndarray:
        cyc0 = float((fr * (gps - T0)) % 1)
        k = np.arange(n)
        return amp * np.cos(2 * np.pi * (cyc0 + f * k / fs) + phase)

    return read


def response(ch: FFTChannelizer, f: float, i: int = 0, span: int = 7200) -> complex:
    """Complex gain with which a unit tone at f appears in band i (least squares over the output)."""
    (n0, y), *_ = [ch.run(tone_reader(f), T0, T0 + span)[i]]
    fs = float(ch.fs_out(i))
    t = (n0 + np.arange(y.size)) / fs
    nu = f - float(ch.fmin[i])
    X = np.stack([np.cos(2 * np.pi * nu * t), -np.sin(2 * np.pi * nu * t)], axis=1)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return complex(c[0], c[1])


@pytest.mark.parametrize("band", [0.1, 0.15, 0.37, 0.55, 1.0])
@pytest.mark.parametrize("fmin", [20.0, 100.68, 107.0 + 7 / 1800])
def test_exact_rate_and_flat_gain(fmin: float, band: float) -> None:
    ch = FFTChannelizer([(fmin, band)], T0, fs_in=FS_IN)
    assert ch.fs_out(0) == 2 * Fraction(round(band * 1800), 1800)
    n0, y = ch.run(tone_reader(fmin + band / 2), T0, T0 + 3600)[0]
    assert n0 == 0 and y.size == round(3600 * 2 * band)
    g = ch.guard(0)
    for f in np.linspace(fmin + g + 1e-4, fmin + band - g - 1e-4, 5):
        G = response(ch, f)
        assert abs(abs(G) - 1) < 2e-6, (f, G)
        assert abs(np.angle(G)) < 1e-7, (f, G)


@pytest.mark.parametrize("offset", [1e-4, 1e-3, 5e-3, 0.03, 0.3, 3.0])
def test_out_of_band_tones_rejected(offset: float) -> None:
    fmin, band = 100.68, 0.1
    ch = FFTChannelizer([(fmin, band)], T0, fs_in=FS_IN)
    for f in (fmin - offset, fmin + band + offset):
        n0, y = ch.run(tone_reader(f), T0, T0 + 7200)[0]
        assert np.max(np.abs(y)) < 10 ** (-120 / 20) * 1.05, (f, np.max(np.abs(y)))


def test_heterodyne_convention_and_amplitude() -> None:
    """A cos(2 pi f (t - T0) + phi) -> A cos(2 pi (f - fmin)(t_n - T0) + phi) on t_n = T0 + n/(2B)."""
    fmin, band, f, amp, phi = 100.68, 0.1, 100.7311, 3.0, 0.7
    ch = FFTChannelizer([(fmin, band)], T0, fs_in=FS_IN)
    n0, y = ch.run(tone_reader(f, amp, phi), T0 - 3600, T0 + 7200)[0]
    t = (n0 + np.arange(y.size)) / 0.2
    expected = amp * np.cos(2 * np.pi * (f - fmin) * t + phi)
    assert n0 == -720
    assert np.max(np.abs(y - expected)) < 1e-5 * amp


def test_chunking_is_bit_identical() -> None:
    bands = [(100.68, 0.1), (100.6, 0.25), (101.0, 1.0)]
    ch = FFTChannelizer(bands, T0, fs_in=FS_IN)
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(40000 + 7200) * FS_IN)

    def read(gps: int, n: int) -> np.ndarray:
        i = (gps - (T0 - 3600)) * FS_IN
        out = np.zeros(n)
        a, b = max(0, -i), min(n, noise.size - i)
        out[a:b] = noise[i + a : i + b]
        return out

    whole = ch.run(read, T0, T0 + 36000)
    cuts = [T0, T0 + 5000, T0 + 13321, T0 + 20000, T0 + 36000]
    for i in range(len(bands)):
        parts = [ch.run(read, a, b)[i] for a, b in zip(cuts, cuts[1:])]
        joined = np.concatenate([p[1] for p in parts])
        assert parts[0][0] == whole[i][0]
        assert np.array_equal(joined, whole[i][1])


def test_low_frequency_line_does_not_leak() -> None:
    """ISSUES C1: a huge 7.93 Hz line must not appear in 56-57, 71-72 or 519-520 Hz."""
    bands = [(56.0, 1.0), (71.0, 1.0), (100.0, 1.0)]
    ch = FFTChannelizer(bands, T0, fs_in=2048)
    line = tone_reader(7.93, amp=1e10, fs=2048)
    for i, (n0, y) in enumerate(ch.run(line, T0, T0 + 3600)):
        assert np.max(np.abs(y)) < 1e10 * 1e-12, bands[i]


@pytest.mark.parametrize("bad", [(100.6801, 0.1), (100.68, 0.123), (100.68, -0.1)])
def test_rejects_off_grid(bad: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        FFTChannelizer([bad], T0, fs_in=FS_IN)


def test_rejects_bad_block_grid() -> None:
    with pytest.raises(ValueError):
        FFTChannelizer([(100.68, 0.1)], T0, L=1000, fs_in=FS_IN)
    with pytest.raises(ValueError):
        FFTChannelizer([(100.68, 0.1)], T0, hop=1000, fs_in=FS_IN)
    with pytest.raises(ValueError):
        FFTChannelizer([(100.68, 0.1)], T0 + 0.5, fs_in=FS_IN)


def test_rejects_half_cycle_blocks() -> None:
    """ISSUES.md A11: hop 900 with an odd-grid fmin makes fmin*T a half-integer."""
    with pytest.raises(ValueError, match="whole number of cycles"):
        FFTChannelizer([(18001 / 1800, 0.1)], T0, hop=900, fs_in=FS_IN)
    FFTChannelizer([(18002 / 1800, 0.1)], T0, hop=900, fs_in=FS_IN)  # even grid index: fine
    FFTChannelizer([(18001 / 1800, 0.1)], T0, hop=3600, fs_in=FS_IN)


def test_shape_gain_scales_each_frequency() -> None:
    """ISSUES.md A8: set_shape multiplies the response at f by exp(-(a1 u + a2 u^2)/2)."""
    ch = FFTChannelizer([(100.68, 0.5)], T0, fs_in=FS_IN)
    shape = (100.93, -2.0, 1.5)
    ch.set_shape(0, shape)
    for f in (100.70, 100.93, 101.15):
        u = f - shape[0]
        expect = np.exp(-0.5 * (shape[1] * u + shape[2] * u * u))
        assert abs(abs(response(ch, f)) / expect - 1) < 1e-5
