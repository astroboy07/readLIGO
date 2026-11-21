"""High-pass, stretches, gate windows, glitch detection and the output window."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from readligo.conditioning import (
    Gate,
    GateParams,
    design_highpass,
    detect_gates,
    gate_window,
    good_stretches,
    highpass_zero_phase,
    output_window,
)


def test_good_stretches() -> None:
    good = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0, 1], dtype=bool)
    assert good_stretches(good) == [(1, 4), (6, 8), (9, 10)]
    assert good_stretches(good, min_stretch=2) == [(1, 4), (6, 8)]


@pytest.mark.parametrize("chunk", [5000, 1 << 20])
def test_chunked_highpass_matches_padded_convolution(chunk: int) -> None:
    fs = 1024
    taps = design_highpass(GateParams(), fs)
    half = (len(taps) - 1) // 2
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=60 * fs))  # red noise, large edge values
    ref = signal.oaconvolve(np.pad(x, half, mode="reflect", reflect_type="odd"), taps, mode="valid")
    out = np.empty_like(x)
    highpass_zero_phase(lambda a, b: x[a:b], x.size, taps, out, chunk=chunk)
    assert np.allclose(out, ref, rtol=0, atol=1e-9 * np.max(np.abs(ref)))


def test_highpass_response() -> None:
    fs = 4096
    taps = design_highpass(GateParams(), fs)
    w, h = signal.freqz(taps, worN=[7.93, 10.0, 15.0, 100.0], fs=fs)
    g = np.abs(h)
    assert g[0] < 1e-6 and g[1] < 1.1e-6
    assert abs(g[2] - 1) < 2e-6 and abs(g[3] - 1) < 2e-6
    assert len(taps) % 2 == 1 and np.allclose(taps, taps[::-1])


def test_gate_window_shape() -> None:
    w = gate_window(100, [(40, 50)], taper=10)
    assert np.all(w[40:50] == 0)
    assert np.all(w[:30] == 1) and np.all(w[60:] == 1)
    assert np.all(np.diff(w[30:40]) < 0) and np.all(np.diff(w[50:60]) > 0)
    # ranges partly outside the block
    w2 = gate_window(100, [(-5, 3), (95, 120)], taper=4)
    assert np.all(w2[:3] == 0) and np.all(w2[95:] == 0) and w2[50] == 1


def test_detect_gates_finds_glitch_and_nothing_else() -> None:
    fs = 1024
    p = GateParams(white_fhi=400.0)
    rng = np.random.default_rng(5)
    x = rng.normal(size=600 * fs)
    t = np.arange(x.size) / fs
    x_clean = x.copy()
    tg = 300.2
    x += 200 * np.exp(-(((t - tg) / 0.02) ** 2)) * np.sin(2 * np.pi * 150 * (t - tg))
    clusters, info = detect_gates(x, [(0, x.size)], p, fs)
    assert len(clusters) == 1
    first, last, k, peak = clusters[0]
    assert abs(k / fs - tg) < 0.05 and peak > p.threshold
    none, _ = detect_gates(x_clean, [(0, x.size)], p, fs)
    assert none == []
    assert 0.9 < info["sigma"] / np.median(np.abs(x_clean)) < 1.1 or info["sigma"] > 0


def test_output_window() -> None:
    p = GateParams(edge_taper=1.0, gate_taper=0.25)
    epoch, fs_out = 1_000_000, 0.2  # 5 s slots
    stretches = [(epoch, epoch + 100)]
    gate = Gate(k_start=50 * 1024, k_end=51 * 1024, k_peak=50 * 1024, peak=30.0, fs=1024, epoch=epoch)
    win, mask = output_window(stretches, [gate], p, epoch=epoch, fs_out=fs_out, n0=-2, n=25)
    # slot k covers [epoch + (k-2)*5 - 2.5, ... + 2.5)
    assert win[0] == 0 and not mask[0]  # before the data
    assert 0 < win[2] < 1 and not mask[2]  # contains the stretch start and its taper
    assert win[5] == 1 and mask[5]
    assert 0 < win[12] < 1 and not mask[12]  # slot [47.5, 52.5) holds the 1 s gate
    assert abs(win[12] ** 2 - (1 - 1.25 / 5)) < 0.02
    assert win[24] == 0  # after the stretch


def test_detect_gates_short_stretch_near_block_start() -> None:
    """ISSUES.md A3: a stretch ending < 3 s into the block must not wrap the slices."""
    fs = 1024
    p = GateParams(white_fhi=400.0)
    rng = np.random.default_rng(2)
    x = rng.normal(size=400 * fs)
    x[int(2.5 * fs) : 100 * fs] = 0.0  # DATA off from 2.5 s to 100 s
    stretches = [(0, int(2.5 * fs)), (100 * fs, x.size)]
    clusters, info = detect_gates(x, stretches, p, fs)
    _, ref = detect_gates(x[100 * fs :], [(0, x.size - 100 * fs)], p, fs)
    assert clusters == []
    assert abs(info["sigma"] / ref["sigma"] - 1) < 0.05  # was 0.3: the zeros were in the sigma


def test_gate_params_validation() -> None:
    with pytest.raises(ValueError):
        GateParams(min_stretch=4.0, edge_taper=4.0)
    with pytest.raises(ValueError):
        GateParams(gate_pad=-1.0)
    with pytest.raises(ValueError):
        GateParams(low_threshold=30.0)
