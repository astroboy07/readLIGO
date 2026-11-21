"""Write small synthetic files in the GWOSC HDF5 layout, for tests."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

DQ_NAMES = ["DATA", "CBC_CAT1", "CBC_CAT2", "CBC_CAT3", "BURST_CAT1", "BURST_CAT2", "BURST_CAT3"]
INJ_NAMES = ["NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_DETCHAR_HW_INJ", "NO_CW_HW_INJ", "NO_STOCH_HW_INJ"]


def write_file(
    directory: Path,
    ifo: str,
    gps_start: int,
    duration: int,
    fs: int,
    strain: np.ndarray,
    good: np.ndarray,
) -> Path:
    """One file; *good* is the 1 Hz DATA flag (strain is set to NaN where it is False)."""
    path = directory / f"{ifo[0]}-{ifo}_GWOSC_O3a_4KHZ_R1-{gps_start}-{duration}.hdf5"
    strain = strain.astype(np.float64).copy()
    strain[~np.repeat(good, fs)] = np.nan
    dq = np.zeros(duration, dtype=np.int32)
    for bit in range(len(DQ_NAMES)):
        dq |= good.astype(np.int32) << bit
    inj = np.full(duration, 0b10111, dtype=np.int32)  # NO_CW_HW_INJ = 0: CW injections on
    with h5py.File(path, "w") as f:
        m = f.create_group("meta")
        m["Detector"] = ifo.encode()
        m["GPSstart"] = np.int64(gps_start)
        m["Duration"] = np.int64(duration)
        m["StrainChannel"] = f"{ifo}:FAKE-STRAIN".encode()
        s = f.create_dataset("strain/Strain", data=strain)
        s.attrs["Xspacing"] = 1.0 / fs
        s.attrs["Xstart"] = np.int64(gps_start)
        q = f.create_group("quality/simple")
        q["DQmask"] = dq
        q["DQShortnames"] = np.array([n.encode() for n in DQ_NAMES])
        i = f.create_group("quality/injections")
        i["Injmask"] = inj
        i["InjShortnames"] = np.array([n.encode() for n in INJ_NAMES])
    return path


def make_dataset(
    directory: Path,
    *,
    ifo: str = "H1",
    gps_start: int = 1238319104,
    n_files: int = 3,
    duration: int = 4096,
    fs: int = 1024,
    signal: Callable[[np.ndarray], np.ndarray] | None = None,
    asd: float = 1e-23,
    seed: int = 1,
    gaps: list[tuple[int, int]] = (),
    glitches: list[tuple[float, float]] = (),
    skip: tuple[int, ...] = (),
) -> list[Path]:
    """Back-to-back files of white noise (one-sided ASD *asd*) plus *signal(t)*.

    *gaps* are GPS ``[start, stop)`` with DATA off; *glitches* are
    ``(gps, amplitude)`` sine-Gaussian bursts at 300 Hz; files whose index is
    in *skip* are not written (a gap between runs).
    """
    rng = np.random.default_rng(seed)
    paths = []
    sigma = asd * np.sqrt(fs / 2)
    for k in range(n_files):
        t0 = gps_start + k * duration
        t = t0 + np.arange(duration * fs) / fs
        x = rng.normal(0.0, sigma, t.size)
        if signal is not None:
            x += signal(t)
        for tg, amp in glitches:
            x += amp * np.exp(-(((t - tg) / 0.02) ** 2)) * np.sin(2 * np.pi * 300 * (t - tg))
        good = np.ones(duration, dtype=bool)
        for a, b in gaps:
            sec = t0 + np.arange(duration)
            good &= ~((sec >= a) & (sec < b))
        if k in skip:
            continue
        paths.append(write_file(directory, ifo, t0, duration, fs, x, good))
    return paths
