"""End-to-end runs on small synthetic GWOSC files (1024 Hz)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fakegwosc import make_dataset, write_file
from readligo import BandSpec, GateParams, PipelineConfig, read_product, run_pipeline
from readligo.inventory import contiguous_runs, scan_directory

EPOCH = 1238319104
P = GateParams(white_fhi=400.0)
F, A = 100.7311, 3e-23


def tone(t: np.ndarray) -> np.ndarray:
    return A * np.cos(2 * np.pi * F * (t - EPOCH) + 0.4)


def _cfg(raw: Path, out: Path, **kw) -> PipelineConfig:
    return PipelineConfig(data_dir=raw, out_dir=out, ifo="H1", bands=[BandSpec(100.68, 0.1), BandSpec(60, 1.0)],
                          gating=kw.pop("gating", P), **kw)


def test_inventory(tmp_path: Path) -> None:
    make_dataset(tmp_path, n_files=4, duration=64, fs=256, skip=(2,))
    files = scan_directory(tmp_path, "H1")
    assert [len(r) for r in contiguous_runs(files)] == [2, 1]
    with pytest.raises(ValueError):
        scan_directory(tmp_path, "V1")


def test_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    glitches = [(EPOCH + 1000.3, 5e-20), (EPOCH + 4095.7, 8e-20)]  # the second straddles a file boundary
    make_dataset(raw, gps_start=EPOCH - 4096, n_files=3, signal=tone, glitches=glitches,
                 gaps=[(EPOCH + 2000, EPOCH + 2300)])
    paths = run_pipeline(_cfg(raw, tmp_path / "out"))
    pr = read_product(paths[0])
    assert pr.fs == 0.2 and pr.n0 == -819 and pr.data.size == 2458  # slots overlapping [-4096, 8192) s
    # gates at both glitches, nowhere else
    peaks = pr.tables["gates"]["gps_peak"] - EPOCH
    assert len(peaks) == 2 and np.allclose(np.sort(peaks), [1000.3, 4095.7], atol=0.05)
    # gaps are exactly zero; the tone keeps its amplitude and phase
    assert np.all(pr.data[pr.window == 0] == 0)
    m = pr.mask
    ph = 2 * np.pi * (F - pr.fmin) * (pr.times - EPOCH)
    c, *_ = np.linalg.lstsq(np.stack([np.cos(ph), -np.sin(ph)], 1)[m], pr.data[m], rcond=None)
    assert abs(np.hypot(*c) / A - 1) < 0.01 and abs(np.arctan2(c[1], c[0]) - 0.4) < 0.01
    # the 1 Hz control band has the input noise level
    q = read_product(paths[1])
    assert abs(q.sqrt_sx / 1e-23 - 1) < 0.03
    assert q.attrs["readligo_git_commit"]
    with pytest.raises(FileExistsError):
        run_pipeline(_cfg(raw, tmp_path / "out"))


def test_file_boundaries_do_not_matter(tmp_path: Path) -> None:
    """ISSUES C6/C8: the same data as 3 files or as 1 file give the same output."""
    split, whole = tmp_path / "split", tmp_path / "whole"
    split.mkdir()
    whole.mkdir()
    paths = make_dataset(split, gps_start=EPOCH, n_files=3, signal=tone, gaps=[(EPOCH + 5000, EPOCH + 5100)])
    import h5py

    xs = []
    for p in paths:
        with h5py.File(p) as f:
            xs.append(f["strain/Strain"][()])
    good = ~np.isnan(np.concatenate(xs))[:: 1024]
    write_file(whole, "H1", EPOCH, 3 * 4096, 1024, np.nan_to_num(np.concatenate(xs)), good)
    a = read_product(run_pipeline(_cfg(split, tmp_path / "o1"))[1])
    b = read_product(run_pipeline(_cfg(whole, tmp_path / "o2"))[1])
    assert a.n0 == b.n0 and np.array_equal(a.mask, b.mask)
    assert np.max(np.abs(a.data - b.data)) <= 1e-12 * np.max(np.abs(b.data))


def test_work_dir_is_private_and_parallel_matches_serial(tmp_path: Path) -> None:
    """ISSUES.md A1 and A4: an existing <work_dir>/H1 is never touched; workers and chunking change nothing."""
    raw = tmp_path / "root" / "H1"  # raw data where the old scratch folder would have been
    raw.mkdir(parents=True)
    make_dataset(raw, gps_start=EPOCH, n_files=2, signal=tone, glitches=[(EPOCH + 3000.2, 5e-20)])
    (raw / "notes.txt").write_text("keep me")
    before = sorted(q.name for q in raw.iterdir())
    a = run_pipeline(_cfg(raw, tmp_path / "o1", work_dir=tmp_path / "root"))
    assert sorted(q.name for q in raw.iterdir()) == before
    assert not any(q.name.startswith("readligo-") for q in (tmp_path / "root").iterdir())
    b = run_pipeline(_cfg(raw, tmp_path / "o2", workers=2, chunk_gb=1e-9))  # one band per chunk
    for pa, pb in zip(a, b):
        qa, qb = read_product(pa), read_product(pb)
        assert np.array_equal(qa.data, qb.data) and np.array_equal(qa.mask, qb.mask)
        assert qa.sqrt_sx == qb.sqrt_sx


def test_skip_existing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    make_dataset(raw, gps_start=EPOCH, n_files=1, signal=tone)
    first = run_pipeline(PipelineConfig(data_dir=raw, out_dir=tmp_path / "o", ifo="H1",
                                        bands=[BandSpec(100.68, 0.1)], gating=P))
    stamp = first[0].stat().st_mtime_ns
    both = run_pipeline(_cfg(raw, tmp_path / "o", skip_existing=True))
    assert both[0] == first[0] and both[0].stat().st_mtime_ns == stamp and both[1].exists()


def test_strain_check(tmp_path: Path) -> None:
    """A DQ-good second of zeros (dead channel) raises, or is dropped on request (ISSUES.md A12)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = make_dataset(raw, gps_start=EPOCH, n_files=1)
    import h5py

    with h5py.File(paths[0], "r+") as f:
        f["strain/Strain"][100 * 1024 : 110 * 1024] = 0.0
    with pytest.raises(ValueError, match="constant"):
        run_pipeline(_cfg(raw, tmp_path / "o"))
    out = run_pipeline(_cfg(raw, tmp_path / "o2", bad_strain="drop"))
    q = read_product(out[1])
    assert q.attrs["dropped_constant_seconds"] == 10
    assert list(q.tables["stretches"]["gps_start"] - EPOCH) == [0, 110]


def test_injection_bits_can_be_required(tmp_path: Path) -> None:
    """ISSUES.md A10: NO_*_HW_INJ bits are accepted in dq_flags; NO_CW_HW_INJ is refused."""
    raw = tmp_path / "raw"
    raw.mkdir()
    make_dataset(raw, gps_start=EPOCH, n_files=1)
    run_pipeline(_cfg(raw, tmp_path / "o", dq_flags=("DATA", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ")))
    with pytest.raises(ValueError, match="CW hardware"):
        _cfg(raw, tmp_path / "o", dq_flags=("DATA", "NO_CW_HW_INJ"))


def test_long_transient_across_file_boundary_is_gated_whole(tmp_path: Path) -> None:
    """ISSUES.md A12: a 200 s burst across a file boundary, longer than the 64 s context."""
    raw = tmp_path / "raw"
    raw.mkdir()
    edge = EPOCH + 4096

    def burst(t: np.ndarray) -> np.ndarray:
        on = (t > edge - 100) & (t < edge + 100)
        return np.where(on, 5e-20 * np.sin(2 * np.pi * 300 * t), 0.0)

    make_dataset(raw, gps_start=EPOCH, n_files=2, signal=burst)
    q = read_product(run_pipeline(_cfg(raw, tmp_path / "o"))[1])
    g = q.tables["gates"]
    assert len(g["gps_start"]) == 1
    assert g["gps_start"][0] <= edge - 100 and g["gps_end"][0] >= edge + 100


def test_cli_plan_and_clean(tmp_path: Path) -> None:
    """plan writes a bands file that clean reads (usable ranges stored); config flags not overridden (A13)."""
    from readligo.cli import main, read_bands_file

    raw = tmp_path / "raw"
    raw.mkdir()
    make_dataset(raw, gps_start=EPOCH, n_files=1, signal=tone)
    bands = tmp_path / "bands.txt"
    assert main(["plan", "--fmin", "100.70", "--fmax", "100.80", "--tspan-days", "1", "-o", str(bands)]) == 0
    rows = read_bands_file(bands)
    assert len(rows) >= 1 and all(len(r) == 4 for r in rows)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"overwrite": True, "gating": {"white_fhi": 400.0}}))
    args = ["clean", "--config", str(cfg), "--data-dir", str(raw), "--out-dir", str(tmp_path / "o"), "--ifo", "H1",
            "--bands-file", str(bands), "-q"]
    assert main(args) == 0
    assert main(args) == 0  # overwrite from --config is kept although --overwrite is not given
    q = read_product(sorted((tmp_path / "o").glob("*.h5"))[0])
    lo, hi = q.attrs["usable_source_band"]
    assert q.fmin < lo < hi < q.fmax
    (tmp_path / "comma.txt").write_text("60,1  # comment\n")
    assert read_bands_file(tmp_path / "comma.txt") == [["60", "1"]]


def test_glitch_storm_is_gated(tmp_path: Path) -> None:
    """ISSUES.md A12: glitches every 3 s through two files of a six-file run are still gated (run reference PSD)."""
    from readligo import RunData, condition_run, detect_gates_run

    raw = tmp_path / "raw"
    raw.mkdir()
    storm = [(EPOCH + 2048 + 3.0 * k + 0.37, 1e-20) for k in range(680)]  # files 2 and 3 (1024 s each)
    make_dataset(raw, gps_start=EPOCH, n_files=6, duration=1024, glitches=storm)
    run = contiguous_runs(scan_directory(raw, "H1"))[0]
    rd = RunData(run, ("DATA",))
    cond = condition_run(rd, P, tmp_path / "c.npy")
    found = {}
    for ref in (True, False):
        p = GateParams(white_fhi=400.0, ref_psd=ref)
        gates, _ = detect_gates_run(cond, p, EPOCH)
        peaks = np.array([g.gps_peak for g in gates]) - EPOCH
        found[ref] = int(np.sum(np.min(np.abs(peaks[:, None] - (np.array([t for t, _ in storm]) - EPOCH)), axis=1)
                                < 0.1)) if peaks.size else 0
    assert found[True] > 0.9 * len(storm) > found[False]


def test_sigterm_removes_scratch(tmp_path: Path) -> None:
    """A job stopped with SIGTERM still deletes its private scratch folder."""
    import subprocess
    import sys
    import time as _time

    raw = tmp_path / "raw"
    raw.mkdir()
    make_dataset(raw, gps_start=EPOCH, n_files=3, signal=tone)
    script = tmp_path / "job.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from readligo import BandSpec, GateParams, PipelineConfig, run_pipeline\n"
        f"run_pipeline(PipelineConfig(data_dir={str(raw)!r}, out_dir={str(tmp_path / 'o')!r}, ifo='H1',\n"
        "    bands=[BandSpec(100.68, 0.1)], gating=GateParams(white_fhi=400.0)))\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)])
    work = tmp_path / "o" / "_work"
    deadline = _time.time() + 60
    while not (work.exists() and any(work.iterdir())) and _time.time() < deadline:
        _time.sleep(0.05)
    proc.terminate()
    assert proc.wait(60) != 0
    assert not work.exists() or not any(work.iterdir())
