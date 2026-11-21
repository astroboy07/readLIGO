"""The v2 cleaning pipeline: raw GWOSC files to one band product per band.

Steps (``docs/cleaning_pipeline.md`` explains each and why):

1. **Inventory.** Find the detector's files, check each against its own
   metadata, and group them into runs of back-to-back files. Read the DQ
   and injection bits.
2. **Strain check.** Every DQ-good second must be finite and not constant
   (a dead channel); such seconds raise, or are dropped (``bad_strain``).
3. **Conditioning**, per run as one series: DQ stretches, zero-phase 10/15 Hz
   high-pass, edge tapers (:func:`conditioning.condition_run`).
4. **Gating.** Loud transients are found on whitened broadband strain, file
   by file with context, and merged into one GPS gate list per run
   (:func:`conditioning.detect_gates_run`).
5. **Channelizing.** One FFT per 2 h block serves every band: real series at
   exactly 2B, heterodyned at fmin against the global epoch
   (:class:`channelize.FFTChannelizer`). Bands are processed in chunks of at
   most ``chunk_gb``; each block's output goes straight to a memory-mapped
   scratch file, so memory does not grow with the number of bands or days.
6. **Window and mask** of each output sample from the DQ stretches, tapers and
   gates (computed once per bandwidth); samples with no live data are set to
   exactly 0.
7. **Vetoes, noise and lines** per band: a narrowband outlier veto and an
   excess-power veto for what the broadband gate cannot see, repeated until
   nothing new is found (:func:`noise.veto_band`), then the noise per segment
   and the lines (:func:`noise.estimate_noise`).
8. **Write** one HDF5 product per band (:mod:`products`).

Steps 2-4 run in parallel over files and runs, step 5 over blocks and steps
6-8 over bands, with ``workers`` processes.

Usage::

    from readligo import BandSpec, PipelineConfig, run_pipeline
    cfg = PipelineConfig(data_dir=..., out_dir=..., ifo="H1",
                         bands=[BandSpec(100.68, 0.1), BandSpec(145.4, 0.1)])
    paths = run_pipeline(cfg)

or ``python -m readligo clean --help``.
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import fft as sfft

from .channelize import FFTChannelizer
from .conditioning import (
    ConditionedRun,
    ConditionedStrain,
    Gate,
    RunData,
    check_strain,
    condition_run,
    detect_gates_run,
    output_window,
)
from .config import BandSpec, PipelineConfig
from .inventory import contiguous_runs, scan_directory
from .noise import (
    estimate_noise,
    gwosc_antialias_gain,
    inband_shape,
    mean_psd,
    veto_band,
)
from .planning import DOPPLER_MAX
from .products import BandProduct, product_path, write_product
from .provenance import provenance_attrs

__all__ = ["check_data", "product_range", "run_pipeline"]

logger = logging.getLogger(__name__)


def _restrict(run: RunData, gps_start: float | None, gps_end: float | None) -> None:
    """Mark seconds outside ``[gps_start, gps_end)`` as not good."""
    t = run.gps_start + np.arange(run.duration)
    if gps_start is not None:
        run.good &= t >= gps_start
    if gps_end is not None:
        run.good &= t + 1 <= gps_end


def product_range(span_lo: int, span_hi: int, epoch: int, fs: Fraction) -> tuple[int, int]:
    """Output samples ``[n0, n1)`` whose slots overlap the live span ``[span_lo, span_hi)``.

    Sample n stands for ``[t_n - 1/(2 fs), t_n + 1/(2 fs))`` with
    ``t_n = epoch + n / fs`` (ISSUES.md A15: the product used to stop up to half
    a slot early and could start with an empty slot).
    """
    n0 = math.floor((Fraction(span_lo) - epoch) * fs + Fraction(1, 2))
    n1 = math.ceil((Fraction(span_hi) - epoch) * fs + Fraction(1, 2))
    return n0, n1


# ---------------------------------------------------------------------------
# Process pool (workers > 1) or in-process execution (workers == 1)
# ---------------------------------------------------------------------------
_G: dict[str, Any] = {}  # per-process shared state, set by _init


def _init(state: dict[str, Any]) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _G.clear()
    _G.update(state)


class _Serial(Executor):
    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        from concurrent.futures import Future

        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001 - re-raised by fut.result()
            fut.set_exception(e)
        return fut


@contextmanager
def _executor(workers: int, state: dict[str, Any]) -> Iterator[Executor]:
    if workers <= 1:
        saved = dict(_G)
        _init(state)
        try:
            yield _Serial()
        finally:
            _G.clear()
            _G.update(saved)
        return
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init, initargs=(state,))
    try:
        yield ex
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)  # do not start queued tasks after an error or a stop
        raise
    ex.shutdown(wait=True)


@contextmanager
def _stop_on_sigterm() -> Iterator[None]:
    """Turn SIGTERM (kill, scheduler time limits) into SystemExit, so the scratch folder is removed."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _map(ex: Executor, fn: Callable[..., Any], items: Iterable[Any]) -> list[Any]:
    futs = [ex.submit(fn, *(it if isinstance(it, tuple) else (it,))) for it in items]
    return [f.result() for f in futs]


# ---------------------------------------------------------------------------
# Tasks (top level so they can run in worker processes)
# ---------------------------------------------------------------------------
def _task_check(path: str, good: npt.NDArray[np.bool_], fs: int) -> dict[str, npt.NDArray[np.int64]]:
    return check_strain(path, good, fs)


@dataclass
class _RunResult:
    cond: ConditionedRun
    gates: list[Gate]
    blocks: list[dict[str, float]]
    cw_on_seconds: int
    seconds: float


def _task_condition(i: int, rd: RunData) -> _RunResult:
    cfg: PipelineConfig = _G["cfg"]
    t0 = time.perf_counter()
    dtype = np.float32 if cfg.work_dtype == "float32" else np.float64
    cond = condition_run(rd, cfg.gating, Path(_G["work"]) / f"run{i:04d}_{rd.gps_start}.npy", dtype=dtype)
    gates, blocks = detect_gates_run(cond, cfg.gating, cfg.epoch, context=cfg.gate_context)
    kept = np.zeros(rd.duration, dtype=bool)
    for a, b in cond.stretches:
        kept[a - rd.gps_start : b - rd.gps_start] = True
    cw_on = int(np.sum(kept & ~rd.cw_injection_off))
    return _RunResult(cond, gates, blocks, cw_on, time.perf_counter() - t0)


def _task_psd(res: _RunResult) -> tuple[npt.NDArray[np.float64], int]:
    cfg: PipelineConfig = _G["cfg"]
    tap = cfg.gating.gate_taper
    spans = (np.array([g.gps_start - tap for g in res.gates]), np.array([g.gps_end + tap for g in res.gates]))
    c = res.cond
    return mean_psd(c.strain(), c.gps_start, c.fs, c.stretches, spans, seg=cfg.channelizer.flatten_psd_seg,
                    edge=cfg.gating.edge_taper + 2.0)


@dataclass(frozen=True)
class _Chunk:
    """Bands channelized together; one scratch array per distinct bandwidth."""

    bands: tuple[int, ...]  # indices into cfg.bands
    rows: dict[Fraction, tuple[int, ...]]  # bandwidth -> band indices, in row order
    files: dict[Fraction, str]  # bandwidth -> .npy path, shape (len(rows[B]), n1 - n0)
    ranges: dict[Fraction, tuple[int, int]]  # bandwidth -> (n0, n1)


def _task_channelize(chunk: _Chunk, blocks: Sequence[int]) -> int:
    """Channelize *blocks* for every band of *chunk* into its scratch arrays."""
    chan: FFTChannelizer = _G["chan"]
    reader: ConditionedStrain = _G["reader"]
    out = {B: np.load(path, mmap_mode="r+") for B, path in chunk.files.items()}
    pos = {i: (B, r) for B, idx in chunk.rows.items() for r, i in enumerate(idx)}
    try:
        for b in blocks:
            block = np.asarray(reader(chan.block_start(b), chan.N), dtype=np.float64)
            if not block.any():
                continue  # the scratch arrays start as zeros
            X = sfft.rfft(block)
            del block
            for i in chunk.bands:
                B, r = pos[i]
                n0, n1 = chunk.ranges[B]
                sph = chan.samples_per_hop(i)
                a, z = b * sph, (b + 1) * sph
                lo, hi = max(a, n0), min(z, n1)
                if hi > lo:
                    y = chan.band_from_spectrum(X, i, b)
                    out[B][r, lo - n0 : hi - n0] = y[lo - a : hi - a]
    finally:
        for arr in out.values():
            arr.flush()
        del out
    return len(blocks)


def _near(times: npt.NDArray[np.float64], lo: npt.NDArray[np.float64], hi: npt.NDArray[np.float64],
          reach: float) -> npt.NDArray[np.bool_]:
    """Boolean: each time within *reach* of some interval ``[lo, hi]``."""
    out = np.zeros(times.size, dtype=bool)
    if lo.size == 0:
        return out
    order = np.argsort(lo)
    l2, h2 = lo[order] - reach, np.maximum.accumulate(hi[order] + reach)
    j = np.searchsorted(l2, times, side="right") - 1
    ok = j >= 0
    out[ok] = times[ok] <= h2[j[ok]]
    return out


def _task_finalize(i: int, B: Fraction, row: int, data_path: str) -> dict[str, Any]:
    """Window, noise, veto and product of band *i* from its row of the chunk array."""
    cfg: PipelineConfig = _G["cfg"]
    band: BandSpec = cfg.bands[i]
    fs = band.fs
    n0, n1 = _G["ranges"][B]
    data = np.array(np.load(data_path, mmap_mode="r")[row], dtype=np.float64)
    win_path, mask_path = _G["windows"][B]
    window = np.load(win_path).copy()
    mask = np.load(mask_path).copy()
    data[window == 0] = 0.0
    guard = _G["guard"][i]
    kw = dict(fmin=float(band.fmin), band=float(band.band), epoch=cfg.epoch, n0=n0, guard=guard, params=cfg.noise)
    nv, xv, scale, passes = veto_band(data, window, mask, epoch=cfg.epoch, n0=n0, fs=float(fs),
                                      segment=cfg.noise.segment, params=cfg.veto)
    v_a, v_b, v_peak = nv["start"], nv["stop"], nv["peak"]
    x_a, x_b, x_mean = xv["start"], xv["stop"], xv["peak"]
    noise = estimate_noise(data, window, mask, **kw)
    # leakage diagnostics: mean z = x^2/s^2 of fully good samples next to gates and stretch edges
    times = cfg.epoch + (n0 + np.arange(data.size)) / float(fs)
    z = data**2 / scale
    reach = 3.0 / float(fs)
    g_lo, g_hi = _G["gate_spans"]
    e_lo, e_hi = _G["edge_spans"]
    near_g = mask & _near(times, g_lo, g_hi, reach)
    near_e = mask & _near(times, e_lo, e_hi, reach)
    far = mask & ~near_g & ~near_e
    gain = gwosc_antialias_gain([float(band.fmin), float(band.fmax)])
    chan: FFTChannelizer = _G["chan"]
    chan_shape = chan.shapes[i]
    g_edges = chan.shape_gain(i, [float(band.fmin), float(band.fmax)])
    usable = band.usable
    d_lo = (float(band.fmin) + guard) / (1 - DOPPLER_MAX)
    d_hi = (float(band.fmax) - guard) / (1 + DOPPLER_MAX)
    attrs = {
        **_G["common_attrs"],
        "fmin_exact": str(band.fmin),
        "band_exact": str(band.band),
        "guard_hz": guard,
        "flat_band": [float(band.fmin) + guard, float(band.fmax) - guard],
        "usable_source_band": [float("nan"), float("nan")] if usable is None else list(usable),
        "usable_doppler_only": [d_lo, d_hi],
        "sqrt_sx_harmonic": noise.sqrt_sx_harmonic,
        "sqrt_sx_median": noise.sqrt_sx_median,
        "variance_ratio": noise.variance_ratio,
        "n_noise_segments": noise.n_segments_used,
        "n_lines": int(noise.lines["freq"].size),
        "n_vetoes": int(v_a.size),
        "vetoed_seconds": float(np.sum(v_b - v_a) / fs),
        "n_excess_vetoes": int(x_a.size),
        "excess_vetoed_seconds": float(np.sum(x_b - x_a) / fs),
        "veto_passes": passes,
        "near_gate_mean_z": float(z[near_g].mean()) if near_g.any() else float("nan"),
        "near_edge_mean_z": float(z[near_e].mean()) if near_e.any() else float("nan"),
        "far_mean_z": float(z[far].mean()) if far.any() else float("nan"),
        "inband_shape": [float("nan")] if chan_shape is None else list(chan_shape),
        "inband_gain_fmin": float(g_edges[0]),
        "inband_gain_fmax": float(g_edges[1]),
        "inband_gain_convention": (
            "every strain component at frequency f is scaled by g(f) = exp(-sum_k a_k u^k / 2), u = f - fc, "
            "with (fc, a1, a2, ...) = inband_shape (g = 1 if NaN); it makes the band's mean noise PSD flat at "
            "sqrt_sx^2. A CW of strain amplitude A at f appears with amplitude A g(f) gwosc_antialias_gain(f)."
        ),
        "gwosc_antialias_gain_fmin": float(gain[0]),
        "gwosc_antialias_gain_fmax": float(gain[1]),
        "rustyprune_band_ok": bool((band.band * 10).denominator == 1 and 1 <= band.band * 10 <= 10),
    }
    product = BandProduct(
        ifo=cfg.ifo, fmin=float(band.fmin), band=float(band.band), epoch=float(cfg.epoch), n0=n0,
        data=data, window=window, mask=mask, sqrt_sx=noise.sqrt_sx,
        tables={
            **_G["tables_common"],
            "noise_segments": noise.segments,
            "spectrum": noise.spectrum,
            "lines": noise.lines,
            "vetoes": {
                "gps_start": cfg.epoch + (n0 + v_a) / float(fs),
                "gps_end": cfg.epoch + (n0 + v_b) / float(fs),
                "peak_z": v_peak,
            },
            "excess_vetoes": {
                "gps_start": cfg.epoch + (n0 + x_a) / float(fs),
                "gps_end": cfg.epoch + (n0 + x_b) / float(fs),
                "mean_z": x_mean,
            },
        },
        attrs=attrs,
    )
    path = Path(_G["paths"][i])
    write_product(product, path, overwrite=cfg.overwrite)
    return {
        "band": band.label(), "path": str(path), "n": int(data.size), "fs": float(fs), "sqrt_sx": noise.sqrt_sx,
        "duty": product.duty_factor, "n_lines": attrs["n_lines"], "n_vetoes": int(v_a.size),
        "n_excess_vetoes": int(x_a.size),
        "near_gate_mean_z": attrs["near_gate_mean_z"],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _chunks(cfg: PipelineConfig, todo: Sequence[int], ranges: dict[Fraction, tuple[int, int]],
            work: Path) -> list[_Chunk]:
    """Split bands (sorted by frequency) into chunks of at most ``chunk_gb`` of output."""
    limit = cfg.chunk_gb * 1e9
    groups: list[list[int]] = []
    size = 0.0
    for i in sorted(todo, key=lambda k: cfg.bands[k].fmin):
        B = cfg.bands[i].band
        nb = 8.0 * (ranges[B][1] - ranges[B][0])
        if groups and size + nb <= limit:
            groups[-1].append(i)
            size += nb
        else:
            groups.append([i])
            size = nb
    out = []
    for c, g in enumerate(groups):
        rows: dict[Fraction, list[int]] = {}
        for i in g:
            rows.setdefault(cfg.bands[i].band, []).append(i)
        files = {}
        for B, idx in rows.items():
            path = work / f"chunk{c:04d}_B{float(B):.6f}.npy"
            files[B] = str(path)
        out.append(_Chunk(tuple(g), {B: tuple(v) for B, v in rows.items()}, files, dict(ranges)))
    return out


def check_data(cfg: PipelineConfig) -> dict[str, Any]:
    """Inventory, DQ and strain check only (no processing): what :func:`run_pipeline` would use.

    Returns a summary: files, runs, GPS span, DQ-good seconds, and the GPS
    seconds that are DQ-good but have NaN/inf or constant strain.
    """
    rng = None
    if cfg.gps_start is not None or cfg.gps_end is not None:
        rng = (cfg.gps_start if cfg.gps_start is not None else -np.inf,
               cfg.gps_end if cfg.gps_end is not None else np.inf)
    files = scan_directory(cfg.data_dir, cfg.ifo, rng)
    runs = contiguous_runs(files)
    run_data = []
    for r in runs:
        rd = RunData(r, cfg.dq_flags)
        _restrict(rd, cfg.gps_start, cfg.gps_end)
        run_data.append(rd)
    jobs = []
    for rd in run_data:
        for f in rd.files:
            o = f.gps_start - rd.gps_start
            g = rd.good[o : o + f.duration]
            if g.any():
                jobs.append((rd.gps_start + o, (str(f.path), g, rd.fs)))
    with _executor(cfg.workers, {"cfg": cfg}) as ex:
        res = _map(ex, _task_check, [j[1] for j in jobs])
    bad = {k: sorted(int(t0 + s) for (t0, _), r in zip(jobs, res) for s in r[k]) for k in ("nonfinite", "constant")}
    return {
        "ifo": cfg.ifo,
        "n_files": len(files),
        "n_runs": len(runs),
        "gps_span": [files[0].gps_start, files[-1].gps_end] if files else None,
        "good_seconds": int(sum(int(rd.good.sum()) for rd in run_data)),
        "nonfinite_seconds": bad["nonfinite"],
        "constant_seconds": bad["constant"],
    }


def run_pipeline(cfg: PipelineConfig) -> list[Path]:
    """Run every step for one detector and write one product per band.

    Returns
    -------
    list of Path
        The product files, in the order of ``cfg.bands`` (including skipped ones).

    Raises
    ------
    FileExistsError
        If a product exists and neither ``cfg.overwrite`` nor
        ``cfg.skip_existing`` is set (checked before any processing).
    ValueError
        If no file or no DQ-good data is found, a setting cannot work with the
        data (checked before any processing), or a DQ-good second has unusable
        strain and ``bad_strain == "raise"``.
    """
    t_begin = time.perf_counter()
    p = cfg.gating
    rng = None
    if cfg.gps_start is not None or cfg.gps_end is not None:
        rng = (cfg.gps_start if cfg.gps_start is not None else -np.inf,
               cfg.gps_end if cfg.gps_end is not None else np.inf)
    files = scan_directory(cfg.data_dir, cfg.ifo, rng)
    if not files:
        raise ValueError(f"no {cfg.ifo} files in {cfg.data_dir} for GPS range {rng}")
    runs = contiguous_runs(files)
    logger.info("%s: %d files in %d contiguous runs", cfg.ifo, len(files), len(runs))

    # -- 1. read DQ, find the span, check every setting against the data -------
    run_data = []
    for r in runs:
        rd = RunData(r, cfg.dq_flags)
        _restrict(rd, cfg.gps_start, cfg.gps_end)
        if rd.good.any():
            run_data.append(rd)
    if not run_data:
        raise ValueError("no DQ-good data in the selected files")
    fs_in = {rd.fs for rd in run_data}
    if len(fs_in) != 1:
        raise ValueError(f"runs have different sample rates {sorted(fs_in)}")
    fs_in = fs_in.pop()
    if p.detect and p.white_fhi + p.white_edge >= fs_in / 2:
        raise ValueError(
            f"whitening band edge {p.white_fhi} Hz is too close to Nyquist ({fs_in / 2} Hz); "
            "lower GateParams.white_fhi"
        )
    cp = cfg.channelizer
    chan = FFTChannelizer(  # raises now, not after hours, for bands too close to Nyquist etc.
        [(b.fmin, b.band) for b in cfg.bands], cfg.epoch, L=cp.kernel_length, hop=cp.hop,
        atten_db=cp.atten_db, edge=cp.edge, fs_in=fs_in,
    )
    span_lo = run_data[0].gps_start + int(np.flatnonzero(run_data[0].good)[0])
    span_hi = run_data[-1].gps_start + int(np.flatnonzero(run_data[-1].good)[-1]) + 1

    paths = [
        product_path(cfg.out_dir, cfg.ifo, float(b.fmin), float(b.band), span_lo, span_hi - span_lo)
        for b in cfg.bands
    ]
    todo = list(range(len(cfg.bands)))
    existing = [i for i in todo if paths[i].exists()]
    if existing and not cfg.overwrite:
        if cfg.skip_existing:
            todo = [i for i in todo if i not in set(existing)]
            logger.info("skipping %d existing products", len(existing))
        else:
            raise FileExistsError(
                f"{len(existing)} product(s) exist, e.g. {paths[existing[0]]}; pass overwrite or skip_existing"
            )
    if not todo:
        return paths

    made_parent = not cfg.work_path.exists()
    cfg.work_path.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"readligo-{cfg.ifo}-", dir=cfg.work_path))  # ISSUES.md A1
    logger.info("scratch folder %s", work)
    stop = _stop_on_sigterm()
    stop.__enter__()
    try:
        with _executor(cfg.workers, {"cfg": cfg, "work": str(work)}) as ex:
            # -- 2. strain check ------------------------------------------------
            t0 = time.perf_counter()
            jobs = []
            for rd in run_data:
                for f in rd.files:
                    o = f.gps_start - rd.gps_start
                    g = rd.good[o : o + f.duration]
                    if g.any():
                        jobs.append((rd, o, (str(f.path), g, rd.fs)))
            res = _map(ex, _task_check, [j[2] for j in jobs])
            n_nonfinite = n_constant = 0
            bad_gps: list[int] = []
            for (rd, o, _), r in zip(jobs, res):
                for key in ("nonfinite", "constant"):
                    if r[key].size:
                        rd.good[o + r[key]] = False
                        bad_gps.extend((rd.gps_start + o + r[key][:5]).tolist())
                n_nonfinite += int(r["nonfinite"].size)
                n_constant += int(r["constant"].size)
            if n_nonfinite + n_constant:
                msg = (f"{n_nonfinite} DQ-good second(s) with NaN/inf strain and {n_constant} with constant "
                       f"strain, e.g. GPS {sorted(bad_gps)[:5]}")
                if cfg.bad_strain == "raise":
                    raise ValueError(msg + "; pass bad_strain='drop' to treat them as DQ-bad")
                logger.warning("%s: dropped", msg)
            run_data = [rd for rd in run_data if rd.good.any()]
            if not run_data:
                raise ValueError("no usable data after the strain check")
            logger.info("checked %d files in %.1f s", len(jobs), time.perf_counter() - t0)

            # -- 3-4. condition and gate each run ----------------------------------
            t0 = time.perf_counter()
            results: list[_RunResult] = _map(ex, _task_condition, list(enumerate(run_data)))
            for i, (rd, r) in enumerate(zip(run_data, results)):
                logger.debug("run %d/%d: GPS %d +%d s, %d files, %d stretches, %d gates (%.1f s)",
                             i + 1, len(run_data), rd.gps_start, rd.duration, len(rd.files),
                             len(r.cond.stretches), len(r.gates), r.seconds)
            # -- in-band flattening: mean PSD of the gate-free conditioned strain ---
            n_psd = 0
            t_cond = time.perf_counter() - t0
            if cp.flatten:
                t1 = time.perf_counter()
                parts = _map(ex, _task_psd, results)
                n_psd = sum(c for _, c in parts)
                if n_psd:
                    psd = sum(t for t, _ in parts) / n_psd
                    freqs = np.fft.rfftfreq(cp.flatten_psd_seg * fs_in, 1.0 / fs_in)
                    for i in todo:
                        b = cfg.bands[i]
                        chan.set_shape(i, inband_shape(freqs, psd, n_psd, float(b.fmin), float(b.band),
                                                       min_change=cp.flatten_min, max_range=cp.flatten_max_range))
                n_flat = sum(chan.shapes[i] is not None for i in todo)
                logger.info("in-band PSD from %d segments of %d s: %d of %d bands flattened (%.1f s)", n_psd,
                            cp.flatten_psd_seg, n_flat, len(todo), time.perf_counter() - t1)
        conds = [r.cond for r in results]
        gates = sorted((g for r in results for g in r.gates), key=lambda g: g.k_start)
        gate_blocks = [b for r in results for b in r.blocks]
        cw_on_seconds = sum(r.cw_on_seconds for r in results)
        stretches = [s for c in conds for s in c.stretches]
        dropped = [s for c in conds for s in c.dropped]
        if not stretches:
            raise ValueError("no DQ-good stretch longer than min_stretch")
        live_seconds = sum(b - a for a, b in stretches)
        logger.info("conditioned %d runs: %d stretches, %d s live, %d gates (%.1f s)", len(conds),
                    len(stretches), live_seconds, len(gates), t_cond)

        # -- common metadata ---------------------------------------------------
        common_attrs = {
            "config": cfg.to_dict(),
            "dq_flags": list(cfg.dq_flags),
            "strain_channel": ",".join(sorted({c for rd in run_data for c in rd.strain_channels})),
            "span_gps_start": span_lo,
            "span_gps_end": span_hi,
            "live_seconds": live_seconds,
            "dropped_short_seconds": sum(b - a for a, b in dropped),
            "dropped_nonfinite_seconds": n_nonfinite,
            "dropped_constant_seconds": n_constant,
            "n_gates": len(gates),
            "gated_zero_seconds": float(sum(g.k_end - g.k_start for g in gates) / fs_in),
            "cw_hwinj_on_fraction": cw_on_seconds / live_seconds,
            "inband_psd_segments": n_psd,
            **chan.describe(),
            **p.as_attrs(),
            **provenance_attrs(),
        }
        tables_common = {
            "gates": {
                "gps_start": np.array([g.gps_start for g in gates]),
                "gps_end": np.array([g.gps_end for g in gates]),
                "gps_peak": np.array([g.gps_peak for g in gates]),
                "peak_sigma": np.array([g.peak for g in gates]),
            },
            "stretches": {
                "gps_start": np.array([a for a, _ in stretches], dtype=np.int64),
                "gps_end": np.array([b for _, b in stretches], dtype=np.int64),
            },
            "dropped_stretches": {
                "gps_start": np.array([a for a, _ in dropped], dtype=np.int64),
                "gps_end": np.array([b for _, b in dropped], dtype=np.int64),
            },
            "runs": {
                "gps_start": np.array([rd.gps_start for rd in run_data], dtype=np.int64),
                "gps_end": np.array([rd.gps_end for rd in run_data], dtype=np.int64),
                "n_files": np.array([len(rd.files) for rd in run_data], dtype=np.int64),
            },
            "files": {
                "name": np.array([f.path.name for rd in run_data for f in rd.files]),
                "gps_start": np.array([f.gps_start for rd in run_data for f in rd.files], dtype=np.int64),
                "duration": np.array([f.duration for rd in run_data for f in rd.files], dtype=np.int64),
            },
            "gate_blocks": {k: np.array([b[k] for b in gate_blocks]) for k in
                            ("gps_start", "sigma", "n_psd_segments", "n_gates")},
        }

        # -- 5. windows (once per bandwidth) --------------------------------------
        ranges: dict[Fraction, tuple[int, int]] = {}
        windows: dict[Fraction, tuple[str, str]] = {}
        for i in todo:
            B = cfg.bands[i].band
            if B in ranges:
                continue
            n0, n1 = product_range(span_lo, span_hi, cfg.epoch, 2 * B)
            ranges[B] = (n0, n1)
            w, m = output_window(stretches, gates, p, epoch=cfg.epoch, fs_out=float(2 * B), n0=n0, n=n1 - n0)
            wp, mp_ = work / f"window_B{float(B):.6f}.npy", work / f"mask_B{float(B):.6f}.npy"
            np.save(wp, w)
            np.save(mp_, m)
            windows[B] = (str(wp), str(mp_))
        tap, edge = p.gate_taper, p.edge_taper
        e_starts = np.array([a for a, _ in stretches], dtype=np.float64)
        e_stops = np.array([b for _, b in stretches], dtype=np.float64)
        state = {
            "cfg": cfg,
            "work": str(work),
            "chan": chan,
            "reader": ConditionedStrain(conds, gates, p, cfg.epoch),
            "ranges": ranges,
            "windows": windows,
            "guard": [chan.guard(i) for i in range(len(cfg.bands))],
            "paths": [str(q) for q in paths],
            "common_attrs": common_attrs,
            "tables_common": tables_common,
            "gate_spans": (np.array([g.gps_start - tap for g in gates]), np.array([g.gps_end + tap for g in gates])),
            "edge_spans": (np.concatenate([e_starts, e_stops - edge]), np.concatenate([e_starts + edge, e_stops])),
        }

        # -- 6-8. channelize in chunks, then window, noise, veto, write per band ---
        chunks = _chunks(cfg, todo, ranges, work)
        blocks = list(chan.blocks_for(Fraction(span_lo) - cp.hop, Fraction(span_hi) + cp.hop))
        n_tasks = max(1, min(len(blocks), 4 * cfg.workers))
        block_groups = [blocks[k::n_tasks] for k in range(n_tasks)]
        summaries: list[dict[str, Any]] = []
        with _executor(cfg.workers, state) as ex:
            for c, chunk in enumerate(chunks):
                t0 = time.perf_counter()
                for B, path in chunk.files.items():
                    n0, n1 = ranges[B]
                    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                                    shape=(len(chunk.rows[B]), n1 - n0))
                    del arr
                _map(ex, _task_channelize, [(chunk, g) for g in block_groups if g])
                t1 = time.perf_counter()
                jobs = [(i, B, r, chunk.files[B]) for B, idx in chunk.rows.items() for r, i in enumerate(idx)]
                summaries.extend(_map(ex, _task_finalize, jobs))
                for path in chunk.files.values():
                    Path(path).unlink(missing_ok=True)
                logger.info("chunk %d/%d: %d bands, channelized in %.1f s, products in %.1f s",
                            c + 1, len(chunks), len(chunk.bands), t1 - t0, time.perf_counter() - t1)
        for s in summaries:
            logger.debug("%s %s: %d samples at %s Hz, sqrt_sx %.3e, duty %.3f, %d lines, %d vetoes -> %s",
                         cfg.ifo, s["band"], s["n"], s["fs"], s["sqrt_sx"], s["duty"], s["n_lines"],
                         s["n_vetoes"], Path(s["path"]).name)
        summary = {
            "ifo": cfg.ifo,
            "span": [span_lo, span_hi],
            "live_seconds": live_seconds,
            "n_gates": len(gates),
            "dropped_nonfinite_seconds": n_nonfinite,
            "dropped_constant_seconds": n_constant,
            "products": [str(q) for q in paths],
            "skipped_existing": [str(paths[i]) for i in existing] if cfg.skip_existing else [],
            "bands": summaries,
            "runtime_s": time.perf_counter() - t_begin,
        }
        stamp = time.strftime("%Y%m%dT%H%M%S")
        name = f"{cfg.ifo}_pipeline_summary_{span_lo}-{span_hi - span_lo}_{stamp}_{os.getpid()}.json"
        (cfg.out_dir / name).write_text(json.dumps(summary, indent=2, default=float) + "\n")
        logger.info("%s: %d products in %.1f s", cfg.ifo, len(todo), time.perf_counter() - t_begin)
    finally:
        stop.__exit__(None, None, None)
        if cfg.keep_work:
            logger.info("kept scratch folder %s", work)
        else:
            shutil.rmtree(work, ignore_errors=True)
            if made_parent:
                try:
                    cfg.work_path.rmdir()  # only if empty: other jobs' folders are never touched
                except OSError:
                    pass
    return paths
