"""v1 batch driver: one output file per raw file and band (kept for comparison).

New work should use :func:`readligo.run_pipeline` (``python -m readligo clean``).
"""
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from rich.progress import track

from .io import loaddata, read_meta
from .preprocessing import process_band, processing_attrs
from .provenance import provenance_attrs


DETECTORS = ("H1", "L1")

#: Default directories come from these environment variables (ISSUES.md E8).
DATA_DIR_ENV = "READLIGO_DATA_DIR"
OUT_DIR_ENV = "READLIGO_OUT_DIR"


def _dir_from_env(value: str | Path | None, env: str) -> Path:
    if value is not None:
        return Path(value)
    if os.environ.get(env):
        return Path(os.environ[env])
    raise ValueError(f"no directory given and ${env} is not set")


def _format_freq(f: float) -> str:
    """Shortest exact decimal form of *f* (``100.0 -> "100"``, ``100.68 -> "100.68"``).

    Distinct floats always give distinct strings, so band filenames never collide.
    """
    return np.format_float_positional(float(f), trim="-")


def output_path(
    out_dir: str | Path,
    source_filename: str,
    f_band: tuple[float, float],
) -> Path:
    """Path of the processed file for one source file and frequency band."""
    band = f"{_format_freq(f_band[0])}-{_format_freq(f_band[1])}"
    return Path(out_dir) / f"{Path(source_filename).stem}_f_band_{band}Hz.hdf5"


def source_attrs(source_path: str | Path) -> dict[str, str | int]:
    """Detector and provenance of the raw GWOSC file, from its ``meta`` group."""
    meta = read_meta(source_path)
    detector = meta["Detector"]
    if detector not in DETECTORS:
        raise ValueError(f"{source_path}: detector {detector!r} is not one of {DETECTORS}")
    return {
        "detector": detector,
        "source_gps_start": meta["GPSstart"],
        "source_duration": meta["Duration"],
        "strain_channel": meta["StrainChannel"],
    }


def save_hdf5_file(
    out_dir: str | Path,
    source_filename: str,
    f_band: tuple[float, float],
    gps_times: np.ndarray,
    cleaned: np.ndarray,
    sample_mask: np.ndarray,
    fs_new: float,
    breakdown: dict[str, int | float] | None,
    *,
    attrs: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Write one processed band to HDF5.

    *attrs* (e.g. from :func:`source_attrs`, :func:`processing_attrs` and
    :func:`provenance_attrs`) are written as root attributes alongside
    ``source_file``, ``f_band`` and ``fs_new``.
    """
    out_path = output_path(out_dir, source_filename, f_band)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True to replace it."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        # ── metadata ──────────────────────────────────────────────
        f.attrs["source_file"]  = source_filename
        f.attrs["f_band"]       = list(f_band)   # [f_min, f_max] Hz
        f.attrs["fs_new"]       = fs_new
        for key, val in (attrs or {}).items():
            f.attrs[key] = val

        # ── breakdown stats ────────────────────────────────────────
        if breakdown is not None:
            stats = f.require_group("stats")
            for key, val in breakdown.items():
                stats.attrs[key] = val

        # ── time series ───────────────────────────────────────────
        f.create_dataset("gps_times",   data=gps_times,               compression="gzip")
        f.create_dataset("cleaned_re",  data=np.real(cleaned),            compression="gzip")
        f.create_dataset("cleaned_im",  data=np.imag(cleaned),            compression="gzip")
        f.create_dataset("sample_mask", data=sample_mask.astype(np.uint8), compression="gzip")


def process_ligo_files(
    f_band: tuple[float, float],
    nperseg: int = 64,
    fap: float = 0.001,
    bound: int = 15,
    data_dirname: str | Path | None = None,
    process_dirname: str | Path | None = None,
    *,
    overwrite: bool = False,
    loss_stats: bool = True,
) -> None:
    """Run the v1 chain on every file in *data_dirname* (default ``$READLIGO_DATA_DIR``).

    ``bound`` defaults to 15, the value the stored v1 products were made with
    (ISSUES.md V1); it is recorded in each output.
    """
    data_dirname = _dir_from_env(data_dirname, DATA_DIR_ENV)
    process_dirname = _dir_from_env(process_dirname, OUT_DIR_ENV)
    hdf5_files = list(Path(data_dirname).glob("*.hdf5"))
    hdf5_files_sorted = sorted(
        hdf5_files,
        key=lambda p: int(p.stem.split('-')[-2])
    )
    # Fail before processing anything rather than partway through the run.
    if not overwrite:
        existing = [
            p for p in (output_path(process_dirname, h.name, f_band) for h in hdf5_files_sorted)
            if p.exists()
        ]
        if existing:
            raise FileExistsError(
                f"{len(existing)} output file(s) already exist for band {f_band}, "
                f"e.g. {existing[0]}; pass overwrite=True to replace them."
            )
    fs = 4096  # GWOSC 4 kHz release
    run_attrs = provenance_attrs()
    for files in track(hdf5_files_sorted, description="Processing files"):
        strain, time, channel_dict = loaddata(str(files))
        assert isinstance(time, np.ndarray), "expected time vector"
        gpsStart = time[0]
        attrs = {
            **source_attrs(files),
            **processing_attrs(gpsStart, f_band, fs=fs, nperseg=nperseg, fap=fap, bound=bound),
            **run_attrs,
        }
        gps_times, cleaned, sample_mask, fs_new, breakdown = process_band(
            strain,
            channel_dict,
            gpsStart,
            f_band,
            fs=fs,
            fap=fap,
            nperseg=nperseg,
            bound=bound,
            plot=False,
            loss_stats=loss_stats,
        )
        save_hdf5_file(
            process_dirname,
            files.name,
            f_band,
            gps_times,
            cleaned,
            sample_mask,
            fs_new,
            breakdown,
            attrs=attrs,
            overwrite=overwrite,
        )
