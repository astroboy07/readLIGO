import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy
from rich.progress import track
from .io import loaddata, read_meta
from .preprocessing import process_band, processing_attrs


DETECTORS = ("H1", "L1")

DATA_DIRNAME = "/Users/saifa/Library/CloudStorage/OneDrive-weizmann.ac.il/O3a_marvin/"
OUT_DIRNAME = "/Users/saifa/Library/CloudStorage/OneDrive-weizmann.ac.il/O3a_marvin_processed"


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


def _git(*args: str) -> str | None:
    """Run git in the package directory; None if git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def provenance_attrs() -> dict[str, str | bool]:
    """Code and library versions, for reproducing a product later."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".")
    return {
        "readligo_git_commit": commit or "unknown",
        # True if readligo/ had uncommitted changes when the file was made.
        "readligo_git_dirty": bool(status) if status is not None else True,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "h5py_version": h5py.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


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
    bound: int = 10,
    data_dirname: str = DATA_DIRNAME,
    process_dirname: str = OUT_DIRNAME,
    *,
    overwrite: bool = False,
) -> None:
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
        strain, time, channel_dict = loaddata(data_dirname + files.name)
        assert isinstance(time, np.ndarray), "expected time vector"
        gpsStart = time[0]
        attrs = {
            **source_attrs(data_dirname + files.name),
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
            loss_stats=True,
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
