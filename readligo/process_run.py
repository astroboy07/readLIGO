import h5py
import numpy as np
from pathlib import Path
from rich.progress import track
from .io import loaddata
from .preprocessing import process_band


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
    overwrite: bool = False,
) -> None:
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
    for files in track(hdf5_files_sorted, description="Processing files"):
        strain, time, channel_dict = loaddata(data_dirname + files.name)
        assert isinstance(time, np.ndarray), "expected time vector"
        gpsStart = time[0]
        gps_times, cleaned, sample_mask, fs_new, breakdown = process_band(
            strain,
            channel_dict,
            gpsStart,
            f_band,
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
            overwrite=overwrite,
        )
