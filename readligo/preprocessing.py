from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

from .io import loaddata


logger = logging.getLogger(__name__)

__all__ = [
    "apply_dq_mask",
    "clean_narrowband",
    "data_loss_stats",
    "heterodyne_downsample",
    "plot_cleaning",
    "process_band",
]


# Data quality masking
def apply_dq_mask(
    strain: np.ndarray,
    channel_dict: dict[str, np.ndarray],
    fs: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out samples flagged as bad by the DQ mask.

    Parameters
    ----------
    strain
        Raw strain at *fs* Hz.
    channel_dict
        Channel dictionary from ``read_ligo.loaddata()``.
    fs
        Sampling frequency of *strain*.

    Returns
    -------
    clean_strain
        Copy of *strain* with bad samples set to zero.
    dq_mask_1hz
        Boolean array at 1 Hz (``True`` = good data).
    """
    dq_mask_1hz = (channel_dict["DATA"] == 1) & (channel_dict["CBC_CAT1"] == 1)
    dq_mask_fs = np.repeat(dq_mask_1hz, fs)
    clean_strain = np.copy(strain)
    clean_strain[~dq_mask_fs] = 0.0
    return clean_strain, dq_mask_1hz


# Heterodyne + multi-stage decimation
def heterodyne_downsample(
    data: np.ndarray,
    fs: float,
    f_band: tuple[float, float],
) -> tuple[np.ndarray, float]:
    """Shift a frequency band to baseband and decimate.

    Multiplies by ``exp(-2 pi i f_h t)`` (where *f_h* is the band centre)
    to shift the positive-frequency band to baseband, then applies
    multi-stage decimation until the sample rate equals the bandwidth.

    Parameters
    ----------
    data
        Real time series at *fs* Hz.
    fs
        Original sampling frequency (Hz).
    f_band
        ``(f_min, f_max)`` defining the frequency band (Hz).

    Returns
    -------
    narrowband
        Complex baseband time series at *fs_new* Hz.
    fs_new
        Actual output sample rate (Hz).
    """
    f_min, f_max = f_band
    f_h = (f_min + f_max) / 2.0
    bandwidth = f_max - f_min

    # Heterodyne: shift band centre to DC
    t = np.arange(len(data)) / fs
    x_h = data * np.exp(-2j * np.pi * f_h * t)

    # Multi-stage decimate (each stage <= 13 per scipy recommendation).
    # Use FIR filter (ftype="fir") because it is guaranteed to handle
    # complex-valued input correctly.
    x_dec = x_h
    current_fs = float(fs)
    while current_fs / bandwidth > 13:
        x_dec = signal.decimate(x_dec, 8, ftype="fir")
        current_fs /= 8

    remaining = int(np.round(current_fs / bandwidth))
    if remaining > 1:
        x_dec = signal.decimate(x_dec, remaining, ftype="fir")
        current_fs /= remaining

    return x_dec, current_fs


# Narrowband cleaning (STFT-based)
def _good_stft_cols(
    dq_mask_1hz: np.ndarray,
    time_bins: np.ndarray,
    win_dur: float,
) -> np.ndarray:
    """Return a boolean mask of STFT columns fully covered by good DQ data.

    Parameters
    ----------
    dq_mask_1hz
        Boolean DQ mask at 1 Hz.
    time_bins
        Centre time (seconds from file start) of each STFT column,
        as returned by ``scipy.signal.stft``.
    win_dur
        Duration of one STFT window in seconds (``nperseg / fs_new``).
    """
    half = win_dur / 2.0
    n_dq = len(dq_mask_1hz)
    good = np.ones(len(time_bins), dtype=bool)
    for i, tc in enumerate(time_bins):
        dq_lo = max(0, int(np.floor(tc - half)))
        dq_hi = min(n_dq, int(np.ceil(tc + half)))
        if dq_lo >= n_dq or dq_hi <= 0 or not np.all(dq_mask_1hz[dq_lo:dq_hi]):
            good[i] = False
    return good


def clean_narrowband(
    data: np.ndarray,
    dq_mask_1hz: np.ndarray,
    fs_new: float,
    *,
    nperseg: int = 64,
    fap: float = 0.001,
    bound: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove transient artifacts from narrowband complex data.

    1. Compute a two-sided STFT (Hann window, 50 % overlap).
    2. Normalise the power in each frequency row by the 39.4th percentile
       of DQ-good columns (maps to chi-squared(2) under Gaussian noise).
    3. Flag time columns where more than *bound* frequency bins exceed the
       chi-squared threshold at false-alarm probability *fap*.
    4. Zero flagged (and DQ-bad) columns, then ISTFT back.

    Parameters
    ----------
    data
        Complex baseband series from :func:`heterodyne_downsample`.
    dq_mask_1hz
        Boolean DQ mask at 1 Hz from :func:`apply_dq_mask`.
    fs_new
        Sample rate of *data* (Hz).
    nperseg
        STFT window length in samples.
    fap
        Per-pixel false-alarm probability for the chi-squared test.
    bound
        Minimum number of flagged frequency bins to declare a column bad.

    Returns
    -------
    cleaned_data
        Cleaned complex time series (same length as *data*).
    sample_mask
        Per-sample boolean mask (``True`` = good, ``False`` = zeroed).
    diag
        Dictionary of STFT intermediates for diagnostics / plotting.
    """
    hop = nperseg // 2

    # Two-sided STFT (input is complex after heterodyning)
    freq, time_bins, zxx = signal.stft(
        data,
        fs=fs_new,
        nperseg=nperseg,
        noverlap=hop,
        window="hann",
        return_onesided=False,
    )

    # Identify STFT columns that fall entirely within DQ-good data
    win_dur = nperseg / fs_new
    good_cols = _good_stft_cols(dq_mask_1hz, time_bins, win_dur)

    # Power spectrum
    sxx = np.abs(zxx) ** 2

    # Normalise using only DQ-good columns so zeroed regions don't bias
    # the noise estimate.
    n_good = int(np.sum(good_cols))
    n = len(data)

    # Chi-squared threshold (|Z|^2 / noise ~ chi2(2) under H0)
    threshold = stats.chi2.isf(fap, df=2)

    if n_good < 2:
        # Too few DQ-good columns to estimate the noise floor, so nothing can
        # be tested: mask the whole series rather than pass it through as good.
        # A lone DQ-good column is reported as artifact loss by data_loss_stats.
        logger.warning(
            "Only %d DQ-good STFT column(s) — cannot estimate noise; masking all samples",
            n_good,
        )
        diag = {
            "freq": freq,
            "time_bins": time_bins,
            "sxx": sxx,
            "sxx_scaled": np.full(sxx.shape, np.nan),
            "good_cols": good_cols,
            "bad_col_mask": np.ones(len(good_cols), dtype=bool),
            "col_hits": np.zeros(len(good_cols), dtype=int),
            "threshold": threshold,
            "fs_new": fs_new,
        }
        return np.zeros_like(data), np.zeros(n, dtype=bool), diag

    q = np.quantile(sxx[:, good_cols], 0.394, axis=1, keepdims=True)
    q = np.where(q > 0, q, 1.0)
    sxx_scaled = sxx / q

    # Column-sum test: flag columns with broadband excess power
    hits = sxx_scaled > threshold
    col_hits = np.sum(hits, axis=0)
    bad_col_mask = (col_hits > bound) | ~good_cols

    n_artifact = int(np.sum((col_hits > bound) & good_cols))
    logger.info(
        "Cleaning: %d/%d good cols, %d artifact cols removed (%.1f%% extra loss)",
        n_good,
        len(good_cols),
        n_artifact,
        100.0 * n_artifact / max(n_good, 1),
    )

    # Zero bad columns in the complex STFT and reconstruct
    zxx_clean = zxx.copy()
    zxx_clean[:, bad_col_mask] = 0

    _, cleaned = signal.istft(
        zxx_clean,
        fs=fs_new,
        nperseg=nperseg,
        noverlap=hop,
        window="hann",
        input_onesided=False,
    )

    # Match output length to input
    if len(cleaned) > n:
        cleaned = cleaned[:n]
    elif len(cleaned) < n:
        cleaned = np.pad(cleaned, (0, n - len(cleaned)))

    # Map STFT column mask to a per-sample boolean mask (True = good).
    # Use time_bins (window centres from scipy) to correctly account for
    # boundary='zeros' padding, which shifts window positions by nperseg//2.
    sample_mask = np.ones(n, dtype=bool)
    for i, is_bad in enumerate(bad_col_mask):
        if is_bad:
            center = int(round(time_bins[i] * fs_new))
            lo = max(0, center - nperseg // 2)
            hi = min(n, center + nperseg // 2)
            sample_mask[lo:hi] = False

    diag = {
        "freq": freq,
        "time_bins": time_bins,
        "sxx": sxx,
        "sxx_scaled": sxx_scaled,
        "good_cols": good_cols,
        "bad_col_mask": bad_col_mask,
        "col_hits": col_hits,
        "threshold": threshold,
        "fs_new": fs_new,
    }

    return cleaned, sample_mask, diag


def data_loss_stats(
    sample_mask: np.ndarray,
    dq_mask_1hz: np.ndarray,
    diag: dict,
    *,
    bound: int = 10,
) -> dict[str, int | float]:
    """Classify masked samples into DQ-bad, STFT-boundary, and artifact loss.

    Parameters
    ----------
    sample_mask
        Per-sample boolean mask from :func:`clean_narrowband`.
    dq_mask_1hz
        Boolean DQ mask at 1 Hz from :func:`apply_dq_mask`.
    diag
        Diagnostics dict from :func:`clean_narrowband`.
    bound
        Column-hit threshold (must match the value used in cleaning).

    Returns
    -------
    dict with keys:

    - ``n_samples``: total number of samples.
    - ``n_dq_loss``: samples lost because their DQ second is bad.
    - ``n_boundary_loss``: DQ-good samples lost because their STFT window
      straddles a DQ-bad region.
    - ``n_artifact_loss``: DQ-good samples lost by the chi-squared
      transient test.
    - ``n_usable``: samples with ``sample_mask == True``.
    - ``duty_factor``: fraction of usable samples.
    """
    good_cols = diag["good_cols"]
    bad_col_mask = diag["bad_col_mask"]
    col_hits = diag["col_hits"]
    time_bins = diag["time_bins"]
    fs_new = diag["fs_new"]
    nperseg = len(diag["freq"])

    n = len(sample_mask)

    # Per-sample DQ mask at the downsampled rate
    dq_sample = np.ones(n, dtype=bool)
    sample_seconds = (np.arange(n) / fs_new).astype(int)
    in_range = sample_seconds < len(dq_mask_1hz)
    dq_sample[in_range] = dq_mask_1hz[sample_seconds[in_range]]
    dq_sample[~in_range] = False

    # Map each bad-column type to per-sample coverage
    covered_by_dq_col = np.zeros(n, dtype=bool)
    half = nperseg // 2
    for i in range(len(bad_col_mask)):
        if not bad_col_mask[i]:
            continue
        center = int(round(time_bins[i] * fs_new))
        lo = max(0, center - half)
        hi = min(n, center + half)
        if not good_cols[i]:
            covered_by_dq_col[lo:hi] = True

    # Classify each masked sample
    masked = ~sample_mask
    dq_bad_samples = ~dq_sample
    dq_loss = masked & dq_bad_samples
    boundary_loss = masked & dq_sample & covered_by_dq_col
    artifact_loss = masked & dq_sample & ~covered_by_dq_col

    n_dq = int(np.sum(dq_loss))
    n_boundary = int(np.sum(boundary_loss))
    n_artifact = int(np.sum(artifact_loss))
    n_usable = int(np.sum(sample_mask))

    if n_dq + n_boundary + n_artifact != int(np.sum(masked)):
        logger.warning(
            "Loss categories (%d + %d + %d = %d) do not sum to total masked (%d)",
            n_dq,
            n_boundary,
            n_artifact,
            n_dq + n_boundary + n_artifact,
            int(np.sum(masked)),
        )
    
    # DQ duty factor
    dq_mask_fs = np.repeat(dq_mask_1hz, 4096)
    dq_duty_factor = 1-(np.sum(~dq_mask_fs)/len(dq_mask_fs))

    return {
        "n_samples": n,
        "n_dq_loss": n_dq,
        "n_boundary_loss": n_boundary,
        "n_artifact_loss": n_artifact,
        "n_usable": n_usable,
        "duty_factor": n_usable / n if n > 0 else 0.0,
        "dq_duty_factor": dq_duty_factor
    }


def process_band(
    raw_strain: np.ndarray,
    channel_dict: dict[str, np.ndarray],
    gps_start: float,
    f_band: tuple[float, float],
    *,
    fs: int = 4096,
    nperseg: int = 64,
    fap: float = 0.001,
    bound: int = 10,
    plot: bool = False,
    loss_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, int | float] | None]:
    """Full single-file, single-band preprocessing pipeline.

    Chains :func:`heterodyne_downsample` and :func:`clean_narrowband`.

    Parameters
    ----------
    raw_strain
        Raw strain.
    channel_dict
        Channel dictionary from ``read_ligo.loaddata()``.
    gps_start
        GPS time of the first sample in *raw_strain*.
    f_band
        ``(f_min, f_max)`` frequency band in Hz.
    fs
        Original sampling frequency (Hz).
    nperseg, fap, bound
        Forwarded to :func:`clean_narrowband`.
    plot
        If ``True``, show a 4-panel diagnostic figure.

    Returns
    -------
    gps_times
        GPS timestamp per output sample.
    cleaned_data
        Cleaned complex baseband time series.
    sample_mask
        Per-sample boolean mask (``True`` = good, ``False`` = zeroed).
    fs_new
        Output sample rate (Hz).
    """
    clean_strain, dq_mask_1hz = apply_dq_mask(raw_strain, channel_dict, fs)
    narrowband, fs_new = heterodyne_downsample(clean_strain, fs, f_band)
    cleaned, sample_mask, diag = clean_narrowband(
        narrowband,
        dq_mask_1hz,
        fs_new,
        nperseg=nperseg,
        fap=fap,
        bound=bound,
    )
    gps_times = gps_start + np.arange(len(cleaned)) / fs_new

    if plot:
        plot_cleaning(narrowband, cleaned, diag, bound=bound)
    
    if loss_stats:
        breakdown = data_loss_stats(sample_mask, dq_mask_1hz, diag, bound=bound)
        return gps_times, cleaned, sample_mask, fs_new, breakdown

    return gps_times, cleaned, sample_mask, fs_new, None


def plot_cleaning(
    data: np.ndarray,
    cleaned: np.ndarray,
    diag: dict,
    *,
    bound: int = 10,
) -> None:
    """Visualize the narrowband cleaning procedure.

    Produces a 3-panel figure showing:
    1. Normalised STFT power spectrogram (dB).
    2. Number of threshold-exceeding bins per STFT column.
    3. Time-domain amplitude before and after cleaning.

    Parameters
    ----------
    data
        Complex baseband series before cleaning (from :func:`heterodyne_downsample`).
    cleaned
        Cleaned complex time series (from :func:`clean_narrowband`).
    diag
        Diagnostics dict returned by :func:`clean_narrowband`.
    bound
        Threshold used for the column-hit test (for the dashed line).
    """
    freq = np.fft.fftshift(diag["freq"])
    time_bins = diag["time_bins"]
    sxx_scaled = np.fft.fftshift(diag["sxx_scaled"], axes=0)
    good_cols = diag["good_cols"]
    bad_col_mask = diag["bad_col_mask"]
    col_hits = diag["col_hits"]
    threshold = diag["threshold"]

    _, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Compute color limits from good columns only
    sxx_scaled_db = 10 * np.log10(sxx_scaled + 1e-30)
    good_db = sxx_scaled_db[:, good_cols]
    if good_db.size and np.all(np.isfinite(good_db)):
        vmin, vmax = np.percentile(good_db, [2, 98])
    else:  # no good columns, or cleaning was skipped (sxx_scaled is NaN)
        vmin = vmax = None

    # 1) Normalised power spectrogram
    axes[0].pcolormesh(
        time_bins, freq, sxx_scaled_db, shading="auto", vmin=vmin, vmax=vmax
    )
    axes[0].set_ylabel("Freq (Hz)")
    axes[0].set_title(f"Normalised power (dB) — chi2 threshold = {threshold:.1f}")

    # 2) Hits per column
    dt = np.median(np.diff(time_bins)) if len(time_bins) > 1 else 1.0
    colors = ["red" if b else "steelblue" for b in bad_col_mask]
    axes[1].bar(time_bins, col_hits, width=dt, color=colors, alpha=0.7)
    axes[1].axhline(bound, color="k", ls="--", label=f"bound = {bound}")
    for i, tc in enumerate(time_bins):
        if not good_cols[i]:
            axes[1].axvspan(tc - dt / 2, tc + dt / 2, color="gray", alpha=0.2)
    axes[1].set_ylabel("# bins > threshold")
    axes[1].set_title("Hits per STFT column (red = flagged, gray = DQ-bad)")
    axes[1].legend()

    # 3) Time-domain before vs after
    t = np.arange(len(data)) / diag["fs_new"]
    axes[2].plot(t, np.abs(data), alpha=0.5, label="before", lw=0.5)
    axes[2].plot(t, np.abs(cleaned), alpha=0.7, label="after", lw=0.5)
    axes[2].set_ylabel("|amplitude|")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Time domain: before vs after cleaning")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
