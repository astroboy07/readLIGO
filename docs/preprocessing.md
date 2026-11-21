# v1 preprocessing chain (superseded)

> Source: [`readligo/preprocessing.py`](../readligo/preprocessing.py) |
> Back to [docs README](README.md)

> **This is the v1 per-file chain, kept for comparison.** Its problems are listed in
> `ISSUES.md` (C1–C8). The pipeline that replaces it is described in
> [`cleaning_report.md`](cleaning_report.md) (`readligo.run_pipeline`,
> `python -m readligo clean`).

## Overview

`readligo.preprocessing` takes raw LIGO strain data and produces a cleaned,
narrowband complex time series suitable for continuous-wave (CW) searches.
The entry point is `process_band()`, which returns four arrays and, with
`loss_stats=True` (the default), a fifth: a dict that breaks down the masked samples.

```python
gps_times, cleaned, sample_mask, fs_new, breakdown = process_band(
    strain, channel_dict, gps_start, f_band=(100, 101)
)
```

## Pipeline stages

### 1. Data-quality masking (`apply_dq_mask`)

Samples flagged as bad by the LIGO DQ channels (`DATA` and `CBC_CAT1`) are
set to zero. A 1 Hz boolean mask (`dq_mask_1hz`) is produced for later use.

### 2. Heterodyne and decimation (`heterodyne_downsample`)

The target frequency band is shifted to baseband by multiplying the strain by
`exp(-2 pi i f_h t)`, where `f_h` is the band centre. This uses the standard
negative-sign convention: a signal at original frequency `f_0` appears at
`f_0 - f_h` in the baseband, preserving both frequency ordering and phase.

The resulting complex signal is then decimated in multiple stages (each factor
<= 13, using FIR filters for complex-data safety) until the sample rate equals
the bandwidth. For a 1 Hz band, the output is at 1 sample/second.

### 3. STFT-based transient cleaning (`clean_narrowband`)

Transient artifacts (glitches) that pass the DQ flags are removed using a
short-time Fourier transform approach:

1. A two-sided STFT is computed (Hann window, 50% overlap,
   `nperseg=64` samples by default).
2. Each frequency row is normalised by the 39.4th percentile of the power
   across DQ-good columns. For Gaussian noise the power is exponential, and
   its 39.4th percentile is half its mean (1 − e^{−1/2} = 0.394), not the
   mean. Power divided by half its mean is chi-squared with 2 degrees of
   freedom, so the normalised power follows that distribution.
3. A per-pixel chi-squared test is applied at false-alarm probability
   `fap=0.001` (threshold ~13.8). STFT columns where more than `bound=15`
   frequency bins exceed the threshold are flagged as containing broadband
   transients.
4. All flagged columns (artifact + DQ-bad) are zeroed in the STFT domain.
5. The inverse STFT reconstructs the cleaned time series.

## Outputs

| Array | Type | Length | Description |
|---|---|---|---|
| `gps_times` | float64 | N | GPS timestamp per output sample |
| `cleaned` | complex128 | N | Cleaned complex baseband time series |
| `sample_mask` | bool | N | Per-sample validity mask (`True` = good) |
| `fs_new` | float | scalar | Output sample rate (Hz) |

## The sample mask

`sample_mask` is a per-sample boolean array aligned with `cleaned` and
`gps_times`. It marks which samples are trustworthy for downstream analysis.

A sample is marked `False` (bad) if it falls within the time span of any
STFT column that was zeroed during cleaning. This includes:

- **DQ-bad columns**: STFT windows overlapping data-quality-flagged segments.
- **Artifact columns**: STFT windows where the chi-squared broadband test
  detected a transient glitch.

### Why `sample_mask` matters

After zeroing bad STFT columns and reconstructing via ISTFT, some samples
marked bad by `sample_mask` may be **non-zero**. This happens because the
STFT uses 50% overlapping Hann windows: at the boundary between a bad column
and a good column, the overlap-add reconstruction mixes a zero contribution
(from the bad column) with a non-zero contribution (from the good column).
The result is non-zero but **corrupted** — it has only a partial signal
contribution and incorrect amplitude.

### How to use the mask

**Option A** — Zero the corrupted samples before further analysis:
```python
cleaned[~sample_mask] = 0.0
```

**Option B** — Exclude masked samples from computations without modifying
the data (preferred when the downstream method supports gating or weighting):
```python
good_data = cleaned[sample_mask]
good_times = gps_times[sample_mask]
```

### Effective duty factor

The fraction of usable data after cleaning:
```python
duty_factor = np.sum(sample_mask) / len(sample_mask)
```

## Default parameters

| Parameter | Default | Role |
|---|---|---|
| `fs` | 4096 | Raw strain sample rate (Hz) |
| `nperseg` | 64 | STFT window length (samples at `fs_new`) |
| `fap` | 0.001 | Per-pixel false-alarm probability |
| `bound` | 15 | Min flagged frequency bins to declare a column bad (`DEFAULT_BOUND`; the stored v1 products used 15) |

## Diagnostics

`clean_narrowband` returns a `diag` dictionary containing STFT intermediates
(`freq`, `time_bins`, `sxx`, `sxx_scaled`, `good_cols`, `bad_col_mask`,
`col_hits`, `threshold`, `fs_new`). Pass this to `plot_cleaning()` for a
3-panel diagnostic figure, or set `plot=True` in `process_band()`:

```python
gps_times, cleaned, sample_mask, fs_new, breakdown = process_band(
    strain, channel_dict, gps_start, (100, 101), plot=True
)
```
