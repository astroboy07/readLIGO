# Pipeline architecture (v2)

> Back to [docs README](README.md). How each step works and why, and the measurements, are in
> [`cleaning_pipeline.md`](cleaning_pipeline.md).

## Stages and data flow

```
plan_bands(f_lo, f_hi, tspan, ...) ──▶ [BandSpec(fmin, B, usable)]             planning.py
scan_directory(data_dir, ifo) ──▶ [RawFile] ──contiguous_runs──▶ [[RawFile]]   inventory.py
          │
          ▼  per run
RunData(files, dq_flags)          1 Hz good mask (DQ and injection bits)         conditioning.py
check_strain(file)                per file, parallel: NaN/inf or constant DQ-good seconds
          │
          ▼  per run, parallel (pipeline._task_condition)
condition_run(run, GateParams, path)  ──▶ ConditionedRun (.npy in a private scratch folder)
   high-pass per DQ stretch (chunked, zero phase) × 8 s edge tapers; zero elsewhere
detect_gates_run(cond, GateParams, epoch, context=64) ──▶ [Gate]
   per file block, whitened by min(file PSD, run PSD); detections merged over blocks
          │
mean_psd (gate-free segments) ──▶ inband_shape per band ──▶ FFTChannelizer.set_shape    noise.py
          │
          ▼  all runs
ConditionedStrain(runs, gates)    reader(gps, n): conditioned strain × gate window
          │
          ▼  per chunk of bands (≤ chunk_gb), blocks in parallel (pipeline._task_channelize)
FFTChannelizer.band_from_spectrum ──▶ memory-mapped (n_bands_B, n) array per bandwidth   channelize.py
          │
          ▼  per band, parallel (pipeline._task_finalize)
output_window (once per bandwidth) ──▶ window, mask;  x[window == 0] = 0
estimate_noise ──▶ Sx per 1800 s, lines, sqrt_sx                                noise.py
narrowband_veto, excess_power_veto ──▶ zero outliers; estimate_noise again
          │
BandProduct ──write_product──▶ H1_<fmin>-<fmax>Hz_<gps>-<dur>.h5               products.py
          │
export_rustyprune([products], out, tstart, duration, tseg) ──▶ data.npy, band.json, ...
```

## Design rules

- **Each run is one series.** Back-to-back files are never cut. Every filter sees continuous data, and file boundaries leave no trace (C6).
- **The output depends only on GPS time and the input range.**
  - The block grid and phase are anchored to the global epoch.
  - The conditioned strain and gate window are functions of GPS time alone.
  - So splitting the bands, blocks or workers gives bit-identical output (tested).
  - Splitting the GPS range is different: the cut becomes a tapered data edge, and the halves differ next to it.
  - Splitting one run into separate files gives the same output only without gates; with gates, gate edges can move by a few ms (different PSD blocks).
- **Exact arithmetic for the grid.**
  - Band edges are `Fraction`s on the 1/1800 Hz grid.
  - The heterodyne phase per block is `(fmin · (s_b − epoch)) mod 1`, computed exactly; `fmin · T` must be an integer (checked).
  - The rate is 2B by construction.
- **One broadband step, then cheap per-band steps.**
  - Conditioning and gating run once per detector.
  - One FFT per 7200 s block serves every band of a chunk.
  - The per-band work is a few thousand bins per block.
- **Memory and disk.** Bounded by `chunk_gb` and `workers`, not by the number of bands or days. The conditioned strain stays in a private scratch folder (`work_dir/readligo-<ifo>-XXXX`), deleted afterwards; `work_dtype="float32"` halves it.
- **Fail early.** Settings, the band grid, Nyquist, existing products and bad strain are checked before the conditioning starts.
- **Store, don't decide for the search.** Gaps are zero, as the searches expect. The weights, mask, per-segment noise, gates, vetoes, lines and usable source range are stored, so a later mask-aware or noise-weighted search can use them.

## Product file (HDF5, `FORMAT_VERSION = "2.0"`)

| Item | Content |
|---|---|
| attrs | `ifo, fmin, band, fmax, fs (= 2B), f_het (= fmin), epoch, n0, gps_start, n_samples, sqrt_sx, variance, duty_factor`, convention strings, `fmin_exact`/`band_exact`, `guard_hz`, `flat_band`, `usable_source_band`, `usable_doppler_only`, `inband_shape`, `inband_gain_fmin/fmax`, `sqrt_sx_harmonic`, `sqrt_sx_median`, `variance_ratio`, `n_gates`, `n_vetoes`, `n_excess_vetoes`, `n_lines`, `near_gate_mean_z`, `near_edge_mean_z`, `far_mean_z`, `live_seconds`, dropped seconds, `cw_hwinj_on_fraction`, GWOSC anti-alias gain, `config` (JSON), channelizer and `gating_*` settings, git commit, versions |
| `data` | float64, sample k at `epoch + (n0 + k)/fs` |
| `window` | float32, RMS of the strain window over each sample's slot |
| `mask` | uint8, 1 where the slot is fully good |
| `noise_segments/` | `gps_start, duration, duty, live, sx` (one-sided PSD per 1800 s) |
| `spectrum/` | `freq, ratio, flat, excluded`: mean normalised power per 1/1800 Hz bin |
| `lines/` | `freq, ratio` |
| `gates/` | `gps_start, gps_end, gps_peak, peak_sigma` |
| `vetoes/` | `gps_start, gps_end, peak_z` (narrowband outlier veto) |
| `excess_vetoes/` | `gps_start, gps_end, mean_z` (excess-power veto) |
| `stretches/`, `dropped_stretches/`, `runs/`, `files/`, `gate_blocks/` | where the data came from |

## Extending

- **L1:** run the same command with `--ifo L1` and the same epoch, then pass both products to `export_rustyprune`, which writes a (2, N) array.
- **New bands:** rerun with more `--band` options or a `plan` bands file. Conditioning is done once per run, so ask for all bands in one run; `--skip-existing` resumes.
- **Other settings:** `GateParams`, `ChannelizerParams`, `NoiseParams` and `VetoParams` in `PipelineConfig`, or a JSON file via `--config`. Every product records the values used.
