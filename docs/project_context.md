# Project context

Last updated: 2026-11-21

## What this project does

Cleans GWOSC LIGO strain (O3a, 4 kHz HDF5) into narrowband series for the rustyprune and
pyloki continuous-wave searches.

- **Format.** The output matches the Makefakedata_v5 data those searches are built on: a real series at exactly 2B, heterodyned at fmin, on one global epoch.
- **Gaps.** Gaps are zero, with a mask and weights stored alongside.
- **Main validation target.** Recovering the CW hardware injections.

## State

- **v2 pipeline, hardened (2026-11-21).** All attack-review issues A1–A15 fixed (A9 partly): private
  scratch folders, Doppler band planner, parallel chunked processing for 180+ days and thousands of
  bands, longer gate and edge tapers (no line leakage), in-band flattening, excess-power veto, new
  line finder, strict settings and export checks. See `docs/cleaning_pipeline.md`.
- **Validated** on the 56 local O3a H1 files (hardware injections P04, P01, P06, P14, P16, P05 recovered)
  and at full scale on O4a H1 on marvin (3,901 files, 160.6 live days).
- **Not done yet:**
  - L1 data (not downloaded);
  - search-code changes (ISSUES D1–D5: mask input, per-segment variance, real-data `prepare`,
    restricting each band to its usable source range);
  - line subtraction; appending data to an earlier run.
- **v1** (`preprocessing.py`, `process_run.py`) is kept for comparison only.

## Layout

```
readligo/
├── io.py            GWOSC reader: read_gwosc (validated), v1 loaddata/read_hdf5
├── inventory.py     find files, group into contiguous runs
├── config.py        BandSpec (exact 1/1800 Hz grid, usable range), band_around, PipelineConfig, params
├── planning.py      plan_bands, usable_range (Doppler-safe band plan)
├── conditioning.py  DQ stretches, high-pass, tapers, gating, ConditionedStrain, output_window
├── channelize.py    FFTChannelizer (overlap-save, Kaiser kernel, exact 2B)
├── noise.py         Sx per segment, lines, vetoes, in-band PSD shape, GWOSC anti-alias gain
├── products.py      BandProduct, HDF5 write/read, export_rustyprune
├── pipeline.py      run_pipeline (parallel, chunked), check_data
├── cli.py           python -m readligo plan | check | clean | export
├── provenance.py    git commit and versions
├── preprocessing.py v1 chain (superseded)
└── process_run.py   v1 batch driver (superseded)
tests/               pytest suite + synthetic GWOSC writer (fakegwosc.py)
scripts/             band list; validation/ (hardware injections, real-data checks, MFDv5, LAL)
products/            outputs (git-ignored)
```

## Conventions

- **Python and types.** Python 3.11, fully type-annotated, numpy arrays throughout.
- **Bands** are `BandSpec(fmin, band)`, both multiples of 1/1800 Hz and held as exact `Fraction`s. 6-decimal forms snap. Use `band_around(f_lo, f_hi, B)` to centre a band on a frequency track.
- **Heterodyne:** `x(t_n) = 2 Re[ z(t_n) exp(-2πi fmin (t_n - epoch)) ]`, with `t_n = epoch + n/(2B)`, real, amplitude preserving. The default epoch is 1238319104, rustyprune's and pyloki's tstart.
- **One epoch everywhere.** Use the same epoch for every detector and band, so the products align sample for sample.
- **DQ:** `DATA`, `NO_STOCH_HW_INJ`, `NO_DETCHAR_HW_INJ` by default (`dq_flags`). CW hardware injections are never masked.
- **Bands for all-sky searches** come from `plan_bands` (`python -m readligo plan`); each product stores its usable source range.
- **Amplitude:** a CW of amplitude A at f appears with A · g(f) · (GWOSC anti-alias gain), g = in-band flattening gain (1 in most bands above 40 Hz).
- **Errors, never silent fixes.** Off-grid bands, bad settings, non-finite or constant DQ-good strain, inconsistent files and existing outputs (without `overwrite`/`skip_existing`) all raise, before any processing where possible.
- **Parallel scripts on macOS** need `if __name__ == "__main__":` when `workers > 1`.
- **Provenance.** Every product records its full configuration, the git commit and dirty flag, and library versions.
- **Import path.** Run scripts with this checkout importable (`pip install -e .`), not a stale installed copy.
