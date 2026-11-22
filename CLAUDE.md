# CLAUDE.md

Read `docs/README.md` first for project orientation, then
`docs/project_context.md` (local only, not in git) for current status and conventions.
Only go deeper into the specific docs below when the task requires it.

## Documentation map

| File | Read when... |
|---|---|
| `docs/README.md` | Always — start here |
| `docs/project_context.md` | Always — current state, layout, conventions (local only, not in git) |
| `docs/cleaning_pipeline.md` | Working on the cleaning: how every step works, running it, validation, limits |
| `docs/cleaning_report.md` | History of the v1 → v2 redesign and its first validation |
| `docs/pipeline_architecture.md` | Changing how the stages connect, or the product format |
| `docs/preprocessing.md` | Only for the v1 chain (`process_band`), kept for comparison |
| `ISSUES.md` | Status of every known issue |

## Source code

| File | Purpose |
|---|---|
| `readligo/pipeline.py` | `run_pipeline(PipelineConfig)` — the v2 entry point (parallel, chunked); `check_data` |
| `readligo/planning.py` | `plan_bands`, `usable_range` — Doppler-safe overlapping bands for all-sky searches |
| `readligo/config.py` | `BandSpec` (exact 1/1800 Hz grid), `band_around`, `PipelineConfig` and its params |
| `readligo/conditioning.py` | DQ stretches, zero-phase high-pass, tapers, gating, `ConditionedStrain` |
| `readligo/channelize.py` | `FFTChannelizer` — real output at exactly 2B, heterodyned at fmin |
| `readligo/noise.py` | Sx per segment, lines, narrowband and excess-power vetoes, in-band PSD shape |
| `readligo/products.py` | `BandProduct`, HDF5 read/write, `export_rustyprune` |
| `readligo/io.py`, `inventory.py` | GWOSC reader (`read_gwosc`), file scan and contiguous runs |
| `readligo/cli.py` | `python -m readligo plan | check | clean | export` |
| `readligo/preprocessing.py`, `process_run.py` | v1 chain, superseded |

## Key conventions

- Python 3.11, fully type-annotated; numpy arrays throughout
- Bands are `BandSpec(fmin, band)` on the 1/1800 Hz grid; output rate is exactly 2B
- Real heterodyned output: `x = 2 Re[z exp(-2 pi i fmin (t - epoch))]`, one global epoch (default 1238319104)
- Detectors are H1 and L1 only
- Never modify the raw data files; never edit the rustyprune or pyloki repositories from here
- Run with this checkout importable (`pip install -e .` or `PYTHONPATH=$PWD`); a stale installed copy exists in site-packages
- `pytest` must pass (`tests/`)
