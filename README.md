# readligo

Reads GWOSC strain files (HDF5) and cleans them into the narrowband series that the
rustyprune and pyloki CW searches read. Detectors: H1 and L1.

Each product is one detector in one band `[fmin, fmin + B]`:
- **Signal:** a real series sampled at exactly 2B, heterodyned at fmin against one global epoch.
  This is the convention of the Makefakedata_v5 data the searches are built on.
- **Weights:** a per-sample weight and mask.
- **Noise:** the noise per 1800 s segment and a line list.
- **Record:** the gate list and every setting.

## Install

```bash
pip install -e .                 # core: numpy, scipy, h5py
pip install -e ".[dev]"          # + pytest
pip install -e ".[validation]"   # + LALSuite, for scripts/validation
```

## Clean data

For an all-sky search, plan Doppler-safe overlapping bands first (each band records the source
frequencies whose whole track stays inside it):

```bash
python -m readligo plan  --fmin 20 --fmax 2000 --tspan-days 246.1 -o bands.txt
python -m readligo check --data-dir /path/to/files --ifo H1 --workers 16
python -m readligo clean --data-dir /path/to/files --out-dir products/H1 --ifo H1 \
       --bands-file bands.txt --workers 16 --skip-existing
```

or give bands directly with `--band 100.68 0.1 --band 145.4 0.1`.

- Band edges must be multiples of 1/1800 Hz.
- Any width works (0.1, 0.15, 0.37, ... Hz), and the output rate is exactly 2B.
- rustyprune and pyloki currently accept only B = k/10 Hz.

```python
from readligo import BandSpec, PipelineConfig, run_pipeline, read_product
paths = run_pipeline(PipelineConfig(data_dir="...", out_dir="products/H1", ifo="H1",
                                    bands=[BandSpec(100.68, 0.1)]))
p = read_product(paths[0])
p.data, p.window, p.mask, p.times, p.sqrt_sx
```

## Feed rustyprune

```bash
python -m readligo export products/H1/H1_100.68-100.78Hz_*.h5 --out-dir ROOT_frames \
       --tstart 1238319104 --duration 345600 --tseg 86400
```

This writes:
- `data.npy`, which replaces `ROOT/frames/data.npy` after `rustyprune prepare`;
- a matching `band.json`;
- `mask.npy`, `window.npy` and `readligo_meta.json`, for later search changes.

## Documentation

- [`docs/cleaning_pipeline.md`](docs/cleaning_pipeline.md): how the cleaning works, how to run it (laptop and marvin), its validation (O3a and full-scale O4a, hardware injections), and what the search must still do.
- [`docs/cleaning_report.md`](docs/cleaning_report.md): the v1 → v2 redesign and its first validation.
- [`docs/README.md`](docs/README.md): documentation hub.
- [`ISSUES.md`](ISSUES.md): issue tracker.

## Tests

```bash
pytest          # 68 tests, about 75 s
```
