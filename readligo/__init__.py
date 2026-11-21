"""readligo: GWOSC strain I/O and the cleaning pipeline that feeds the CW searches.

v2 pipeline (``docs/cleaning_report.md``)::

    from readligo import BandSpec, PipelineConfig, run_pipeline, read_product
    paths = run_pipeline(PipelineConfig(data_dir=..., out_dir=..., ifo="H1",
                                        bands=[BandSpec(100.68, 0.1)]))
    product = read_product(paths[0])

The v1 per-file chain (``preprocessing``, ``process_run``) is kept unchanged
for comparison; its names are still importable from here but are loaded only
when used.
"""
from __future__ import annotations

import importlib
from typing import Any

from readligo.channelize import FFTChannelizer
from readligo.conditioning import ConditionedStrain, Gate, GateParams, RunData, condition_run, detect_gates_run
from readligo.config import (
    DEFAULT_EPOCH,
    BandSpec,
    ChannelizerParams,
    NoiseParams,
    PipelineConfig,
    VetoParams,
    band_around,
    to_grid,
)
from readligo.inventory import DETECTORS, RawFile, contiguous_runs, scan_directory
from readligo.io import GwoscFile, loaddata, read_gwosc, read_hdf5, read_meta
from readligo.noise import NoiseEstimate, estimate_noise, gwosc_antialias_gain, narrowband_veto
from readligo.pipeline import run_pipeline
from readligo.products import BandProduct, export_rustyprune, product_path, read_product, write_product
from readligo.provenance import provenance_attrs

# v1 names, imported on first use (process_run needs `rich`, preprocessing `matplotlib`).
_LEGACY = {
    "apply_dq_mask": "readligo.preprocessing",
    "clean_narrowband": "readligo.preprocessing",
    "data_loss_stats": "readligo.preprocessing",
    "heterodyne_downsample": "readligo.preprocessing",
    "plot_cleaning": "readligo.preprocessing",
    "process_band": "readligo.preprocessing",
    "processing_attrs": "readligo.preprocessing",
    "process_ligo_files": "readligo.process_run",
    "save_hdf5_file": "readligo.process_run",
    "source_attrs": "readligo.process_run",
}


def __getattr__(name: str) -> Any:
    if name in _LEGACY:
        return getattr(importlib.import_module(_LEGACY[name]), name)
    raise AttributeError(f"module 'readligo' has no attribute {name!r}")


__all__ = [
    # v2
    "DEFAULT_EPOCH",
    "DETECTORS",
    "BandProduct",
    "BandSpec",
    "ChannelizerParams",
    "ConditionedStrain",
    "FFTChannelizer",
    "Gate",
    "GateParams",
    "GwoscFile",
    "NoiseEstimate",
    "NoiseParams",
    "PipelineConfig",
    "RawFile",
    "VetoParams",
    "RunData",
    "band_around",
    "condition_run",
    "contiguous_runs",
    "detect_gates_run",
    "estimate_noise",
    "export_rustyprune",
    "gwosc_antialias_gain",
    "narrowband_veto",
    "product_path",
    "provenance_attrs",
    "read_gwosc",
    "read_product",
    "run_pipeline",
    "scan_directory",
    "to_grid",
    "write_product",
    # I/O shared with v1
    "loaddata",
    "read_hdf5",
    "read_meta",
    # v1 (lazy)
    *_LEGACY,
]
