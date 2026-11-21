"""Band grid rules, product file round trip, rustyprune export."""
from __future__ import annotations

import json
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from readligo.config import BandSpec, PipelineConfig, band_around, to_grid
from readligo.products import BandProduct, export_rustyprune, read_product, write_product

# Keys rustyprune's band.json accepts (AllSkySpec fields + patches, scheme, generator).
RUSTYPRUNE_KEYS = {
    "duration", "tseg", "ifos", "detector", "f_inj", "f1dot", "fmin", "band", "h0", "sqrt_sx", "cosi", "psi",
    "inject_ra", "inject_dec", "seed", "fdot_range", "poly_order", "patch_radius_deg", "phase_tol_cycles",
    "tiling_mode", "branch_max", "interpolation_width", "use_moving_grid", "n_runs", "n_workers",
    "max_sugg", "batch_size", "patches", "scheme", "generator",
}


def test_to_grid() -> None:
    assert to_grid(100.68) == Fraction(181224, 1800)
    assert to_grid("0.37") == Fraction(37, 100)
    assert to_grid(1390.885556) == Fraction(2503594, 1800)  # 6-decimal form snaps
    with pytest.raises(ValueError, match="nearest valid"):
        to_grid(100.6801)


def test_bandspec_exact() -> None:
    b = BandSpec(20.1, 0.1)
    assert b.fmax - b.fmin == Fraction(1, 10)  # no 20.1 - 20.0 float error
    assert b.fs == Fraction(1, 5)
    with pytest.warns(UserWarning, match="k/10"):
        BandSpec(100.6, 0.15)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BandSpec(100.6, 0.5)


def test_band_around() -> None:
    b = band_around(145.448450, 145.451397, 0.1)  # hardware injection P06
    assert float(b.fmin) == 145.4
    with pytest.raises(ValueError):
        band_around(889.896, 890.212, 0.1)  # P17 needs 0.4 Hz


def test_config_validation(tmp_path: Path) -> None:
    ok = dict(data_dir=tmp_path, out_dir=tmp_path, ifo="H1", bands=[(100.68, 0.1)])
    cfg = PipelineConfig(**ok)
    assert PipelineConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).bands == cfg.bands
    with pytest.raises(ValueError):
        PipelineConfig(**{**ok, "ifo": "V1"})
    with pytest.raises(ValueError):
        PipelineConfig(**{**ok, "dq_flags": ("CBC_CAT1",)})
    with pytest.raises(ValueError):
        PipelineConfig(**{**ok, "bands": [(12.0, 0.1)]})  # below the high-pass


def _product(ifo: str = "H1", n0: int = -10, n: int = 200) -> BandProduct:
    rng = np.random.default_rng(1)
    window = np.ones(n, dtype=np.float32)
    window[50:60] = 0
    return BandProduct(
        ifo=ifo, fmin=100.68, band=0.1, epoch=1238319104.0, n0=n0, data=rng.normal(size=n) * window,
        window=window, mask=window == 1, sqrt_sx=1e-23,
        tables={
            "files": {"name": np.array(["a.hdf5", "b.hdf5"]), "gps_start": np.array([1, 2])},
            # two 450 s noise segments from the epoch: Sx = 1e-46, then 9e-46
            "noise_segments": {"gps_start": 1238319104.0 + np.array([0.0, 450.0]), "duration": np.full(2, 450.0),
                               "duty": np.ones(2), "sx": np.array([1e-46, 9e-46])},
        },
        attrs={"config": {"x": 1}, "dq_flags": ["DATA"], "n_gates": 3},
    )


def test_product_round_trip(tmp_path: Path) -> None:
    p = _product()
    path = write_product(p, tmp_path / "p.h5")
    q = read_product(path)
    assert np.array_equal(q.data, p.data) and np.array_equal(q.mask, p.mask)
    assert q.n0 == p.n0 and q.fs == 0.2 and q.gps_start == 1238319104.0 - 50
    assert list(q.tables["files"]["name"]) == ["a.hdf5", "b.hdf5"]
    assert q.attrs["n_gates"] == 3
    with pytest.raises(FileExistsError):
        write_product(p, path)


def test_export_rustyprune(tmp_path: Path) -> None:
    h1, l1 = _product("H1"), _product("L1")
    out = export_rustyprune([h1, l1], tmp_path / "root", tstart=1238319104, duration=900, tseg=450)
    d = np.load(out / "data.npy")
    assert d.shape == (2, 180) and d.dtype == np.dtype("<f8")
    assert np.array_equal(d[0, :170], h1.data[10:180]) and np.all(d[0, 190 - 10 :] == 0)
    band = json.loads((out / "band.json").read_text())
    assert set(band) <= RUSTYPRUNE_KEYS
    assert band["duration"] == 900 and band["ifos"] == ["H1", "L1"]
    meta = json.loads((out / "readligo_meta.json").read_text())
    assert meta["sqrt_sx_zero_filled"] < meta["sqrt_sx_live"]
    with pytest.raises(ValueError, match="grid"):
        export_rustyprune([h1], tmp_path / "r2", tstart=1238319104 + 1, duration=900, tseg=450,
                          allow_any_tstart=True)
    with pytest.raises(ValueError, match="segments"):
        export_rustyprune([h1], tmp_path / "r3", tstart=1238319104, duration=900, tseg=600)


def test_export_checks(tmp_path: Path) -> None:
    """ISSUES.md A6: span noise level, rustyprune's tstart, k/10 band, empty spans, every file guarded."""
    h1 = _product("H1", n0=0, n=400)
    out = export_rustyprune([h1], tmp_path / "a", tstart=1238319104, duration=900, tseg=450)
    meta = json.loads((out / "readligo_meta.json").read_text())
    assert abs(meta["sqrt_sx_live"] / np.sqrt(5e-46) - 1) < 1e-12  # both segments, equal weight
    with pytest.raises(ValueError, match="rustyprune reads"):
        export_rustyprune([h1], tmp_path / "b", tstart=1238319104 + 450, duration=900, tseg=450)
    out = export_rustyprune([h1], tmp_path / "c", tstart=1238319104 + 450, duration=900, tseg=450,
                            allow_any_tstart=True)
    assert abs(json.loads((out / "band.json").read_text())["sqrt_sx"] / 3e-23 - 1) < 1e-12  # second half only
    with pytest.raises(ValueError, match="no live data"):
        export_rustyprune([h1], tmp_path / "d", tstart=1238319104 + 9000, duration=900, tseg=450,
                          allow_any_tstart=True)
    odd = _product("H1", n0=0, n=400)
    odd.band = 0.15
    with pytest.raises(ValueError, match="k/10"):
        export_rustyprune([odd], tmp_path / "e", tstart=1238319104, duration=1800, tseg=900)
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "band.json").write_text("{}")
    with pytest.raises(FileExistsError, match="band.json"):
        export_rustyprune([h1], tmp_path / "f", tstart=1238319104, duration=900, tseg=450)
