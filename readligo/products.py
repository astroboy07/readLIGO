"""Band products: the pipeline's output, its HDF5 file format, and the
export to the layout the rustyprune search reads.

One product holds one detector's cleaned data in one frequency band, over the
whole processed span, on one uniform time grid:

* ``data``: real float64 series sampled at ``fs = 2 * band``, heterodyned at
  ``fmin`` against the global epoch (see :data:`HETERODYNE_CONVENTION`), with
  gaps, gates and vetoed stretches at (or tapered to) zero. This is the same
  convention as the synthetic Makefakedata_v5 series rustyprune is built for.
* ``window``: effective weight of the original strain in each sample, in
  [0, 1] (DQ gaps, gap tapers, gates and narrowband vetoes all lower it).
* ``mask``: ``True`` where the sample is fully good (window == 1).
* noise estimates, gate and veto lists, flagged lines, and every setting
  needed to reproduce the product.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

__all__ = [
    "AMPLITUDE_CONVENTION",
    "RUSTYPRUNE_DEFAULT_TSEG",
    "RUSTYPRUNE_TSTART",
    "FORMAT_VERSION",
    "HETERODYNE_CONVENTION",
    "BandProduct",
    "export_rustyprune",
    "format_freq",
    "product_path",
    "read_product",
    "span_sqrt_sx",
    "write_product",
]

FORMAT_VERSION = "2.0"

HETERODYNE_CONVENTION = (
    "data(t_n) = 2 Re[ x_+(t_n) exp(-2j*pi*fmin*(t_n - epoch)) ], t_n = epoch + n/fs, "
    "where x_+ is the positive-frequency part of the cleaned strain restricted to [fmin, fmin+band]"
)
AMPLITUDE_CONVENTION = (
    "amplitude preserving: a strain component A cos(2 pi f t + phi) with fmin <= f <= fmin+band "
    "appears as A cos(2 pi (f - fmin)(t - epoch) + phi + 2 pi f epoch)"
)


def format_freq(f: float) -> str:
    """Shortest exact decimal form of *f* (``100.0 -> "100"``, ``100.68 -> "100.68"``)."""
    return np.format_float_positional(float(f), trim="-")


@dataclass
class BandProduct:
    """Cleaned data of one detector in one band (see module docstring)."""

    ifo: str
    fmin: float
    band: float
    epoch: float
    n0: int
    data: npt.NDArray[np.float64]
    window: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_]
    sqrt_sx: float
    #: Tables keyed by name, each a dict of equal-length 1-D arrays
    #: (e.g. ``noise_segments``, ``psd``, ``lines``, ``gates``, ``vetoes``, ``runs``).
    tables: dict[str, dict[str, npt.NDArray[Any]]] = field(default_factory=dict)
    #: Scalar metadata: settings, provenance, statistics.
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def fs(self) -> float:
        return 2.0 * self.band

    @property
    def fmax(self) -> float:
        return self.fmin + self.band

    @property
    def gps_start(self) -> float:
        """GPS time of the first sample."""
        return self.epoch + self.n0 / self.fs

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """GPS time of every sample."""
        return self.epoch + (self.n0 + np.arange(self.data.size)) / self.fs

    @property
    def variance(self) -> float:
        """Noise variance of ``data`` in fully good stretches (rustyprune's ``sqrt_sx**2 * band``)."""
        return self.sqrt_sx**2 * self.band

    @property
    def duty_factor(self) -> float:
        """Mean squared window: the fraction of full-duty signal power kept."""
        return float(np.mean(self.window.astype(np.float64) ** 2)) if self.window.size else 0.0


def _grid_name(f: float) -> str:
    """Band edge for file names: 6 decimals, trailing zeros dropped.

    Grid values are at least 1/1800 Hz apart, so 6 decimals keep every name
    distinct (``100.68``, ``1390.885556``, ``101.15``).
    """
    return f"{float(f):.6f}".rstrip("0").rstrip(".")


def product_path(out_dir: str | Path, ifo: str, fmin: float, band: float, gps_start: float, duration: float) -> Path:
    """Canonical file name of a band product, e.g. ``H1_100.68-100.78Hz_1238166018-501246.h5``."""
    return Path(out_dir) / (
        f"{ifo}_{_grid_name(fmin)}-{_grid_name(fmin + band)}Hz_"
        f"{format_freq(gps_start)}-{format_freq(duration)}.h5"
    )


def _attr_value(value: Any) -> Any:
    """Convert a metadata value to something h5py can store as an attribute."""
    if value is None:
        return "None"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return np.array(value, dtype=h5py.string_dtype())
    return value


def write_product(product: BandProduct, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write *product* to HDF5 at *path*.

    The file is written under a unique temporary name and moved into place
    only when complete, so a crash never leaves a partial product and two
    jobs never write the same temporary file. Without *overwrite* the final
    step is an exclusive link, so a product another job finished meanwhile
    is not replaced.

    Raises
    ------
    FileExistsError
        If *path* exists and *overwrite* is false.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format_version"] = FORMAT_VERSION
            f.attrs["ifo"] = product.ifo
            f.attrs["fmin"] = product.fmin
            f.attrs["band"] = product.band
            f.attrs["fmax"] = product.fmax
            f.attrs["fs"] = product.fs
            f.attrs["f_het"] = product.fmin
            f.attrs["epoch"] = product.epoch
            f.attrs["n0"] = product.n0
            f.attrs["gps_start"] = product.gps_start
            f.attrs["n_samples"] = product.data.size
            f.attrs["sqrt_sx"] = product.sqrt_sx
            f.attrs["variance"] = product.variance
            f.attrs["duty_factor"] = product.duty_factor
            f.attrs["heterodyne_convention"] = HETERODYNE_CONVENTION
            f.attrs["amplitude_convention"] = AMPLITUDE_CONVENTION
            for key, value in product.attrs.items():
                f.attrs[key] = _attr_value(value)
            f.create_dataset("data", data=product.data, compression="gzip", shuffle=True)
            f.create_dataset("window", data=product.window.astype(np.float32), compression="gzip", shuffle=True)
            f.create_dataset("mask", data=product.mask.astype(np.uint8), compression="gzip", shuffle=True)
            for name, table in product.tables.items():
                g = f.create_group(name)
                for col, values in table.items():
                    arr = np.asarray(values)
                    if arr.dtype.kind in "UO":
                        arr = arr.astype(object).astype(h5py.string_dtype())
                    g.create_dataset(col, data=arr)
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)  # fails if another job created path meanwhile
            os.unlink(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def read_product(path: str | Path) -> BandProduct:
    """Read a band product written by :func:`write_product`."""
    core = {
        "format_version", "ifo", "fmin", "band", "fmax", "fs", "f_het", "epoch", "n0",
        "gps_start", "n_samples", "sqrt_sx", "variance", "duty_factor",
        "heterodyne_convention", "amplitude_convention",
    }
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        tables = {
            name: {
                col: (f[name][col].asstr()[()] if h5py.check_string_dtype(f[name][col].dtype) else f[name][col][()])
                for col in f[name]
            }
            for name in f
            if isinstance(f[name], h5py.Group)
        }
        product = BandProduct(
            ifo=str(a["ifo"]),
            fmin=float(a["fmin"]),
            band=float(a["band"]),
            epoch=float(a["epoch"]),
            n0=int(a["n0"]),
            data=f["data"][()],
            window=f["window"][()],
            mask=f["mask"][()].astype(bool),
            sqrt_sx=float(a["sqrt_sx"]),
            tables=tables,
            attrs={k: v for k, v in a.items() if k not in core},
        )
    return product


#: rustyprune's default segment length (2 days), used when *tseg* is not given.
RUSTYPRUNE_DEFAULT_TSEG = 172800

#: rustyprune's hard-coded start of the data (``src/simulation/cw_mfdv5.rs``).
RUSTYPRUNE_TSTART = 1238319104


def span_sqrt_sx(product: BandProduct, gps_start: float, gps_end: float) -> float:
    """Live-time average ``sqrt(<Sx>)`` over ``[gps_start, gps_end)`` from the noise segments.

    Each segment is weighted by its duty factor times the fraction of it
    inside the span. NaN if no estimated segment overlaps.
    """
    seg = product.tables.get("noise_segments")
    if not seg or "sx" not in seg:
        return float("nan")
    a = np.asarray(seg["gps_start"], dtype=np.float64)
    d = np.asarray(seg["duration"], dtype=np.float64)
    overlap = np.clip(np.minimum(a + d, gps_end) - np.maximum(a, gps_start), 0.0, None) / d
    sx = np.asarray(seg["sx"], dtype=np.float64)
    w = overlap * np.asarray(seg["duty"], dtype=np.float64)
    ok = np.isfinite(sx) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.sum(w[ok] * sx[ok]) / np.sum(w[ok])))


def export_rustyprune(
    products: list[BandProduct],
    out_dir: str | Path,
    *,
    tstart: int,
    duration: int,
    tseg: int = RUSTYPRUNE_DEFAULT_TSEG,
    sqrt_sx_mode: str = "live",
    overwrite: bool = False,
    allow_any_tstart: bool = False,
) -> Path:
    """Write products in the layout the rustyprune search reads.

    Writes into *out_dir*:

    * ``data.npy``: float64, shape ``(n,)`` for one detector or
      ``(n_det, n)`` for several (rows in ``ifos`` order),
      ``n = duration * 2 * band`` samples, sample k at ``tstart + k/(2 band)``.
      This replaces ``ROOT/frames/data.npy`` after ``rustyprune prepare``
      (whose ``tstart`` is fixed at 1238319104). Time not covered by a
      product, and every gap, is exactly 0.
    * ``mask.npy`` and ``window.npy``, same shape, for search code that can
      use them (rustyprune and pyloki cannot yet; ISSUES.md D1).
    * ``band.json``: the rustyprune band settings that describe this data,
      with only keys rustyprune accepts (``duration``, ``tseg``, ``fmin``,
      ``band``, ``sqrt_sx``, ``ifos``, ``f_inj``, ``n_runs``, ``generator``).
      Add ``patches``, ``scheme`` etc. as needed before ``prepare``.
    * ``readligo_meta.json``: everything else (epoch, conventions, per-detector
      noise and duty factor, both sqrt_sx choices, the usable source band,
      source products).

    ``sqrt_sx`` is computed over the exported span from each product's
    per-segment noise, not taken from the whole product (ISSUES.md A6).

    Parameters
    ----------
    sqrt_sx_mode
        ``"live"`` (default): the noise ASD of the live data, correct for a
        search that applies the mask. ``"zero_filled"``: ``sqrt(duty) x`` that,
        the variance of the zero-filled series, which gives noise 2F with
        mean ~4 in the unchanged searches (signals still lose a factor ~duty;
        a stopgap only, see ISSUES.md D1).
    allow_any_tstart
        rustyprune reads ``data.npy`` as starting at 1238319104 whatever the
        export says, so any other *tstart* raises unless this is True (for
        search code that takes the start time from ``readligo_meta.json``).

    Raises
    ------
    ValueError
        If the products disagree on band or epoch, *tstart* is off their
        sample grid or not rustyprune's, the band is not k/10 Hz,
        ``duration * 2 * band`` is not a whole number, the samples do not
        split into ``duration // tseg >= 2`` equal segments, a product has no
        live sample or no noise estimate in the span, or the data are not
        finite.
    FileExistsError
        If any output file exists and *overwrite* is false.
    """
    if not products:
        raise ValueError("no products to export")
    if sqrt_sx_mode not in ("live", "zero_filled"):
        raise ValueError(f"sqrt_sx_mode must be 'live' or 'zero_filled', not {sqrt_sx_mode!r}")
    if int(duration) != duration or int(tstart) != tstart or int(tseg) != tseg:
        raise ValueError("tstart, duration and tseg must be whole seconds")
    duration, tstart, tseg = int(duration), int(tstart), int(tseg)
    if tstart != RUSTYPRUNE_TSTART and not allow_any_tstart:
        raise ValueError(
            f"tstart {tstart} != {RUSTYPRUNE_TSTART}: rustyprune reads data.npy as starting at "
            f"{RUSTYPRUNE_TSTART} whatever is exported; pass allow_any_tstart=True only for search code "
            "that reads the start time from readligo_meta.json"
        )
    ref = products[0]
    for p in products[1:]:
        if (p.fmin, p.band, p.epoch) != (ref.fmin, ref.band, ref.epoch):
            raise ValueError("products must share fmin, band and epoch")
    if len({p.ifo for p in products}) != len(products):
        raise ValueError("one product per detector")
    k10 = ref.band * 10
    if abs(k10 - round(k10)) > 1e-9 or not (1 <= round(k10) <= 10):
        raise ValueError(f"band {ref.band} Hz: rustyprune accepts only band = k/10 Hz with k = 1..10")
    fs = ref.fs
    n_float = duration * fs
    if abs(n_float - round(n_float)) > 1e-6:
        raise ValueError(f"duration {duration} s x fs {fs} Hz is not a whole number of samples")
    n = int(round(n_float))
    nseg = duration // tseg
    if nseg < 2 or n % nseg:
        raise ValueError(f"{n} samples do not split into duration // tseg = {nseg} (>= 2) equal segments")
    k0 = (tstart - ref.epoch) * fs
    if abs(k0 - round(k0)) > 1e-6:
        raise ValueError(
            f"tstart {tstart} is not on the products' sample grid (epoch {ref.epoch}, fs {fs} Hz); "
            f"re-run the pipeline with epoch = {tstart}"
        )
    k0 = int(round(k0))

    out_dir = Path(out_dir)
    names = ("data.npy", "mask.npy", "window.npy", "band.json", "readligo_meta.json")
    if not overwrite:
        existing = [q for q in names if (out_dir / q).exists()]
        if existing:
            raise FileExistsError(f"{out_dir}: {', '.join(existing)} already exist; pass overwrite=True to replace them.")

    def cut(p: BandProduct, values: npt.NDArray[Any], dtype: type) -> npt.NDArray[Any]:
        out = np.zeros(n, dtype=dtype)
        lo = k0 - p.n0  # index into the product of the first exported sample
        src_lo, src_hi = max(lo, 0), min(lo + n, values.size)
        if src_hi > src_lo:
            out[src_lo - lo : src_hi - lo] = values[src_lo:src_hi]
        return out

    data = np.stack([cut(p, p.data, np.float64) for p in products])
    mask = np.stack([cut(p, p.mask, np.bool_) for p in products])
    window = np.stack([cut(p, p.window, np.float32) for p in products])
    if not np.all(np.isfinite(data)):
        raise ValueError("non-finite values in the exported data")
    duty = (window.astype(np.float64) ** 2).mean(axis=1)
    for p, d in zip(products, duty):
        if not d > 0:
            raise ValueError(f"{p.ifo} product has no live data in [{tstart}, {tstart + duration})")
    span_sx = [span_sqrt_sx(p, tstart, tstart + duration) for p in products]
    for p, v in zip(products, span_sx):
        if not (np.isfinite(v) and v > 0):
            raise ValueError(f"{p.ifo} product has no valid noise estimate in [{tstart}, {tstart + duration})")
    if len(products) == 1:
        data, mask, window = data[0], mask[0], window[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "data.npy", np.ascontiguousarray(data, dtype="<f8"))
    np.save(out_dir / "mask.npy", mask)
    np.save(out_dir / "window.npy", window)

    # rustyprune takes one sqrt_sx for all detectors: the quadratic mean.
    live = float(np.sqrt(np.mean([v**2 for v in span_sx])))
    zero_filled = float(np.sqrt(np.mean([v**2 * d for v, d in zip(span_sx, duty)])))
    band_json: dict[str, Any] = {
        "duration": duration,
        "tseg": tseg,
        "fmin": ref.fmin,
        "band": ref.band,
        "sqrt_sx": live if sqrt_sx_mode == "live" else zero_filled,
        "ifos": [p.ifo for p in products],
        "f_inj": ref.fmin + ref.band / 2,
        "n_runs": min(16, nseg),
        "generator": "lal",
    }
    (out_dir / "band.json").write_text(json.dumps(band_json, indent=2) + "\n")

    usable = ref.attrs.get("usable_source_band")
    meta = {
        "tstart": tstart,
        "duration": duration,
        "tseg": tseg,
        "n_samples": n,
        "fs": fs,
        "fmin": ref.fmin,
        "band": ref.band,
        "f_het": ref.fmin,
        "epoch": ref.epoch,
        "heterodyne_convention": HETERODYNE_CONVENTION,
        "amplitude_convention": AMPLITUDE_CONVENTION,
        "sqrt_sx_mode": sqrt_sx_mode,
        "sqrt_sx_live": live,
        "sqrt_sx_zero_filled": zero_filled,
        "usable_source_band": None if usable is None else [float(u) for u in np.ravel(usable)],
        "detectors": {
            p.ifo: {
                "sqrt_sx_span": span_sx[i],
                "sqrt_sx_whole_product": p.sqrt_sx,
                "duty_factor_exported": float(duty[i]),
                "source_product_gps_start": p.gps_start,
                "source_product_n_samples": int(p.data.size),
                "source_git_commit": str(p.attrs.get("readligo_git_commit", "unknown")),
            }
            for i, p in enumerate(products)
        },
        "notes": (
            "Gaps are zero-filled. rustyprune and pyloki have no mask input yet (ISSUES.md D1), so "
            "with sqrt_sx_mode='live' their noise 2F has mean ~4 x duty. mask.npy/window.npy are for "
            "when they do. f_inj in band.json is only the band centre (rustyprune uses it as the "
            "SFT-bank carrier); set it to a hardware injection's frequency to label it in aggregate. "
            "Only source frequencies in usable_source_band keep the whole Doppler track inside the band."
        ),
    }
    (out_dir / "readligo_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir
