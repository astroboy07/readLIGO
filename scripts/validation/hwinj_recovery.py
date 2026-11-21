"""Recover the O3 CW hardware injections from readLIGO band products (ISSUES.md V6, V7, C7).

For each injection whose band product exists in PRODUCT_DIR, computes the
known-template coherent F-statistic (cwtools.fstat) on the product:

* 2F at the injection's exact parameters, and its expectation 4 + rho^2 from
  the product's own noise estimate and mask;
* the fitted amplitude and phase against the injected ones (1 and 0 if the
  product preserves amplitude and phase; P14 is scaled by the GWOSC
  anti-alias gain g = 0.0071 at 1991 Hz, which is applied to the expectation);
* an off-source background: 2F at frequency offsets >= 1 mHz from the
  injection, every 0.1 mHz, with the whole template track inside the band's
  flat part (independent trials, since 1/T_span = 2 uHz);
* the frequency of the 2F peak in a fine scan, to show there is no frequency
  bias (ISSUES.md C7).

Usage::

    python scripts/validation/hwinj_recovery.py PRODUCT_DIR [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwtools  # noqa: E402

from readligo import read_product  # noqa: E402

RECOVERABLE = ("P04", "P14", "P01", "P06", "P16", "P05")


def find_product(product_dir: Path, inj: dict) -> Path | None:
    k, kb = inj["band_bins_1800"]
    for f in glob.glob(str(product_dir / "*.h5")):
        import h5py

        with h5py.File(f, "r") as h:
            if round(h.attrs["fmin"] * 1800) == k and round(h.attrs["band"] * 1800) == kb:
                return Path(f)
    return None


def analyse(path: Path, inj: dict, edat) -> dict:
    p = read_product(path)
    g = inj.get("gwosc4k_amplitude_gain") or 1.0
    gain = g if g < 0.5 else 1.0  # only P14 is materially attenuated; elsewhere g is the 1% passband ripple
    shape = np.atleast_1d(np.asarray(p.attrs.get("inband_shape", [np.nan]), dtype=float))
    if shape.size > 1 and np.all(np.isfinite(shape)):  # in-band flattening gain at the injection frequency
        u = 0.5 * sum(inj["h1_track_Hz"]) - shape[0]
        gain *= float(np.exp(-0.5 * sum(a * u ** (k + 1) for k, a in enumerate(shape[1:]))))
    res0 = cwtools.fstat(p, inj, edat, [0.0], gain=gain)
    # off-source background
    guard = float(p.attrs["guard_hz"])
    lo, hi = inj["h1_track_Hz"]
    d_min = p.fmin + guard + 1e-4 - lo
    d_max = p.fmax - guard - 1e-4 - hi
    grid = np.arange(math.ceil(d_min / 1e-4) * 1e-4, d_max, 1e-4)
    grid = grid[np.abs(grid) >= 1e-3]
    bg = cwtools.fstat(p, inj, edat, grid).twoF
    # fine scan for the peak frequency
    T = float(p.data.size / p.fs)
    fine = np.linspace(-30 / T, 30 / T, 241)
    sc = cwtools.fstat(p, inj, edat, fine).twoF
    peak_df = float(fine[np.argmax(sc)])
    p_value = float((np.sum(bg >= res0.twoF[0]) + 1) / (bg.size + 1))
    return {
        "name": inj["name"],
        "product": path.name,
        "band": [p.fmin, p.band],
        "n_samples_used": res0.n_samples,
        "sqrt_sx": p.sqrt_sx,
        "twoF": float(res0.twoF[0]),
        "expected_2F": res0.expected_2F,
        "amplitude_ratio": res0.amplitude_ratio,
        "phase_offset_rad": res0.phase_offset,
        "background_n": int(bg.size),
        "background_mean": float(bg.mean()) if bg.size else float("nan"),
        "background_max": float(bg.max()) if bg.size else float("nan"),
        "p_value": p_value,
        "peak_offset_hz": peak_df,
        "peak_offset_over_1_T": peak_df * T,
        "peak_twoF": float(sc.max()),
        "raw_strain_study": inj["raw_strain_study"],
        "gain_applied": gain,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product_dir", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    injections = cwtools.load_injections()
    edat = cwtools.load_ephemeris()
    rows = []
    for name, inj in injections.items():
        if args.only and name not in args.only:
            continue
        if not inj.get("band_bins_1800"):
            continue
        path = find_product(args.product_dir, inj)
        if path is None:
            continue
        t = time.perf_counter()
        r = analyse(path, inj, edat)
        rows.append(r)
        st = r["raw_strain_study"]
        print(
            f"{name}: 2F {r['twoF']:6.1f} (expected {r['expected_2F']:6.1f}; raw study gated exp/meas "
            f"{st['gated']['expected_2F'] or float('nan'):5.1f}/{st['gated']['measured_2F'] or float('nan'):5.1f}) "
            f"amp {r['amplitude_ratio']:.3f} phase {r['phase_offset_rad']:+.3f} | bg mean {r['background_mean']:.2f} "
            f"max {r['background_max']:.1f} (n={r['background_n']}) p={r['p_value']:.4f} | peak at {r['peak_offset_over_1_T']:+.2f}/T "
            f"({time.perf_counter() - t:.1f} s)",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
