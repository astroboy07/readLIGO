"""One line per band product: noise, duty, lines, vetoes, leakage, flatness, Gaussianity.

    python scripts/validation/product_summary.py OUT_DIR [--workers 8] [--csv out.csv]

Columns:
  sqrt_sx         noise ASD of the live data
  duty            mean window^2 over the product span
  lines           flagged lines; vetoes, xveto%: narrowband vetoes and the
                  fraction of good samples removed by the excess-power veto
  gate_z, edge_z  mean z = x^2/s^2 of fully good samples within 3 samples of a
                  gate / stretch edge (1 = no leakage)
  hi/lo           mean noise power in the top quarter of the band over the
                  bottom quarter (1 = white in band)
  out4            rate of |x|/s > 4 over the Gaussian rate (1 = Gaussian tails)
  kurt            E[z^2]/3 (1 = Gaussian)
"""
from __future__ import annotations

import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from readligo.noise import robust_scale  # noqa: E402
from readligo.products import read_product  # noqa: E402

GAUSS_4 = 6.334e-5


def one(path: str) -> dict[str, float]:
    p = read_product(path)
    a = p.attrs
    s = robust_scale(p.data, p.window, epoch=p.epoch, n0=p.n0, fs=p.fs, segment=1800)
    z = p.data**2 / s
    m = p.mask
    sp = p.tables["spectrum"]
    ok = sp["flat"].astype(bool) & ~sp.get("excluded", np.zeros(sp["flat"].size)).astype(bool)
    f, r = sp["freq"][ok], sp["ratio"][ok]
    q = (f.max() - f.min()) / 4 if f.size else 0
    hilo = float(r[f >= f.max() - q].mean() / r[f <= f.min() + q].mean()) if f.size else np.nan
    good = int(m.sum())
    return {
        "fmin": p.fmin, "band": p.band, "sqrt_sx": p.sqrt_sx, "duty": p.duty_factor,
        "lines": int(a.get("n_lines", 0)), "vetoes": int(a.get("n_vetoes", 0)),
        "xveto_pct": 100 * float(a.get("excess_vetoed_seconds", 0.0)) * p.fs / max(good, 1),
        "gate_z": float(a.get("near_gate_mean_z", np.nan)), "edge_z": float(a.get("near_edge_mean_z", np.nan)),
        "hilo": hilo, "out4": float(np.mean(z[m] > 16) / GAUSS_4), "kurt": float(np.mean(z[m] ** 2) / 3),
        "flattened": bool(np.all(np.isfinite(np.atleast_1d(a.get("inband_shape", [np.nan]))))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv")
    args = ap.parse_args()
    files = sorted(glob.glob(str(Path(args.out_dir) / "*.h5")), key=lambda q: float(Path(q).name.split("_")[1].split("-")[0]))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, files))
    hdr = (f"{'fmin':>11} {'B':>4} {'sqrt_sx':>9} {'duty':>6} {'lines':>5} {'vetoes':>6} {'xveto%':>6} "
           f"{'gate_z':>6} {'edge_z':>6} {'hi/lo':>6} {'out4':>6} {'kurt':>5} flat")
    print(hdr)
    for r in rows:
        print(f"{r['fmin']:11.4f} {r['band']:4.1f} {r['sqrt_sx']:9.3e} {r['duty']:6.4f} {r['lines']:5d} {r['vetoes']:6d} "
              f"{r['xveto_pct']:6.2f} {r['gate_z']:6.2f} {r['edge_z']:6.2f} {r['hilo']:6.3f} {r['out4']:6.2f} "
              f"{r['kurt']:5.3f} {'y' if r['flattened'] else '-'}")
    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
