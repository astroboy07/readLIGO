"""Line leakage and outlier statistics of band products near gates and stretch edges.

For each product: z = data^2 / s^2 with s^2 the robust per-segment variance
(median of x^2 / 0.4549 over live samples of each 1800 s segment). Reports the
mean z of fully good samples (mask = 1) within ``--near`` samples of a gate or
of a stretch edge, the mean z far from both, and the rate of |x|/s > 4 against
the Gaussian rate. Leakage of strong lines through gates shows up as a near-gate
mean z well above 1 (ISSUES.md A5).

    python scripts/validation/leakage_check.py OUT_DIR/*.h5 [--near 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from readligo.noise import robust_scale  # noqa: E402
from readligo.products import read_product  # noqa: E402

GAUSS_4SIGMA = 6.334e-5  # P(|N(0,1)| > 4)


def near_samples(times: np.ndarray, t_lo: np.ndarray, t_hi: np.ndarray, reach: float) -> np.ndarray:
    """Boolean: sample time within *reach* s of any interval [t_lo, t_hi]."""
    out = np.zeros(times.size, dtype=bool)
    if t_lo.size == 0:
        return out
    order = np.argsort(t_lo)
    lo, hi = t_lo[order] - reach, t_hi[order] + reach
    i = np.searchsorted(lo, times, side="right") - 1
    ok = i >= 0
    out[ok] = times[ok] <= np.maximum.accumulate(hi)[i[ok]]
    return out


def stats(path: str, near: int) -> dict[str, float]:
    p = read_product(path)
    fs = p.fs
    x = p.data
    scale = robust_scale(x, p.window, epoch=p.epoch, n0=p.n0, fs=fs, segment=1800)
    z = x**2 / scale
    t = p.times
    taper = float(p.attrs.get("gating_gate_taper", 0.25))
    edge = float(p.attrs.get("gating_edge_taper", 1.0))
    g = p.tables["gates"]
    st = p.tables["stretches"]
    reach = near / fs
    near_gate = near_samples(t, g["gps_start"] - taper, g["gps_end"] + taper, reach)
    edges = np.concatenate([st["gps_start"], st["gps_end"] - edge]).astype(float)
    near_edge = near_samples(t, edges, edges + edge, reach)
    good = p.mask
    far = good & ~near_samples(t, g["gps_start"] - taper, g["gps_end"] + taper, 60.0) & ~near_samples(
        t, edges, edges + edge, 60.0)
    out = {
        "fmin": p.fmin,
        "near_gate_z": float(z[good & near_gate].mean()) if np.any(good & near_gate) else np.nan,
        "n_near_gate": int(np.sum(good & near_gate)),
        "near_edge_z": float(z[good & near_edge].mean()) if np.any(good & near_edge) else np.nan,
        "n_near_edge": int(np.sum(good & near_edge)),
        "far_z": float(z[far].mean()),
        "out4_ratio": float(np.mean(z[good] > 16.0) / GAUSS_4SIGMA),
        "n_vetoes": int(p.attrs.get("n_vetoes", 0)),
        "duty": p.duty_factor,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("products", nargs="+")
    ap.add_argument("--near", type=int, default=3, help="samples from a gate or edge (default 3)")
    a = ap.parse_args()
    print(f"{'fmin':>10} {'near_gate_z':>11} {'n':>6} {'near_edge_z':>11} {'n':>5} {'far_z':>6} "
          f"{'4sig/gauss':>10} {'vetoes':>6} {'duty':>6}")
    for path in sorted(a.products):
        s = stats(path, a.near)
        print(f"{s['fmin']:10.4f} {s['near_gate_z']:11.2f} {s['n_near_gate']:6d} {s['near_edge_z']:11.2f} "
              f"{s['n_near_edge']:5d} {s['far_z']:6.3f} {s['out4_ratio']:10.2f} {s['n_vetoes']:6d} {s['duty']:6.4f}")


if __name__ == "__main__":
    main()
