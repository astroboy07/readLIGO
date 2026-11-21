"""In-band flatness of band products (ISSUES.md A8).

For each product, from its segment-averaged spectrum (power / Sx on the flat
part of the band, line bins excluded): the mean in the lowest and highest
quarter of the band and their ratio. A white in-band noise gives 1.00, 1.00.
The searches assume one variance per band, so a ratio far from 1 biases the
noise 2F at the band ends.

    python scripts/validation/inband_flatness.py OUT_DIR/*.h5
"""
from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from readligo.products import read_product  # noqa: E402


def main() -> None:
    print(f"{'band':>16} {'low quarter':>11} {'high quarter':>12} {'high/low':>8} {'g(fmin)':>8} {'g(fmax)':>8}")
    for path in sorted(sys.argv[1:]):
        p = read_product(path)
        sp = p.tables["spectrum"]
        f, r, flat = sp["freq"], sp["ratio"], sp["flat"].astype(bool)
        if "excluded" in sp:
            flat &= ~sp["excluded"].astype(bool)
        ff, rr = f[flat], r[flat]
        q = (ff.max() - ff.min()) / 4
        lo, hi = rr[ff <= ff.min() + q].mean(), rr[ff >= ff.max() - q].mean()
        g0, g1 = p.attrs.get("inband_gain_fmin", 1.0), p.attrs.get("inband_gain_fmax", 1.0)
        print(f"{p.fmin:9.3f}+{p.band:<6.2f} {lo:11.3f} {hi:12.3f} {hi / lo:8.3f} {g0:8.3f} {g1:8.3f}")


if __name__ == "__main__":
    main()
