"""Real-data checks of the v2 products against ISSUES.md C1, C3, C4, C6, C8, O4.

* C1: on a 6 h gap-free H1 stretch, the band output with and without the
  10/15 Hz high-pass must have the same variance (v1: 789x, 493x, 901x in
  71-72, 56-57, 519-520 Hz). Also: none of v1's fake lines in the products.
* C3: every product is sampled at exactly 2B with a whole number of samples.
* C4: no fully good sample above 30 sigma^2 (sigma^2 = Sx B) in any band, and
  the usable fraction is the same at 0.1 Hz and 1 Hz.
* C6: no excess power near internal file boundaries.
* C8: the time grid is uniform (by construction: t_n = epoch + n/fs).
* O4: the variance of good samples over Sx B is ~1 - 2.5 mHz / B.

Usage::

    python scripts/validation/real_data_checks.py PRODUCT_DIR --data-dir RAW_DIR [--json out.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from readligo import (
    ConditionedStrain,
    FFTChannelizer,
    GateParams,
    RunData,
    condition_run,
    contiguous_runs,
    read_product,
    scan_directory,
)

C1_STRETCH = (1238386000, 1238386000 + 6 * 3600)
C1_BANDS = [(56, 1), (71, 1), (519, 1), (100, 1), (56, 0.1), (71.9, 0.1), (120, 0.1), (135.9, 0.1),
            (199.9, 0.1), (519.9, 0.1), (1016, 0.1)]
V1_FAKE_LINES = [56.07, 71.93, 120.07, 135.93, 199.93, 519.93, 1016.07]


def c1_highpass_ratio(data_dir: Path, work: Path) -> list[dict]:
    files = scan_directory(data_dir, "H1", C1_STRETCH)
    run = next(r for r in contiguous_runs(files) if r[0].gps_start <= C1_STRETCH[0])
    out = []
    results = {}
    for tag, p in (("hp", GateParams(detect=False)), ("nohp", GateParams(detect=False, hp_type="none"))):
        rd = RunData(run, ("DATA",))
        t = rd.gps_start + np.arange(rd.duration)
        rd.good &= (t >= C1_STRETCH[0]) & (t < C1_STRETCH[1])
        assert rd.good.sum() == C1_STRETCH[1] - C1_STRETCH[0], "C1 stretch is not fully DQ-good"
        cond = condition_run(rd, p, work / f"c1_{tag}.npy")
        ch = FFTChannelizer(C1_BANDS, 1238319104)
        results[tag] = ch.run(ConditionedStrain([cond], [], p, 1238319104), C1_STRETCH[0] + 1800,
                              C1_STRETCH[1] - 1800)
        cond.path.unlink()
    for (fmin, band), (_, y_hp), (_, y_no) in zip(C1_BANDS, results["hp"], results["nohp"]):
        ratio = float(np.var(y_no) / np.var(y_hp))
        out.append({"band": [fmin, band], "ratio_without_over_with_highpass": ratio})
        print(f"C1 [{fmin}, {fmin + band}] Hz: var(no high-pass) / var(high-pass) = {ratio:.8f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product_dir", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--work", type=Path, default=Path("_work_checks"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    report: dict = {"bands": []}

    for f in sorted(glob.glob(str(args.product_dir / "*.h5"))):
        p = read_product(f)
        fs_exact = 2 * Fraction(p.attrs["band_exact"])
        seg = p.tables["noise_segments"]
        sx = np.where(np.isfinite(seg["sx"]), seg["sx"], p.sqrt_sx**2)
        t = p.times
        k = np.clip(np.searchsorted(seg["gps_start"], t, side="right") - 1, 0, sx.size - 1)
        z = p.data**2 / (sx[k] * p.band)
        good = p.mask
        # C6: internal boundaries between back-to-back files
        files = p.tables["files"]
        ends = files["gps_start"] + files["duration"]
        internal = files["gps_start"][1:][files["gps_start"][1:] == ends[:-1]]
        near = np.zeros(t.size, dtype=bool)
        for b in internal:
            near |= np.abs(t - b) <= 60
        near &= good
        dt = np.diff(t[:1000])
        row = {
            "file": Path(f).name,
            "fmin": p.fmin,
            "band": p.band,
            "fs_exact": str(fs_exact),
            "fs_is_2B": fs_exact == 2 * Fraction(p.attrs["band_exact"]) and float(fs_exact) == p.fs,
            "uniform_grid": bool(np.allclose(dt, 1 / p.fs, rtol=1e-12, atol=0)),
            "usable_fraction": float(good.mean()),
            "duty_factor": p.duty_factor,
            "n_good": int(good.sum()),
            "n_above_30": int(np.sum(z[good] > 30)),
            "max_z": float(z[good].max()),
            "mean_z_good": float(z[good].mean()),
            "n_internal_boundaries": int(internal.size),
            "mean_z_near_boundaries": float(z[near].mean()) if near.any() else float("nan"),
            "max_z_near_boundaries": float(z[near].max()) if near.any() else float("nan"),
            "variance_ratio": float(p.attrs["variance_ratio"]),
            "variance_ratio_expected": 1 - 2 * 1.2617e-3 / p.band,
            "lines": [float(x) for x in p.tables["lines"]["freq"]],
        }
        row["v1_fake_lines_present"] = [fl for fl in V1_FAKE_LINES
                                        if p.fmin <= fl <= p.fmax and any(abs(fl - x) < 3e-3 for x in row["lines"])]
        report["bands"].append(row)
        print(f"{p.fmin:>10.6f}+{p.band:<4} fs={row['fs_exact']:>6} usable {row['usable_fraction']:.4f} "
              f"n>30sig2 {row['n_above_30']} max z {row['max_z']:6.1f} mean z {row['mean_z_good']:.3f} | "
              f"near {row['n_internal_boundaries']} boundaries: mean z {row['mean_z_near_boundaries']:.3f} "
              f"max {row['max_z_near_boundaries']:.1f} | var ratio {row['variance_ratio']:.4f} "
              f"(exp {row['variance_ratio_expected']:.4f}) fake lines {row['v1_fake_lines_present']}")
    if args.data_dir:
        args.work.mkdir(parents=True, exist_ok=True)
        report["c1"] = c1_highpass_ratio(args.data_dir, args.work)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
