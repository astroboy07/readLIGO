"""Score the CW hardware injections on O4a band products (known-template F-statistic).

The O4 injections keep the O3 phase-evolution parameters (``hwinj_o3a.json``)
but most have new amplitudes. ``O4_H0`` holds the O4 values of h0 from
Table I of arXiv:2603.14168 (LVK O4a all-sky search, "Inj" n = Pn); P16 is
a binary and is not in that table, so its O3 h0 is used.

For each injection the product whose flat band best contains its track is
used. Reports 2F, the expectation from the product's own noise and mask, the
fitted amplitude (1 = as injected) and phase, and an off-source background
(2F at random frequency offsets >= 2 mHz inside the flat band).

    python scripts/validation/hwinj_o4a_check.py PRODUCT_DIR [P05 P06 P16 P01 P04 P14]

Result on O4a H1 (3,901 files, 160.6 live days, 2026-11-21): amplitudes
0.956 (P05), 0.981 (P06), 0.962 (P16, O3 h0), 0.969 (P01), 1.004 (P04),
0.981 (P14); phases within 0.05 rad; 2F 3,300-6,000 against a background
mean of 3.3-4.6.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cwtools  # noqa: E402

from readligo import read_product  # noqa: E402

#: O4 h0 per injection (arXiv:2603.14168, Table I).
O4_H0 = {"P00": 4.01e-26, "P01": 2.79e-25, "P02": 3.20e-26, "P03": 1.30e-25, "P04": 4.90e-25, "P05": 3.99e-25,
         "P06": 3.20e-25, "P07": 7.16e-26, "P08": 8.28e-26, "P09": 6.87e-26, "P10": 6.26e-25, "P11": 3.17e-25,
         "P12": 2.63e-25, "P14": 7.61e-25}
DOPPLER = 1.06e-4


def main() -> None:
    d = sys.argv[1]
    names = sys.argv[2:] or ["P05", "P06", "P16", "P01", "P04", "P14"]
    injs = cwtools.load_injections()
    edat = cwtools.load_ephemeris()
    files = glob.glob(str(Path(d) / "*.h5"))
    for name in names:
        inj = dict(injs[name])
        if name in O4_H0:
            inj["h0"] = O4_H0[name]
        with h5py.File(files[0]) as h:
            t0 = float(h.attrs["span_gps_start"])
        f_src = inj["f0_gw_Hz"] + inj["f1dot_gw_Hz_s"] * (t0 - inj["ref_time_gps"])
        best = None
        for f in files:
            with h5py.File(f) as h:
                lo, hi = h.attrs["flat_band"]
            mg = DOPPLER * f_src + 1e-3
            if lo + mg < f_src < hi - mg:
                w = min(f_src - lo, hi - f_src)
                if best is None or w > best[0]:
                    best = (w, f)
        if best is None:
            print(f"{name}: no product holds its track")
            continue
        p = read_product(best[1])
        g = inj["gwosc4k_amplitude_gain"]
        gain = g if g < 0.5 else 1.0
        sh = np.atleast_1d(np.asarray(p.attrs.get("inband_shape", [np.nan]), float))
        if sh.size > 1 and np.all(np.isfinite(sh)):
            u = f_src - sh[0]
            gain *= float(np.exp(-0.5 * sum(a * u ** (k + 1) for k, a in enumerate(sh[1:]))))
        t = time.time()
        r = cwtools.fstat(p, inj, edat, [0.0], gain=gain)
        lo, hi = p.attrs["flat_band"]
        half = max((hi - lo) / 2 - DOPPLER * f_src - 2e-3, 3e-3)
        offs = np.random.default_rng(0).uniform(-1, 1, 400) * half + (0.5 * (lo + hi) - f_src)
        offs = offs[np.abs(offs) > 2e-3][:40]
        bg = cwtools.fstat(p, inj, edat, offs).twoF
        print(f"{name}: {p.fmin:.4f}+{p.band:.1f} Hz  2F {r.twoF[0]:9.1f}  expected {r.expected_2F:9.1f}  "
              f"amplitude {r.amplitude_ratio:.3f}  phase {r.phase_offset:+.3f}  background mean {bg.mean():.2f} "
              f"max {bg.max():.1f}  ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
