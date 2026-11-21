"""Check that the channelizer reproduces Makefakedata_v5 narrowband frames.

MFDv5 (the generator behind rustyprune's and pyloki's synthetic data) can
write the same noise-free CW both broadband (fmin = 0, Band = 2048 Hz, i.e.
4096 Hz strain) and narrowband (heterodyned at fmin, sampled at 2B). Feeding
the broadband series through readLIGO's channelizer with t0 = tstart must give
the narrowband series sample for sample. The comparison uses the interior of
the record (L/2 = 1800 s from each end), since the broadband record is
truncated there and MFDv5 has no edge.

Usage::

    python scripts/validation/mfdv5_equivalence.py --fixtures DIR   # existing .npy fixtures
    python scripts/validation/mfdv5_equivalence.py --work DIR        # run lalpulsar_Makefakedata_v5

``--fixtures`` expects ``bb4096_signal_H1_1238319104_14400.npy`` and
``nb_signal_H1_fmin{fmin}_B{B}.npy`` as written by the November 2026 study.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from readligo import FFTChannelizer

T0, DUR = 1238319104, 14400
SOURCE = "{Alpha=2.5; Delta=0.4; Freq=100.7325; f1dot=-1e-10; h0=1e-22; cosi=0.3; psi=0.7; phi0=0.0; refTime=1238326304}"
BANDS = [(100.68, 0.1), (100.48, 0.5), (100.18, 1.0)]


def ephem() -> tuple[str, str]:
    import solar_system_ephemerides as sse

    base = Path(sse.__file__).parent / "ephemerides"
    return str(base / "earth" / "earth00-40-DE405.dat.gz"), str(base / "sun" / "sun00-40-DE405.dat.gz")


def mfdv5(work: Path, fmin: float, band: float) -> np.ndarray:
    import lal
    import lalframe

    exe = shutil.which("lalpulsar_Makefakedata_v5")
    if exe is None:
        sys.exit("lalpulsar_Makefakedata_v5 not found on PATH")
    out = work / f"f{fmin}_B{band}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    e, s = ephem()
    cmd = [exe, "--IFOs=H1", "--sqrtSX=0.0", f"--startTime={T0}", f"--duration={DUR}", f"--fmin={fmin!r}",
           f"--Band={band!r}", f"--outFrameDir={out}", "--outLabel=CW", f"--injectionSources={SOURCE}",
           "--randSeed=30", f"--ephemEarth={e}", f"--ephemSun={s}"]
    subprocess.run(cmd, check=True, capture_output=True)
    frame = next(out.glob("*.gwf"))
    st = lalframe.FrStreamOpen(str(out), frame.name)
    ts = lalframe.FrStreamReadREAL8TimeSeries(st, "H1:CW", lal.LIGOTimeGPS(T0), float(DUR), 0)
    return np.array(ts.data.data, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixtures", type=Path)
    g.add_argument("--work", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.fixtures:
        bb = np.load(args.fixtures / f"bb4096_signal_H1_{T0}_{DUR}.npy")
        nbs = [np.load(args.fixtures / f"nb_signal_H1_fmin{f}_B{b}.npy") for f, b in BANDS]
    else:
        bb = mfdv5(args.work, 0.0, 2048.0)
        nbs = [mfdv5(args.work, f, b) for f, b in BANDS]
    if bb.size != DUR * 4096:
        sys.exit(f"broadband series has {bb.size} samples, expected {DUR * 4096}")

    def reader(gps: int, n: int) -> np.ndarray:
        out = np.zeros(n)
        i = (gps - T0) * 4096
        a, b = max(0, -i), min(n, bb.size - i)
        if b > a:
            out[a:b] = bb[i + a : i + b]
        return out

    ch = FFTChannelizer(BANDS, T0)
    outs = ch.run(reader, T0, T0 + DUR)
    rows = []
    for (fmin, band), (n0, y), nb in zip(BANDS, outs, nbs):
        fs = 2 * band
        assert n0 == 0 and y.size == nb.size == round(DUR * fs)
        k = slice(round(1800 * fs), round((DUR - 1800) * fs))
        res = y[k] - nb[k]
        rel = float(np.sqrt(np.mean(res**2) / np.mean(nb[k] ** 2)))
        gfit = complex(np.vdot(nb[k], y[k]) / np.vdot(nb[k], nb[k]))
        rows.append({"fmin": fmin, "band": band, "interior_rel_rms": rel, "gain_minus_1": abs(gfit) - 1,
                     "max_abs_err_over_peak": float(np.max(np.abs(res)) / np.max(np.abs(nb[k])))})
        print(f"[{fmin}, {fmin + band}] Hz: interior relative rms {rel:.2e}, |gain|-1 {abs(gfit) - 1:+.1e}, "
              f"max error {rows[-1]['max_abs_err_over_peak']:.1e} of peak")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
