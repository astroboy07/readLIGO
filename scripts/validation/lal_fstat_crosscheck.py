"""Independent check of the hardware-injection 2F with LAL's ComputeFstatistic_v2.

CFSv2 loads SFT bins over the injection's worst-case Doppler band, about
+-0.3 Hz, so this needs products wider than the 0.1 Hz search bands (e.g. 1 Hz
bands around each injection, made by the same pipeline). For each, the fully good 1800 s segments (aligned to the epoch, duty >= 98%) are
turned into SFTs with ``lalpulsar.MakeSFTsFromREAL8TimeSeries`` (the product is
a real series heterodyned at fmin, exactly as Makefakedata_v5 makes them).
``lalapps_ComputeFstatistic_v2`` then evaluates 2F at the injection's exact
parameters, estimating the noise itself (running median per SFT). The same
segments are scored with ``cwtools.fstat``. The two numbers should agree to a
few percent (different noise estimators); a large difference would point to
a bug in the product's timing, heterodyne or amplitude, or in cwtools.

Usage::

    python scripts/validation/lal_fstat_crosscheck.py PRODUCT_DIR --work DIR [--only P04 P06]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import lal
import lalpulsar
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwtools  # noqa: E402
from hwinj_recovery import RECOVERABLE  # noqa: E402

from readligo import read_product  # noqa: E402

TSFT = 1800


def make_sfts(p, out: Path) -> tuple[Path, np.ndarray]:
    fs = p.fs
    per = int(round(TSFT * fs))
    m_first = int(np.ceil(p.n0 / per))
    m_last = int(np.floor((p.n0 + p.data.size) / per))
    w2 = p.window.astype(np.float64) ** 2
    # SFT segments with at least 98% duty (gates inside are fine: their samples are zero/tapered)
    full = [m for m in range(m_first, m_last) if w2[m * per - p.n0 : (m + 1) * per - p.n0].mean() >= 0.98]
    if not full:
        raise ValueError("no fully good SFT segment")
    a = full[0] * per - p.n0
    b = (full[-1] + 1) * per - p.n0
    ts = lal.CreateREAL8TimeSeries(f"{p.ifo}:READLIGO", lal.LIGOTimeGPS(int(p.epoch) + full[0] * TSFT),
                                   p.fmin, 1.0 / fs, lal.StrainUnit, b - a)
    ts.data.data = np.ascontiguousarray(p.data[a:b])
    stamps = lalpulsar.CreateTimestampVector(len(full))
    stamps.deltaT = TSFT
    for i, m in enumerate(full):
        stamps.data[i] = lal.LIGOTimeGPS(int(p.epoch) + m * TSFT)
    sfts = lalpulsar.MakeSFTsFromREAL8TimeSeries(ts, stamps, "rectangular", 0)
    for s in sfts.data:
        s.name = p.ifo  # SFT files need the bare detector prefix
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{p.ifo[0]}-{len(full)}_{p.ifo}_{TSFT}SFT_readligo-{int(p.epoch) + full[0] * TSFT}-{(full[-1] - full[0] + 1) * TSFT}.sft"
    lalpulsar.WriteSFTVector2NamedFile(sfts, str(path), "rectangular", 0, "readligo v2 product")
    return path, np.array(full)


def find_covering_product(product_dir: Path, inj: dict) -> Path | None:
    """The widest product whose band contains the injection's track +- 0.3 Hz (CFSv2's Doppler cover)."""
    import glob

    import h5py

    lo, hi = inj["h1_track_Hz"]
    best = None
    for f in glob.glob(str(product_dir / "*.h5")):
        with h5py.File(f, "r") as h:
            fmin, band = float(h.attrs["fmin"]), float(h.attrs["band"])
        if fmin <= lo - 0.3 and hi + 0.3 <= fmin + band and (best is None or band > best[0]):
            best = (band, Path(f))
    return None if best is None else best[1]


def cfs_v2(sft: Path, inj: dict, work: Path) -> float:
    exe = shutil.which("lalpulsar_ComputeFstatistic_v2") or shutil.which("lalapps_ComputeFstatistic_v2")
    if exe is None:
        sys.exit("ComputeFstatistic_v2 not found on PATH")
    import solar_system_ephemerides as sse

    base = Path(sse.__file__).parent / "ephemerides"
    loud = work / f"loud_{inj['name']}.dat"
    cmd = [exe, f"--DataFiles={sft}", f"--Alpha={inj['ra_rad']!r}", f"--Delta={inj['dec_rad']!r}",
           f"--Freq={inj['f0_gw_Hz']!r}", f"--f1dot={inj['f1dot_gw_Hz_s']!r}", f"--refTime={inj['ref_time_gps']}",
           "--FreqBand=0", "--f1dotBand=0", f"--outputLoudest={loud}",
           f"--ephemEarth={base / 'earth' / 'earth00-40-DE405.dat.gz'}",
           f"--ephemSun={base / 'sun' / 'sun00-40-DE405.dat.gz'}"]
    b = inj.get("binary")
    if b:
        cmd += [f"--orbitasini={b['asini_lts']!r}", f"--orbitPeriod={b['porb_s']!r}",
                f"--orbitTp={b['tp_ssb_gps']!r}", f"--orbitArgp={b['argp_rad']!r}", "--orbitEcc=0",
                # resampling has no SFT-length loss; the length check is for the SFT-based methods
                "--FstatMethod=ResampBest", "--allowedMismatchFromSFTLength=10"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"ComputeFstatistic_v2 failed:\n{proc.stderr[-3000:]}")
    for line in loud.read_text().splitlines():
        if line.strip().startswith("twoF"):
            return float(line.split("=")[1].strip().rstrip(";"))
    raise RuntimeError("twoF not found in --outputLoudest")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product_dir", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=list(RECOVERABLE))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    injections = cwtools.load_injections()
    edat = cwtools.load_ephemeris()
    rows = []
    for name in args.only:
        inj = injections[name]
        path = find_covering_product(args.product_dir, inj)
        if path is None:
            continue
        p = read_product(path)
        sft, full = make_sfts(p, args.work / name)
        lal2f = cfs_v2(sft, inj, args.work)
        q = copy.copy(p)
        per = int(round(TSFT * p.fs))
        sel = np.zeros(p.data.size, dtype=bool)
        for m in full:
            sel[m * per - p.n0 : (m + 1) * per - p.n0] = True
        q.mask = (p.window > 0) & sel  # the same samples the SFTs contain
        g = inj.get("gwosc4k_amplitude_gain") or 1.0
        ours = cwtools.fstat(q, inj, edat, [0.0], gain=g if g < 0.5 else 1.0)
        rows.append({"name": name, "n_sfts": int(full.size), "twoF_lal_cfsv2": lal2f,
                     "twoF_readligo_cwtools": float(ours.twoF[0]), "expected_2F": ours.expected_2F})
        print(f"{name}: {full.size} full SFTs | ComputeFstatistic_v2 2F = {lal2f:6.1f} | cwtools 2F = "
              f"{ours.twoF[0]:6.1f} (expected {ours.expected_2F:5.1f})", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
