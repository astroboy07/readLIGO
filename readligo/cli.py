"""Command line: ``python -m readligo plan | check | clean | export``.

* ``plan``: overlapping bands for an all-sky search range (``planning.plan_bands``),
  written as a bands file (``fmin band usable_lo usable_hi`` per line).
* ``check``: inventory, DQ and strain check only.
* ``clean``: the pipeline (``pipeline.run_pipeline``).
* ``export``: band products to rustyprune's ``data.npy`` + ``band.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .conditioning import GateParams
from .config import DEFAULT_EPOCH, PipelineConfig
from .pipeline import check_data, run_pipeline
from .planning import DOPPLER_MAX, plan_bands
from .products import RUSTYPRUNE_DEFAULT_TSEG, export_rustyprune, read_product

__all__ = ["main", "read_bands_file"]


def read_bands_file(path: str | Path) -> list[list[str]]:
    """Bands from a text file: ``fmin band [usable_lo usable_hi]`` per line, separated by
    spaces or commas; ``#`` starts a comment."""
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#")[0].replace(",", " ").strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) not in (2, 4):
            raise ValueError(f"{path}:{n}: expected 'fmin band' or 'fmin band usable_lo usable_hi', got {line!r}")
        out.append(cols)
    return out


def _cfg_dict(args: argparse.Namespace) -> dict:
    cfg_dict = json.loads(Path(args.config).read_text()) if args.config else {}
    for key in ("data_dir", "out_dir", "ifo", "gps_start", "gps_end", "epoch", "work_dir", "workers",
                "chunk_gb", "work_dtype", "bad_strain"):
        value = getattr(args, key, None)
        if value is not None:
            cfg_dict[key] = value
    bands = [list(b) for b in (getattr(args, "band", None) or [])]
    if getattr(args, "bands_file", None):
        bands.extend(read_bands_file(args.bands_file))
    if bands:
        cfg_dict["bands"] = bands
    if args.dq_flags:
        cfg_dict["dq_flags"] = args.dq_flags
    for flag in ("overwrite", "keep_work", "skip_existing"):  # only a given flag overrides --config
        if getattr(args, flag, False):
            cfg_dict[flag] = True
    return cfg_dict


def _clean(args: argparse.Namespace) -> int:
    cfg_dict = _cfg_dict(args)
    gating = dict(cfg_dict.get("gating", {}))
    if args.threshold is not None:
        gating["threshold"] = args.threshold
    if args.no_gating:
        gating["detect"] = False
    if gating:
        cfg_dict["gating"] = gating
    missing = [k for k in ("data_dir", "out_dir", "ifo", "bands") if k not in cfg_dict]
    if missing:
        print(f"missing settings: {', '.join(missing)} (give them as options or in --config)", file=sys.stderr)
        return 2
    cfg = PipelineConfig.from_dict(cfg_dict)
    paths = run_pipeline(cfg)
    for p in paths:
        print(p)
    return 0


def _check(args: argparse.Namespace) -> int:
    cfg_dict = _cfg_dict(args)
    cfg_dict.setdefault("out_dir", ".")
    cfg_dict.setdefault("bands", [["100", "0.1"]])
    missing = [k for k in ("data_dir", "ifo") if k not in cfg_dict]
    if missing:
        print(f"missing settings: {', '.join(missing)}", file=sys.stderr)
        return 2
    rep = check_data(PipelineConfig.from_dict(cfg_dict))
    print(json.dumps(rep, indent=2))
    return 1 if rep["nonfinite_seconds"] or rep["constant_seconds"] else 0


def _plan(args: argparse.Namespace) -> int:
    bw = "auto" if args.bandwidth == "auto" else float(args.bandwidth)
    bands = plan_bands(args.fmin, args.fmax, tspan=args.tspan_days * 86400.0, bandwidth=bw,
                       fdot_range=(args.fdot_min, args.fdot_max), t_ref_frac=args.t_ref_frac,
                       doppler=args.doppler, min_efficiency=args.min_efficiency)
    out = open(args.output, "w") if args.output else sys.stdout
    try:
        print(f"# readligo plan: {args.fmin}-{args.fmax} Hz, T = {args.tspan_days} d, fdot in "
              f"[{args.fdot_min}, {args.fdot_max}] Hz/s, t_ref at {args.t_ref_frac} of T, D = {args.doppler}, "
              f"bandwidth {args.bandwidth}; {len(bands)} bands, {sum(float(b.band) for b in bands):.1f} Hz in total",
              file=out)
        print("# fmin band usable_lo usable_hi   (fmin and band exact on the 1/1800 Hz grid)", file=out)
        for b in bands:
            print(f"{float(b.fmin):.6f} {float(b.band):g} {b.usable[0]:.9f} {b.usable[1]:.9f}", file=out)
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def _export(args: argparse.Namespace) -> int:
    products = [read_product(p) for p in args.products]
    out = export_rustyprune(
        products, args.out_dir, tstart=args.tstart, duration=args.duration, tseg=args.tseg,
        sqrt_sx_mode=args.sqrt_sx_mode, overwrite=args.overwrite, allow_any_tstart=args.allow_any_tstart,
    )
    print(out)
    return 0


def _data_args(c: argparse.ArgumentParser) -> None:
    c.add_argument("--config", help="JSON file with PipelineConfig fields (options override it)")
    c.add_argument("--data-dir", dest="data_dir")
    c.add_argument("--ifo", choices=("H1", "L1"))
    c.add_argument("--gps-start", type=float)
    c.add_argument("--gps-end", type=float)
    c.add_argument("--dq-flags", nargs="+", help="DQ/injection bits that must pass "
                   "(default DATA NO_STOCH_HW_INJ NO_DETCHAR_HW_INJ)")
    c.add_argument("--workers", type=int, help="worker processes (default 1)")
    c.add_argument("-q", "--quiet", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="readligo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("plan", help="overlapping bands for an all-sky search range")
    pl.add_argument("--fmin", type=float, required=True, help="lowest source frequency (Hz)")
    pl.add_argument("--fmax", type=float, required=True, help="highest source frequency (Hz)")
    pl.add_argument("--tspan-days", type=float, required=True, help="observation span (days)")
    pl.add_argument("--bandwidth", default="auto", help="band width in Hz, or 'auto' (default)")
    pl.add_argument("--fdot-min", type=float, default=-1e-9, help="most negative spin-down (Hz/s, default -1e-9)")
    pl.add_argument("--fdot-max", type=float, default=0.0, help="largest spin-up (Hz/s, default 0)")
    pl.add_argument("--t-ref-frac", type=float, default=0.0, help="reference time as a fraction of the span")
    pl.add_argument("--doppler", type=float, default=DOPPLER_MAX, help=f"max |v.n|/c (default {DOPPLER_MAX})")
    pl.add_argument("--min-efficiency", type=float, default=0.5, help="auto: minimum usable fraction of B")
    pl.add_argument("-o", "--output", help="write the bands file here (default stdout)")
    pl.set_defaults(func=_plan, quiet=True)

    k = sub.add_parser("check", help="inventory, DQ and strain check only")
    _data_args(k)
    k.set_defaults(func=_check)

    c = sub.add_parser("clean", help="clean raw GWOSC files into band products")
    _data_args(c)
    c.add_argument("--out-dir", dest="out_dir")
    c.add_argument("--band", nargs=2, action="append", metavar=("FMIN", "BAND"),
                   help="band [FMIN, FMIN+BAND] in Hz, both multiples of 1/1800 Hz; repeatable")
    c.add_argument("--bands-file", help="text file with 'FMIN BAND [USABLE_LO USABLE_HI]' per line (see plan)")
    c.add_argument("--epoch", type=int, help=f"global epoch, integer GPS (default {DEFAULT_EPOCH})")
    c.add_argument("--threshold", type=float, help=f"gate seed threshold in sigma (default {GateParams.threshold})")
    c.add_argument("--no-gating", action="store_true", help="high-pass and tapers only")
    c.add_argument("--work-dir", dest="work_dir", help="scratch parent (a private folder is made in it)")
    c.add_argument("--work-dtype", dest="work_dtype", choices=("float32", "float64"))
    c.add_argument("--chunk-gb", dest="chunk_gb", type=float, help="channelizer output held at once (GB)")
    c.add_argument("--bad-strain", dest="bad_strain", choices=("raise", "drop"))
    c.add_argument("--keep-work", action="store_true")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--skip-existing", action="store_true", help="skip bands whose product exists (resume)")
    c.set_defaults(func=_clean)

    e = sub.add_parser("export", help="write band products as rustyprune's data.npy + band.json")
    e.add_argument("products", nargs="+", help="product .h5 files (one per detector, same band)")
    e.add_argument("--out-dir", dest="out_dir", required=True)
    e.add_argument("--tstart", type=int, default=DEFAULT_EPOCH)
    e.add_argument("--duration", type=int, required=True)
    e.add_argument("--tseg", type=int, default=RUSTYPRUNE_DEFAULT_TSEG)
    e.add_argument("--sqrt-sx-mode", dest="sqrt_sx_mode", choices=("live", "zero_filled"), default="live")
    e.add_argument("--allow-any-tstart", action="store_true",
                   help="allow tstart != 1238319104 (rustyprune itself ignores it)")
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("-q", "--quiet", action="store_true")
    e.set_defaults(func=_export)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
