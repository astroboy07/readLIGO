"""Find raw GWOSC files and group them into contiguous runs.

GWOSC 4 kHz files are named ``<obs>-<IFO>_GWOSC_<run>_4KHZ_R<n>-<gps>-<dur>.hdf5``
and adjacent files are sample-continuous. A *run* is a maximal sequence of
files where each one starts exactly where the previous one ends; the pipeline
processes each run as one continuous series, so file boundaries never act as
data edges (ISSUES.md C6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .io import read_meta

__all__ = ["DETECTORS", "RawFile", "contiguous_runs", "scan_directory"]

#: Detectors the pipeline supports.
DETECTORS: tuple[str, ...] = ("H1", "L1")

_NAME = re.compile(r"^[A-Z]-(?P<ifo>[A-Z][0-9])_GWOSC_.*-(?P<gps>\d+)-(?P<dur>\d+)\.hdf5$")


@dataclass(frozen=True, order=True)
class RawFile:
    """One raw strain file. Sorting orders files by GPS start."""

    gps_start: int
    duration: int
    ifo: str
    path: Path

    @property
    def gps_end(self) -> int:
        return self.gps_start + self.duration


def scan_directory(
    data_dir: str | Path,
    ifo: str,
    gps_range: tuple[float, float] | None = None,
    *,
    verify: bool = True,
) -> list[RawFile]:
    """List the GWOSC files of one detector in *data_dir*, sorted by GPS start.

    Parameters
    ----------
    data_dir
        Directory holding the raw ``.hdf5`` files.
    ifo
        Detector to select (``"H1"`` or ``"L1"``).
    gps_range
        Keep only files overlapping ``[start, end)``.
    verify
        Open each file and check that ``meta/Detector``, ``meta/GPSstart`` and
        ``meta/Duration`` agree with its name.

    Raises
    ------
    ValueError
        For an unsupported detector, a name/content mismatch, or overlapping
        files.
    """
    if ifo not in DETECTORS:
        raise ValueError(f"detector {ifo!r} is not one of {DETECTORS}")
    files: list[RawFile] = []
    for path in Path(data_dir).glob("*.hdf5"):
        m = _NAME.match(path.name)
        if m is None or m["ifo"] != ifo:
            continue
        raw = RawFile(int(m["gps"]), int(m["dur"]), ifo, path)
        if gps_range is not None and not (raw.gps_end > gps_range[0] and raw.gps_start < gps_range[1]):
            continue
        if verify:
            meta = read_meta(path)
            if (meta["Detector"], meta["GPSstart"], meta["Duration"]) != (ifo, raw.gps_start, raw.duration):
                raise ValueError(
                    f"{path.name}: name says {ifo} {raw.gps_start}+{raw.duration} s but meta says "
                    f"{meta['Detector']} {meta['GPSstart']}+{meta['Duration']} s"
                )
        files.append(raw)
    files.sort()
    for a, b in zip(files, files[1:]):
        if b.gps_start < a.gps_end:
            raise ValueError(f"{a.path.name} and {b.path.name} overlap in time")
    return files


def contiguous_runs(files: list[RawFile]) -> list[list[RawFile]]:
    """Split time-sorted files into runs of back-to-back files."""
    runs: list[list[RawFile]] = []
    for f in files:
        if runs and runs[-1][-1].gps_end == f.gps_start:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs
