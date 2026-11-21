"""Code and library versions, recorded in every product so it can be reproduced."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy

__all__ = ["provenance_attrs"]


def _git(*args: str) -> str | None:
    """Run git in the package directory; None if git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def provenance_attrs() -> dict[str, str | bool]:
    """Git commit (and whether ``readligo/`` had uncommitted changes), versions, time."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".")
    return {
        "readligo_git_commit": commit or "unknown",
        "readligo_git_dirty": bool(status) if status is not None else True,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "h5py_version": h5py.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
