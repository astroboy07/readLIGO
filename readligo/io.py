"""LIGO HDF5 Data Reader.

This module provides functions to read LIGO gravitational wave data
from HDF5 files (GWOSC format). It extracts strain data, data quality
information, injection masks, and metadata.

Only HDF5 files (.hdf5) are supported. For frame files (.gwf),
use the LALSuite tools directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

# Configure logging
logger = logging.getLogger(__name__)

# Public API
__all__ = ["GwoscFile", "loaddata", "read_gwosc", "read_hdf5", "read_meta"]


def read_hdf5(
    filename: str,
    *,
    readstrain: bool = True,
) -> tuple[
    npt.NDArray[np.float64],
    float,
    float,
    npt.NDArray[np.int32],
    list[str],
    npt.NDArray[np.int32],
    list[str],
]:
    """Read HDF5 files and extract LIGO data (legacy tuple interface).

    New code should use :func:`read_gwosc`, which validates the file and
    returns named fields.
    """
    with h5py.File(filename, "r") as data_file:
        # -- Read the strain
        strain_group = cast(h5py.Group, data_file["strain"])
        strain_ds = cast(h5py.Dataset, strain_group["Strain"])
        strain: npt.NDArray[np.float64] = (
            cast(npt.NDArray[np.float64], strain_ds[...])
            if readstrain
            else np.empty(0, dtype=np.float64)
        )
        ts = float(cast(np.floating, strain_ds.attrs["Xspacing"]))

        # -- Read the DQ information
        quality_group = cast(h5py.Group, data_file["quality"])
        dq_info = cast(h5py.Group, quality_group["simple"])
        qmask = cast(h5py.Dataset, dq_info["DQmask"])[...]
        shortname_array = cast(h5py.Dataset, dq_info["DQShortnames"])[...]
        shortname_list = list(shortname_array)

        # -- Read the INJ information
        inj_info = cast(h5py.Group, data_file["quality/injections"])
        injmask = cast(h5py.Dataset, inj_info["Injmask"])[...]
        injname_array = cast(h5py.Dataset, inj_info["InjShortnames"])[...]
        injname_list = list(injname_array)

        # -- Read the meta data
        meta = cast(h5py.Group, data_file["meta"])
        gps_start = float(cast(np.floating, cast(h5py.Dataset, meta["GPSstart"])[...]))

    return strain, gps_start, ts, qmask, shortname_list, injmask, injname_list


def _decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _bitmask_channels(
    mask: npt.NDArray[np.integer], names: npt.NDArray[np.bytes_]
) -> dict[str, npt.NDArray[np.bool_]]:
    """Split a GWOSC bitmask time series into one boolean array per bit."""
    return {_decode(name): ((mask >> bit) & 1).astype(bool) for bit, name in enumerate(names)}


@dataclass(frozen=True)
class GwoscFile:
    """Contents of one GWOSC strain file, validated.

    Attributes
    ----------
    path
        File the data came from.
    ifo
        Detector name from ``meta/Detector`` (e.g. ``"H1"``).
    gps_start, duration
        Integer GPS start and length in seconds.
    fs
        Sample rate in Hz, from ``strain/Strain.attrs["Xspacing"]``.
    strain_channel
        LIGO channel the strain was taken from.
    dq
        One boolean array per DQ bit (1 value per second, ``True`` = passes).
    injections
        One boolean array per hardware-injection bit (1 value per second;
        GWOSC names them ``NO_*_HW_INJ``, so ``True`` = no injection).
    strain
        Strain samples (NaN where ``DATA`` is off), or ``None`` if not read.
    """

    path: Path
    ifo: str
    gps_start: int
    duration: int
    fs: int
    strain_channel: str
    dq: dict[str, npt.NDArray[np.bool_]]
    injections: dict[str, npt.NDArray[np.bool_]]
    strain: npt.NDArray[np.float64] | None


def read_gwosc(path: str | Path, *, read_strain: bool = True) -> GwoscFile:
    """Read and validate one GWOSC HDF5 strain file.

    Raises
    ------
    ValueError
        If the file is empty, not ``.hdf5``, or internally inconsistent (sample
        spacing not 1/integer Hz, sample count or DQ length not matching the
        duration, or ``Xstart`` not matching ``GPSstart``).
    """
    path = Path(path)
    if path.suffix.lower() != ".hdf5":
        raise ValueError(f"Only HDF5 files (.hdf5) are supported, got {path.name}")
    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    with h5py.File(path, "r") as f:
        meta = cast(h5py.Group, f["meta"])
        ifo = _decode(cast(h5py.Dataset, meta["Detector"])[()])
        gps_start = int(cast(h5py.Dataset, meta["GPSstart"])[()])
        duration = int(cast(h5py.Dataset, meta["Duration"])[()])
        strain_channel = _decode(cast(h5py.Dataset, meta["StrainChannel"])[()])

        strain_ds = cast(h5py.Dataset, f["strain/Strain"])
        spacing = float(cast(np.floating, strain_ds.attrs["Xspacing"]))
        fs = int(round(1.0 / spacing))
        if abs(fs * spacing - 1.0) > 1e-9:
            raise ValueError(f"{path.name}: Xspacing {spacing} is not 1/integer Hz")
        if strain_ds.shape != (duration * fs,):
            raise ValueError(
                f"{path.name}: {strain_ds.shape[0]} samples, expected {duration} s x {fs} Hz"
            )
        if int(cast(np.integer, strain_ds.attrs["Xstart"])) != gps_start:
            raise ValueError(f"{path.name}: strain Xstart does not match meta/GPSstart")
        strain = cast(npt.NDArray[np.float64], strain_ds[...]) if read_strain else None

        simple = cast(h5py.Group, f["quality/simple"])
        dq_mask = cast(h5py.Dataset, simple["DQmask"])[...]
        if dq_mask.shape != (duration,):
            raise ValueError(f"{path.name}: DQmask has {dq_mask.shape[0]} values, expected {duration}")
        dq = _bitmask_channels(dq_mask, cast(h5py.Dataset, simple["DQShortnames"])[...])

        inj = cast(h5py.Group, f["quality/injections"])
        injections = _bitmask_channels(
            cast(h5py.Dataset, inj["Injmask"])[...], cast(h5py.Dataset, inj["InjShortnames"])[...]
        )

    return GwoscFile(
        path=path,
        ifo=ifo,
        gps_start=gps_start,
        duration=duration,
        fs=fs,
        strain_channel=strain_channel,
        dq=dq,
        injections=injections,
        strain=strain,
    )


def read_meta(filename: str | Path) -> dict[str, str | int]:
    """Return the ``meta`` group of a GWOSC file as a plain dict.

    Keys are the GWOSC names (``Detector``, ``GPSstart``, ``Duration``,
    ``StrainChannel``, ...); byte strings are decoded to ``str``.
    """
    out: dict[str, str | int] = {}
    with h5py.File(filename, "r") as f:
        meta = cast(h5py.Group, f["meta"])
        for key in meta:
            value = cast(h5py.Dataset, meta[key])[()]
            if isinstance(value, bytes):
                out[key] = value.decode()
            elif isinstance(value, np.integer):
                out[key] = int(value)
            else:
                out[key] = str(value)
    return out


def loaddata(
    filename: str,
    *,
    tvec: bool = True,
    readstrain: bool = True,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | dict[str, float],
    dict[str, npt.NDArray[np.int32]],
]:
    """Load LIGO data from HDF5 files.

    The input filename should be a LOSC .hdf5 file.

    The return value is:
    STRAIN, TIME, CHANNEL_DICT

    STRAIN is a vector of strain values
    TIME is a vector of time values to match the STRAIN vector
         unless the flag tvec=False. In that case, TIME is a
         dictionary of meta values.
    CHANNEL_DICT is a dictionary of data quality channels

    Raises
    ------
    ValueError
        If the file is empty or is not an HDF5 file.
    """
    # -- Check for zero length file
    filepath = Path(filename)
    if filepath.stat().st_size == 0:
        msg = f"File is empty: {filename}"
        raise ValueError(msg)

    # -- Verify it's an HDF5 file
    if filepath.suffix.upper() != ".HDF5":
        msg = f"Only HDF5 files (.hdf5) are supported. Got: {filepath.suffix}"
        raise ValueError(msg)

    # -- Load HDF5 data
    (
        strain,
        gps_start,
        ts,
        qmask,
        shortname_list,
        injmask,
        injname_list,
    ) = read_hdf5(
        filename,
        readstrain=readstrain,
    )

    # -- Create the time vector
    gps_end = gps_start + len(qmask)
    if tvec:
        time = np.arange(gps_start, gps_end, ts)
    else:
        meta = {
            "start": gps_start,
            "stop": gps_end,
            "dt": ts,
        }
        time = meta

    # -- Create 1 Hz DQ channel for each DQ and INJ channel
    channel_dict: dict[str, npt.NDArray[np.int32]] = {}

    for i, flag_name in enumerate(shortname_list):
        # Special check for python 3
        if isinstance(flag_name, bytes):
            decoded_flag = flag_name.decode("utf-8")
        else:
            decoded_flag = flag_name

        channel_dict[decoded_flag] = (qmask >> i) & 1

    for i, flag_name in enumerate(injname_list):
        # Special check for python 3
        if isinstance(flag_name, bytes):
            decoded_flag = flag_name.decode("utf-8")
        else:
            decoded_flag = flag_name

        channel_dict[decoded_flag] = (injmask >> i) & 1

    # -- Calculate the DEFAULT channel
    try:
        channel_dict["DEFAULT"] = channel_dict["DATA"]
    except KeyError:
        logger.warning("Failed to calculate DEFAULT data quality channel")

    return strain, time, channel_dict
