"""LIGO HDF5 Data Reader.

This module provides functions to read LIGO gravitational wave data
from HDF5 files (GWOSC format). It extracts strain data, data quality
information, injection masks, and metadata.

Only HDF5 files (.hdf5) are supported. For frame files (.gwf),
use the LALSuite tools directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import numpy.typing as npt

# Configure logging
logger = logging.getLogger(__name__)

# Public API
__all__ = ["loaddata", "read_hdf5", "read_meta"]


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
    """Read HDF5 files and extract LIGO data."""
    data_file = h5py.File(filename, "r")

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

    data_file.close()
    return strain, gps_start, ts, qmask, shortname_list, injmask, injname_list


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
