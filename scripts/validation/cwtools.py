"""Known-template coherent F-statistic on a readLIGO band product (validation only).

For a CW with known sky position, frequency evolution and (optionally) a
circular binary orbit, the band product holds (amplitude preserving, global
epoch t0, heterodyned at fmin)::

    x_n = F+(t_n) A+ cos Psi_n + Fx(t_n) Ax sin Psi_n + noise,
    Psi_n = Phi(t_n) - 2 pi fmin (t_n - t0)

with Phi the GW phase at the detector. The JKS F-statistic over the samples
where the product's mask is True, each weighted by its segment's noise
variance ``Sx_seg * B``, is

    2F = s^T M^-1 s,  s_mu = sum x_n h_mu,n / sigma_n^2,  M = sum h h^T / sigma^2,

with ``h = (a cos psi, b cos psi, a sin psi, b sin psi)`` and a, b the
antenna functions at psi_pol = 0. Its expectation is ``4 + A^T M A``.

The SSB timing uses LAL (GetDetectorStates, GetSSBtimes, DE405) on a 60 s grid,
interpolated with cubic splines (error << 1e-9 s).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path

import lal
import lalpulsar
import numpy as np
from scipy.interpolate import CubicSpline

getcontext().prec = 50
HERE = Path(__file__).resolve().parent
TWO_PI = 2 * np.pi


def load_injections(path: Path = HERE / "hwinj_o3a.json") -> dict[str, dict]:
    return {d["name"]: d for d in json.loads(Path(path).read_text())["injections"]}


def load_ephemeris():
    try:
        import solar_system_ephemerides as sse

        base = Path(sse.__file__).parent / "ephemerides"
        e, s = base / "earth" / "earth00-40-DE405.dat.gz", base / "sun" / "sun00-40-DE405.dat.gz"
        if e.exists() and s.exists():
            return lalpulsar.InitBarycenter(str(e), str(s))
    except ImportError:
        pass
    return lalpulsar.InitBarycenter("earth00-40-DE405.dat.gz", "sun00-40-DE405.dat.gz")


def _timestamps(gps: np.ndarray):
    tsv = lalpulsar.CreateTimestampVector(len(gps))
    for i, t in enumerate(gps):
        tsv.data[i] = lal.LIGOTimeGPS(int(t))
    return tsv


@dataclass
class Timing:
    """Splines of the source-frame delay and antenna functions over a product's span."""

    t0: int
    delay: CubicSpline  # (tau_src - t0) - (t - t0), seconds
    a: CubicSpline
    b: CubicSpline


def timing(inj: dict, ifo: str, t_lo: float, t_hi: float, t0: int, edat, step: int = 60) -> Timing:
    det = lal.cached_detector_by_prefix[ifo]
    grid = np.arange(math.floor(t_lo) - 4 * step, math.ceil(t_hi) + 4 * step, step, dtype=np.int64)
    states = lalpulsar.GetDetectorStates(_timestamps(grid), det, edat, 0.0)
    sky = lal.SkyPosition()
    sky.longitude, sky.latitude, sky.system = inj["ra_rad"], inj["dec_rad"], lal.COORDINATESYSTEM_EQUATORIAL
    ssb = lalpulsar.GetSSBtimes(states, sky, lal.LIGOTimeGPS(int(t0)), lalpulsar.SSBPREC_RELATIVISTICOPT)
    dtau = np.array(ssb.DeltaT.data, dtype=np.float64)  # tau_ssb - t0
    b_par = inj.get("binary")
    if b_par:
        if b_par["ecc"] != 0.0:
            raise NotImplementedError("circular orbits only")
        om, asini = TWO_PI / b_par["porb_s"], b_par["asini_lts"]
        tau = dtau  # relative to t0
        tp = b_par["tp_ssb_gps"] - t0
        src = tau.copy()
        for _ in range(12):
            src = tau - asini * np.sin(om * (src - tp) + b_par["argp_rad"])
        dtau = src
    delay = dtau - (grid - t0).astype(np.float64)
    a = np.empty(grid.size)
    b = np.empty(grid.size)
    for i, t in enumerate(grid):
        gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(int(t)))
        a[i], b[i] = lal.ComputeDetAMResponse(det.response, inj["ra_rad"], inj["dec_rad"], 0.0, gmst)
    x = (grid - t0).astype(np.float64)
    return Timing(t0=t0, delay=CubicSpline(x, delay), a=CubicSpline(x, a), b=CubicSpline(x, b))


def phase_at_t0(inj: dict, t0: int) -> tuple[float, float, float]:
    """(phi at SSB/source time t0 [rad, mod 2 pi], f at t0, f1dot), from the injection's ref-time values."""
    f0, f1 = Decimal(repr(inj["f0_gw_Hz"])), Decimal(repr(inj["f1dot_gw_Hz_s"]))
    dt = Decimal(t0) - Decimal(inj["ref_time_gps"])
    cycles = f0 * dt + f1 * dt * dt / 2
    frac = float(cycles - cycles.to_integral_value(rounding="ROUND_FLOOR"))
    phi = (inj["phi0_gw_rad"] + TWO_PI * frac) % TWO_PI
    return phi, float(f0 + f1 * dt), float(f1)


def jks_amplitudes(inj: dict, phi: float, gain: float = 1.0) -> np.ndarray:
    h0 = inj["h0"] * gain
    ap, ax = h0 * (1 + inj["cosi"] ** 2) / 2, h0 * inj["cosi"]
    c2, s2 = math.cos(2 * inj["psi_rad"]), math.sin(2 * inj["psi_rad"])
    cp, sp = math.cos(phi), math.sin(phi)
    return np.array([
        ap * c2 * cp - ax * s2 * sp,
        ap * s2 * cp + ax * c2 * sp,
        -ap * c2 * sp - ax * s2 * cp,
        -ap * s2 * sp + ax * c2 * cp,
    ])


@dataclass
class FstatResult:
    df: np.ndarray  # frequency offsets (Hz)
    twoF: np.ndarray
    expected_2F: float  # 4 + A^T M A at df = 0
    amplitude_ratio: float  # |fitted| / |injected| (1 = as injected)
    phase_offset: float  # rad (0 = as injected)
    n_samples: int


def fstat(product, inj: dict, edat, df: np.ndarray | None = None, *, gain: float = 1.0) -> FstatResult:
    """2F at frequency offsets *df* (default: 0 only) from the injection's frequency track."""
    df = np.atleast_1d(np.zeros(1) if df is None else np.asarray(df, dtype=np.float64))
    t0 = int(product.epoch)
    fs = Fraction(product.attrs.get("band_exact", str(product.band))) * 2
    fmin = Fraction(product.attrs.get("fmin_exact", str(product.fmin)))
    n = product.n0 + np.arange(product.data.size, dtype=np.int64)
    sel = product.mask & (product.window > 0)
    n = n[sel]
    x = product.data[sel]
    t_rel = n / float(fs)  # t_n - t0
    tm = timing(inj, product.ifo, t0 + t_rel.min(), t0 + t_rel.max(), t0, edat)
    delta = t_rel + tm.delay(t_rel)  # tau_src - t0
    a, b = tm.a(t_rel), tm.b(t_rel)
    phi, f_loc, f1 = phase_at_t0(inj, t0)
    # heterodyne: fmin (t_n - t0) = fmin n / fs, exactly mod 1
    r = fmin / fs
    het = ((r.numerator * (n % r.denominator)) % r.denominator) / r.denominator
    base = np.mod(f_loc * delta, 1.0) + np.mod(0.5 * f1 * delta * delta, 1.0) - het
    # noise per sample from the per-segment Sx
    seg = product.tables["noise_segments"]
    sx = np.where(np.isfinite(seg["sx"]), seg["sx"], product.sqrt_sx**2)
    k = np.clip(np.searchsorted(seg["gps_start"], t0 + t_rel, side="right") - 1, 0, sx.size - 1)
    w = 1.0 / (sx[k] * product.band)
    out = np.empty(df.size)
    for i, d in enumerate(df):
        psi = TWO_PI * (base + d * delta)
        c, s = np.cos(psi), np.sin(psi)
        H = np.stack([a * c, b * c, a * s, b * s])
        Hw = H * w
        M = Hw @ H.T
        sv = Hw @ x
        out[i] = sv @ np.linalg.solve(M, sv)
        if d == 0:
            A = jks_amplitudes(inj, phi, gain)
            expected = 4.0 + A @ M @ A
            Ahat = np.linalg.solve(M, sv)
            za, zb = complex(A[0], -A[2]), complex(A[1], -A[3])
            zha, zhb = complex(Ahat[0], -Ahat[2]), complex(Ahat[1], -Ahat[3])
            wa, wb = M[0, 0], M[1, 1]
            cfit = (wa * zha * za.conjugate() + wb * zhb * zb.conjugate()) / (wa * abs(za) ** 2 + wb * abs(zb) ** 2)
    if not np.any(df == 0):
        expected, cfit = float("nan"), complex("nan")
    return FstatResult(df=df, twoF=out, expected_2F=float(expected), amplitude_ratio=abs(cfit),
                       phase_offset=float(np.angle(cfit)), n_samples=int(x.size))
