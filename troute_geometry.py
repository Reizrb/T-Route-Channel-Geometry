"""
Update the channel geometry of the NWM / T-Route RouteLink file.

T-Route's Muskingum-Cunge routing uses a trapezoidal channel for every reach, defined
in the RouteLink NetCDF file by its bottom width (BtmWdth), top width (TopWdth), and
side slope (ChSlp). This code replaces those with ML-predicted bankfull geometry from
HydroGeom (Zarrabi et al., 2025) for a channel shape you choose, then recomputes the
Muskingum parameters MusK and MusX and the compound-channel top width TopWdthCC.

Channel shape: a trapezoid with bottom width = a x top width
    a = 0      triangular
    0 < a < 1  trapezoidal
    a -> 1     rectangular (use a = 0.99; a = 1 gives a zero side slope that the
               T-Route Muskingum-Cunge kernel cannot represent)

Use:
    import troute_geometry as tg
    geometry = tg.channel_geometry(huc2="11")            # or tg.read_geometry_file(path)
    tg.update_routelink("RouteLink_CONUS.nc", "RouteLink_CONUS_a0.4.nc", geometry, a=0.4)

Requires numpy, pandas, netCDF4, and (for channel_geometry) HydroGeomKit:
    pip install "git+https://github.com/Reizrb/HydroGeom"
"""

import datetime
import os
import shutil

import numpy as np
import pandas as pd

GEOMETRY_VARS = ("BtmWdth", "TopWdth", "ChSlp", "TopWdthCC", "MusK", "MusX")


# ----------------------------------------------------------------- geometry
def channel_geometry(huc2=None, state=None, huc8=None, comids=None):
    """Bankfull width and depth for one area from HydroGeom (COMID, bnk_width, bnk_depth).

    Choose one: huc2 (e.g. "11"), state (e.g. "AR"), huc8, or a list of comids.
    For all of CONUS, download the dataset from Zenodo
    (https://doi.org/10.5281/zenodo.19208847) and use read_geometry_file().
    """
    try:
        from hydrogeomkit import get_channel_geometry
    except ImportError as e:
        raise ImportError(
            'Needs HydroGeomKit: pip install "git+https://github.com/Reizrb/HydroGeom"'
        ) from e
    df = get_channel_geometry(huc2=huc2, state=state, huc8=huc8, comids=comids)
    return _clean(df.rename(columns={"comid": "COMID"}))


def read_geometry_file(path):
    """Read the HydroGeom / Zenodo reach geometry table (COMID, bnk_width, bnk_depth)."""
    return _clean(pd.read_csv(path, usecols=["COMID", "bnk_width", "bnk_depth"]))


def _clean(df):
    df = df[["COMID", "bnk_width", "bnk_depth"]].copy()
    for c in ("bnk_width", "bnk_depth"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = np.isfinite(df["bnk_width"]) & np.isfinite(df["bnk_depth"])
    ok &= (df["bnk_width"] > 0) & (df["bnk_depth"] > 0)
    df = df[ok].astype({"COMID": np.int64})
    if df["COMID"].duplicated().any():
        raise ValueError("Geometry table has duplicate COMIDs.")
    return df.reset_index(drop=True)


def trapezoid(top_width, depth, a):
    """Bottom width, top width, and T-Route side slope ChSlp for bottom/top width ratio a.

    The H:V side slope is m = (1 - a) * top_width / (2 * depth); T-Route stores its
    reciprocal (V:H), ChSlp = 1 / m.
    """
    if not 0 <= a < 1:
        raise ValueError(
            "a must be at least 0 and less than 1. For a rectangular channel use "
            "a = 0.99: a = 1 gives a zero side slope that the T-Route "
            "Muskingum-Cunge kernel cannot represent."
        )
    top = np.asarray(top_width, dtype=np.float64)
    m = (1.0 - a) * top / (2.0 * np.asarray(depth, dtype=np.float64))
    chslp = 1.0 / m
    if not np.isfinite(chslp.astype(np.float32)).all():
        raise ValueError("Computed ChSlp is not representable as float32.")
    return a * top, top, chslp


# ---------------------------------------------------- Muskingum parameters
def compute_ck(b, B, z, n, S0, d=None):
    """Kinematic wave speed ck from trapezoidal channel geometry.

    z is the H:V side slope, NOT the stored ChSlp (which is 1/z). When d is omitted,
    use the bankfull-depth branches in the MC Fortran kernel, including its special
    treatment of b == B and b > B.
    """
    if d is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(
                b > B, b / 1e-5, np.where(b == B, b / (2.0 * z), (B - b) / (2.0 * z))
            )
        d = np.maximum(d, 0.01)

    A = b * d + z * d**2
    P = b + 2 * d * np.sqrt(1 + z**2)
    R = np.where(P > 0, A / P, 0.0)
    Q = (1.0 / n) * S0**0.5 * A * R ** (2 / 3)  # Manning's Q at bankfull

    sqr = np.sqrt(1 + z**2)  # ck = dQ/dA
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (5 / 3) / np.where(A > 0, A, np.nan)
        term2 = (4 * (sqr - z)) / (
            3 * np.where(d > 0, d, np.nan) * sqr * np.where(P > 0, P, np.nan)
        )
    ck = Q * (term1 - term2)
    ck = np.where(np.isfinite(ck) & (ck > 0), ck, 0.5)  # fallback 0.5 m/s
    return ck, Q, A, B


def compute_musk_x(ck, Q, B, S0, dx, dt=3600.0):
    """Muskingum K (MusK, s) = dx / ck and X (MusX) = 0.5 (1 - Q / (ck B S0 dx)), clipped to [0, 0.5]."""
    K = dx / np.where(ck > 0, ck, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = ck * B * S0 * dx
        X = 0.5 * (1.0 - Q / np.where(denom > 0, denom, np.nan))
    X = np.where(np.isfinite(X), X, 0.2)
    X = np.clip(X, 0.0, 0.5)
    return K.astype(np.float32), X.astype(np.float32)


# ------------------------------------------------------------ RouteLink file
def update_routelink(routelink_in, routelink_out, geometry, a=0.4, verbose=True):
    """Write a copy of the RouteLink file with the new channel geometry.

    routelink_in:  original RouteLink NetCDF (never modified)
    routelink_out: new file to write
    geometry:      table with COMID, bnk_width, bnk_depth (from channel_geometry
                   or read_geometry_file); reaches not in it keep their geometry
    a:             bottom width / top width
    MusK, MusX, and TopWdthCC (= 3 x TopWdth) are recomputed for every reach.
    Returns a dict with the number of updated reaches.
    """
    import netCDF4 as nc

    if os.path.abspath(routelink_in) == os.path.abspath(routelink_out):
        raise ValueError("routelink_out must be a different file from routelink_in.")
    geom = _clean(geometry).set_index("COMID")
    btm, top, chslp = trapezoid(geom["bnk_width"], geom["bnk_depth"], a)

    with nc.Dataset(routelink_in, "r") as ds:
        comids = ds.variables["link"][:].astype(np.int64)
        btm_new = ds.variables["BtmWdth"][:].copy()
        top_new = ds.variables["TopWdth"][:].copy()
        slp_new = ds.variables["ChSlp"][:].copy()
        So = ds.variables["So"][:].copy()
        n = ds.variables["n"][:].copy()
        dx = ds.variables["Length"][:].copy()

    matched = geom.index.get_indexer(pd.Index(comids))
    idx = np.flatnonzero(matched >= 0)
    src = matched[idx]
    btm_new[idx], top_new[idx], slp_new[idx] = btm[src], top[src], chslp[src]

    S0 = np.where(So > 0, So, 1e-5)
    z = np.divide(
        1.0, slp_new, out=np.ones_like(slp_new, dtype=np.float64), where=(slp_new != 0)
    )
    ck, Q, _, _ = compute_ck(b=btm_new, B=top_new, z=z, n=n, S0=S0)
    musk, musx = compute_musk_x(ck=ck, Q=Q, B=top_new, S0=S0, dx=dx)

    shutil.copy2(routelink_in, routelink_out)
    with nc.Dataset(routelink_out, "r+") as ds:
        ds.variables["BtmWdth"][:] = btm_new.astype(np.float32)
        ds.variables["TopWdth"][:] = top_new.astype(np.float32)
        ds.variables["ChSlp"][:] = slp_new.astype(np.float32)
        ds.variables["TopWdthCC"][:] = (3.0 * top_new).astype(np.float32)
        ds.variables["MusK"][:] = musk
        ds.variables["MusX"][:] = musx
        ds.history = (
            getattr(ds, "history", "")
            + f"\nUpdated {datetime.datetime.now().isoformat()}: channel geometry from "
            f"HydroGeom with bottom/top width ratio a = {a}; MusK and MusX recomputed. "
            f"{len(idx):,} reaches modified."
        )

    summary = {
        "updated": len(idx),
        "not_in_routelink": len(geom) - len(idx),
        "unchanged": len(comids) - len(idx),
    }
    if verbose:
        print(
            f"Updated {summary['updated']:,} reaches (a = {a}); "
            f"{summary['unchanged']:,} RouteLink reaches kept their geometry; "
            f"{summary['not_in_routelink']:,} geometry reaches are not in the RouteLink file."
        )
    return summary


def compare(routelink_a, routelink_b, comids=None, variables=GEOMETRY_VARS):
    """Summary statistics of the geometry and Muskingum variables in two RouteLink files.

    comids: optionally limit the comparison to these reaches (e.g. geometry["COMID"]).
    """
    import netCDF4 as nc

    rows = []
    for label, path in (("original", routelink_a), ("updated", routelink_b)):
        with nc.Dataset(path) as ds:
            keep = slice(None)
            if comids is not None:
                links = ds.variables["link"][:].astype(np.int64)
                keep = np.isin(links, np.asarray(comids, dtype=np.int64))
            for v in variables:
                x = np.asarray(ds.variables[v][:], dtype=np.float64)[keep]
                rows.append(
                    {
                        "variable": v,
                        "file": label,
                        "median": np.median(x),
                        "mean": x.mean(),
                        "min": x.min(),
                        "max": x.max(),
                    }
                )
    out = pd.DataFrame(rows).set_index(["variable", "file"]).unstack("file")
    return out.reindex(list(variables)).round(3)
