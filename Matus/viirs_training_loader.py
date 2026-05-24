"""
VIIRS training data loader.
Loads geolocation from a standalone GITCO file and I-bands from a
separate GIMGO-SVI file, verifies they are the same granule (same
date, time range, orbit), and returns clipped arrays ready for
reprojection.

Usage:
    from viirs_training_loader import load_viirs_training_pair
    lat, lon, bands, angles = load_viirs_training_pair(
        gitco_path, gimgo_path, clip_bbox=(lat_min, lat_max, lon_min, lon_max)
    )
"""

import os, re
import h5py
import numpy as np


# ── filename parsing ──────────────────────────────────────────────────────────

def _parse_viirs_filename(path):
    """Extract date, start time, end time, orbit from VIIRS filename."""
    fname = os.path.basename(path)
    m = re.search(
        r'd(\d{8})_t(\d{7})_e(\d{7})_b(\d+)', fname)
    if not m:
        raise ValueError(f"Cannot parse VIIRS filename: {fname}")
    return {
        "date":   m.group(1),
        "t_start": m.group(2),
        "t_end":   m.group(3),
        "orbit":   m.group(4),
    }


def _verify_matched_pair(gitco_path, gimgo_path):
    """Assert both files are the same granule. Raises ValueError if not."""
    g = _parse_viirs_filename(gitco_path)
    s = _parse_viirs_filename(gimgo_path)
    mismatches = []
    for key in ("date", "t_start", "t_end", "orbit"):
        if g[key] != s[key]:
            mismatches.append(f"{key}: GITCO={g[key]}, GIMGO={s[key]}")
    if mismatches:
        raise ValueError(
            "GITCO and GIMGO-SVI are NOT matched granules:\n  " +
            "\n  ".join(mismatches)
        )
    print(f"[verify] Matched pair confirmed:")
    print(f"  Date:   {g['date'][:4]}-{g['date'][4:6]}-{g['date'][6:]}")
    print(f"  Start:  {g['t_start'][:2]}:{g['t_start'][2:4]}:{g['t_start'][4:6]} UTC")
    print(f"  End:    {g['t_end'][:2]}:{g['t_end'][2:4]}:{g['t_end'][4:6]} UTC")
    print(f"  Orbit:  b{g['orbit']}")
    return g


# ── dataset discovery ────────────────────────────────────────────────────────

def _list_datasets(h5, prefix=""):
    paths = []
    for key in h5.keys():
        p = f"{prefix}/{key}" if prefix else key
        if isinstance(h5[key], h5py.Dataset):
            paths.append(p)
        else:
            paths.extend(_list_datasets(h5[key], p))
    return paths


def _find_dataset(h5, *keywords):
    """Return first dataset path whose name contains ALL keywords."""
    all_paths = _list_datasets(h5)
    for path in all_paths:
        if all(kw.lower() in path.lower() for kw in keywords):
            return path
    return None


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_geolocation(gitco_path):
    """Load lat/lon from GITCO file."""
    with h5py.File(gitco_path, "r") as h5:
        lat_path = _find_dataset(h5, "Latitude")
        lon_path = _find_dataset(h5, "Longitude")
        if not lat_path or not lon_path:
            raise KeyError(f"Lat/Lon not found in GITCO: {gitco_path}")
        lat = h5[lat_path][...].astype(np.float32)
        lon = h5[lon_path][...].astype(np.float32)
    lat[(lat < -90)  | (lat > 90)]   = np.nan
    lon[(lon < -180) | (lon > 180)]  = np.nan
    print(f"[geo] Lat {np.nanmin(lat):.3f}–{np.nanmax(lat):.3f}  "
          f"Lon {np.nanmin(lon):.3f}–{np.nanmax(lon):.3f}  "
          f"shape {lat.shape}")
    return lat, lon


def _load_bands(gimgo_path):
    """Load I1–I5 reflectance/BT from GIMGO-SVI file."""
    bands = {}
    with h5py.File(gimgo_path, "r") as h5:
        all_paths = _list_datasets(h5)
        for i in range(1, 6):
            token   = f"VIIRS-I{i}-SDR_All"
            targets = ["Reflectance"] if i <= 3 else ["BrightnessTemperature"]
            for target in targets:
                cands = [p for p in all_paths
                         if token in p and p.endswith(target)]
                if cands:
                    raw = h5[cands[0]][...].astype(np.float32)
                    # find scale factor
                    factor_path = cands[0] + "Factors"
                    if factor_path in h5:
                        fac = h5[factor_path][...].ravel()
                        scale  = float(fac[0]) if len(fac) >= 1 else 1.0
                        offset = float(fac[1]) if len(fac) >= 2 else 0.0
                    else:
                        scale, offset = 1.0, 0.0
                    arr = raw * scale + offset
                    arr[raw > 65529] = np.nan  # fill/invalid
                    bands[f"I{i}"] = arr
                    print(f"[bands] I{i} from {cands[0]}")
                    break
            if f"I{i}" not in bands:
                raise KeyError(f"Band I{i} not found in: {gimgo_path}")
    return bands


def _load_angles(gitco_path, r0, r1, c0, c1):
    """Load solar and satellite angle arrays from GITCO, clipped."""
    angle_map = {
        "SZA": ["SolarZenithAngle",     "SZA"],
        "SAA": ["SolarAzimuthAngle",    "SAA"],
        "VZA": ["SatelliteZenithAngle", "SatZenithAngle",  "VZA"],
        "VAA": ["SatelliteAzimuthAngle","SatAzimuthAngle", "VAA"],
    }
    angles = {}
    with h5py.File(gitco_path, "r") as h5:
        all_paths = _list_datasets(h5)
        for key, candidates in angle_map.items():
            found = False
            for cand in candidates:
                matches = [p for p in all_paths
                           if cand.lower() in p.lower()]
                # prefer 2D arrays
                for p in matches:
                    arr = h5[p][...].astype(np.float32)
                    if arr.ndim == 2:
                        arr[arr > 1e6] = np.nan
                        angles[key] = arr[r0:r1+1, c0:c1+1]
                        print(f"[angles] {key} from {p}")
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"[angles] {key}: NOT FOUND — filling NaN")
                shape = (r1 - r0 + 1, c1 - c0 + 1)
                angles[key] = np.full(shape, np.nan, dtype=np.float32)
    return angles


# ── main entry point ─────────────────────────────────────────────────────────

def load_viirs_training_pair(gitco_path, gimgo_path,
                              clip_bbox=None):
    """
    Load and verify a matched GITCO + GIMGO-SVI pair.

    Parameters
    ----------
    gitco_path : str   Path to GITCO h5 file.
    gimgo_path : str   Path to GIMGO-SVI h5 file.
    clip_bbox  : tuple (lat_min, lat_max, lon_min, lon_max) or None.
                 If given, clips swath to this bounding box.

    Returns
    -------
    lat, lon : np.ndarray  Geolocation arrays (clipped if bbox given).
    bands    : dict        Keys I1–I5, clipped.
    angles   : dict        Keys SZA, SAA, VZA, VAA, clipped.
    meta     : dict        date, orbit, etc.
    """
    meta = _verify_matched_pair(gitco_path, gimgo_path)

    lat_full, lon_full = _load_geolocation(gitco_path)
    bands_full         = _load_bands(gimgo_path)

    # clip to bbox
    if clip_bbox is not None:
        lat_min, lat_max, lon_min, lon_max = clip_bbox
        mask = ((lat_full >= lat_min) & (lat_full <= lat_max) &
                (lon_full >= lon_min) & (lon_full <= lon_max))
        rows, cols = np.where(mask)
        if rows.size == 0:
            raise ValueError(
                f"No VIIRS pixels within bbox {clip_bbox}. "
                "Check file coverage.")
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
    else:
        r0, r1 = 0, lat_full.shape[0] - 1
        c0, c1 = 0, lat_full.shape[1] - 1

    lat    = lat_full[r0:r1+1, c0:c1+1]
    lon    = lon_full[r0:r1+1, c0:c1+1]
    bands  = {k: v[r0:r1+1, c0:c1+1] for k, v in bands_full.items()}
    angles = _load_angles(gitco_path, r0, r1, c0, c1)

    print(f"[clip] Clipped shape: {lat.shape}")
    print(f"[clip] Lat {np.nanmin(lat):.3f}–{np.nanmax(lat):.3f}  "
          f"Lon {np.nanmin(lon):.3f}–{np.nanmax(lon):.3f}")

    return lat, lon, bands, angles, meta
