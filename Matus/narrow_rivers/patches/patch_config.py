"""
Shared definitions for the Nenana patch work: the study box, the dates, the
grids and the class codes.

GRIDS (decided with the professor, 2026-09-29): everything is EPSG:4326 on the
main pipeline's Alaska grid, so our cells coincide with the existing dataset's
cells. Pixels are rectangular on the ground at 64 N (~13 x 30 m for Landsat,
~163 x 374 m for VIIRS); that is accepted.

  Alaska grid : lat 54..72, lon -171..-129, 12432 x 5328 cells of 375/111000 deg
  VIIRS cell  : SHARED_RES        = 0.00337838 deg
  Landsat cell: SHARED_RES / 12.5 = 0.00027027 deg  (exact nesting)

Labels are vectors, so they can be re-rasterised onto any other grid later if
the team changes its mind.
"""

import math

# ── Alaska domain (from Matus/extract_training_pixels.py — do not redefine) ──
ALASKA_LAT_MIN, ALASKA_LAT_MAX = 54.0, 72.0
ALASKA_LON_MIN, ALASKA_LON_MAX = -171.0, -129.0
ALASKA_W, ALASKA_H = 12432, 5328

RESOLUTION_M = 375.0
SHARED_RES   = RESOLUTION_M / 111_000.0      # VIIRS cell, degrees
LS_SUB       = 12.5                          # Landsat cells per VIIRS cell
LS_RES       = SHARED_RES / LS_SUB           # Landsat cell, degrees
CRS          = "EPSG:4326"

# ── Classes ─────────────────────────────────────────────────────────────────
CLASSES = {1: "river", 2: "ice", 3: "snow", 4: "land"}
NODATA_CLASS = 9                             # drawn where Landsat has no data
CLASS_COLOURS = {1: "#0066FF", 2: "#FF00FF", 3: "#00FFFF", 4: "#8B4513",
                 NODATA_CLASS: "#999999"}

# ── Study box (drawn in GEE, 2026-09-29) ────────────────────────────────────
BOX = (-149.3376600178792, 64.093728680575,      # lon_min, lat_min
       -149.0691858059353, 64.53537170844751)    # lon_max, lat_max
RIVER = "Nenana River"

# ── Scenes: 2 summer (water only), then one per season ──────────────────────
SCENES = {
    # NOTE: VIIRS SDRs on the NOAA AWS buckets start ~2022-08 (S-NPP),
    # ~2023-02 (NOAA-21), and NOAA-20 has gaps before 2023 — so only Landsat
    # dates from about Feb 2023 onward can be paired from that source.
    # (NASA's VNP02/VNP03 archive goes back to 2012 if pre-2023 is ever needed.)
    "2024-06-22": {"scene_id": "LC90710152024174LGN00", "season": "summer",
                   "collection": "LANDSAT/LC09/C02/T1_L2", "cloud": 6.9},
    "2023-07-24": {"scene_id": "LC90690152023205LGN00", "season": "summer",
                   "collection": "LANDSAT/LC09/C02/T1_L2", "cloud": 5.6},
    "2024-04-21": {"scene_id": "LC90690152024112LGN00", "season": "breakup",
                   "collection": "LANDSAT/LC09/C02/T1_L2", "cloud": 1.2},
    "2026-03-26": {"scene_id": "LC90690152026085LGN00", "season": "ice",
                   "collection": "LANDSAT/LC09/C02/T1_L2", "cloud": 0.0},
    "2023-10-04": {"scene_id": "LC80690152023277LGN00", "season": "freeze-up",
                   "collection": "LANDSAT/LC08/C02/T1_L2", "cloud": 2.6},
}

SR_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]


def _snap(box, res):
    """Box -> grid aligned to the Alaska grid origin (lon -171, lat 72)."""
    lon0, lat0, lon1, lat1 = box
    c0 = math.floor((lon0 - ALASKA_LON_MIN) / res)
    c1 = math.ceil((lon1 - ALASKA_LON_MIN) / res)
    r0 = math.floor((ALASKA_LAT_MAX - lat1) / res)
    r1 = math.ceil((ALASKA_LAT_MAX - lat0) / res)
    return {"x0": ALASKA_LON_MIN + c0 * res, "y1": ALASKA_LAT_MAX - r0 * res,
            "W": c1 - c0, "H": r1 - r0, "res": res,
            "col0": c0, "row0": r0, "crs": CRS}


def landsat_grid(box=BOX):
    """30 m-ish Landsat grid for the box, on the Alaska grid."""
    return _snap(box, LS_RES)


def viirs_grid(box=BOX):
    """375 m VIIRS grid for the box, on the Alaska grid (cells match the
    main pipeline's shared grid exactly)."""
    return _snap(box, SHARED_RES)


def describe(g, label=""):
    lat = g["y1"] - g["H"] * g["res"] / 2
    ew = g["res"] * 111.320 * math.cos(math.radians(lat)) * 1000
    ns = g["res"] * 110.574 * 1000
    return (f"{label}{g['W']} x {g['H']} cells of {g['res']:.8f} deg "
            f"(~{ew:.0f} x {ns:.0f} m at {lat:.1f} N), "
            f"Alaska grid col {g['col0']}, row {g['row0']}")
