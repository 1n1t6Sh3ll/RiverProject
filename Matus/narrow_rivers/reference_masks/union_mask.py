"""
Union of the RF river mask and a second water source, for one date.

The union is the BASE layer for hand overrides in QGIS: it forces in the small
channels the RF misses. Pixel values say which source called it water, so the
variants can be compared by eye:

  0  land          neither source says water
  1  RF only       RF river, second source dry
  2  source only   second source wet, RF not river   <- what the union adds
  3  both          RF river and second source wet
  9  unknown       RF below its confidence gate and second source dry / no data

Union (water) = values 1, 2, 3.

Second sources (--source):
  s2   same-day Sentinel-2 at 10 m: a Landsat cell is wet when at least K of its
       3 x 3 Sentinel cells have NDWI > T. One variant per (T, K).
       Cells with no Sentinel data count as dry (the RF still decides there).
  jrc  JRC occurrence >= cutoff %. Tried first (2026-11-03) and rejected: a long-
       term average, so it fattens the channel over bars/islands that are dry on
       the day and still misses narrow channels (mixed pixels keep their
       occurrence low).

Inputs (read only):
  patches/<patch>/rf/<date>_rf_class.tif          1 river, 4 land, 9 unknown
  patches/<patch>/<date>_s2_ndwi_10m.tif          Sentinel-2 NDWI, NaN = no data
  patches/<patch>/<date>_water_occurrence.tif     JRC occurrence 0-100 %, 255 no data
  reference_masks/<date>/<date>_nlcd2016_water.tif  (optional, summary only)

Outputs (in reference_masks/<date>/):
  <date>_union_s2_ndwi<T>_k<K>.tif   or   <date>_union_jrc<NN>.tif
  matching .qml (colours by source; default style)
  union_water.qml              alternative style: all water one colour
  runs/<timestamp>/config.json + summary.csv   record of this run

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\union_mask.py --date 2024-06-22
    .venv\\Scripts\\python.exe narrow_rivers\\reference_masks\\union_mask.py --date 2024-06-22 --source jrc
"""

import os, csv, json, argparse, datetime
import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(os.path.dirname(HERE), "patches")
RF_RIVER, RF_UNKNOWN = 1, 9
JRC_NODATA = 255


def qml(entries):
    rows = "\n".join(
        f'        <paletteEntry value="{v}" color="{c}" alpha="{a}" label="{lab}"/>'
        for v, c, a, lab in entries)
    return f"""<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{rows}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


def sources_qml(name):
    return qml([(0, "#000000", 0, "0 land"), (1, "#2ca02c", 255, "1 RF only"),
                (2, "#ff7f0e", 255, f"2 {name} only (added by union)"),
                (3, "#1f5fbf", 255, "3 both"), (9, "#9e9e9e", 120, "9 unknown")])


WATER_QML = qml([(0, "#000000", 0, "land"), (1, "#1f5fbf", 255, "water"),
                 (2, "#1f5fbf", 255, "water"), (3, "#1f5fbf", 255, "water"),
                 (9, "#9e9e9e", 120, "unknown")])


def pixel_area_km2(transform, H):
    """Area of one cell per row (EPSG:4326 cells shrink east-west with latitude)."""
    lat = transform.f + (np.arange(H) + 0.5) * transform.e
    dx = abs(transform.a) * 111_320.0 * np.cos(np.radians(lat))
    dy = abs(transform.e) * 110_574.0
    return (dx * dy / 1e6)[:, None]


# ── second sources: each yields (tag, label, wet mask, params) ───────────────

def s2_variants(patch, date, tf, H, W, thresholds, min_wets):
    path = os.path.join(patch, f"{date}_s2_ndwi_10m.tif")
    with rasterio.open(path) as s:
        st = s.transform
        assert abs(st.a * 3 - tf.a) < 1e-12 and abs(st.c - tf.c) < 1e-9 \
            and abs(st.f - tf.f) < 1e-9, "Sentinel grid does not nest 3x3 in Landsat"
        ndwi = s.read(1)[:3 * H, :3 * W]
    valid = np.isfinite(ndwi)
    n_valid = valid.reshape(H, 3, W, 3).sum((1, 3))
    print(f"  Sentinel-2 NDWI: {int((n_valid == 0).sum()):,} Landsat cells without "
          f"Sentinel data (counted dry), {int(((n_valid > 0) & (n_valid < 9)).sum()):,} "
          f"partly covered")
    for t in thresholds:
        wet9 = ((ndwi > t) & valid).reshape(H, 3, W, 3).sum((1, 3))
        for k in min_wets:
            tag = f"s2_ndwi{'m' if t < 0 else ''}{abs(t):.2f}".replace(".", "") + f"_k{k}"
            yield tag, "Sentinel", wet9 >= k, {"ndwi_gt": t, "min_wet_of_9": k}, path


def jrc_variants(patch, date, cutoffs):
    path = os.path.join(patch, f"{date}_water_occurrence.tif")
    with rasterio.open(path) as s:
        occ = s.read(1)
    for c in cutoffs:
        yield f"jrc{c:02d}", "JRC", (occ != JRC_NODATA) & (occ >= c), \
            {"jrc_ge": c}, path


# ── main ─────────────────────────────────────────────────────────────────────

def main(date, source, cutoffs, thresholds, min_wets):
    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    rf_path = os.path.join(patch, "rf", f"{date}_rf_class.tif")
    out_dir = os.path.join(HERE, date)
    nlcd_path = os.path.join(out_dir, f"{date}_nlcd2016_water.tif")
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(rf_path) as s:
        rf, profile, tf = s.read(1), s.profile, s.transform
    H, W = rf.shape
    nlcd = None
    if os.path.exists(nlcd_path):
        with rasterio.open(nlcd_path) as s:
            assert s.transform == tf, "NLCD grid differs"
            nlcd = s.read(1).astype(bool)

    area = np.broadcast_to(pixel_area_km2(tf, H), (H, W))
    rf_water = rf == RF_RIVER
    rf_unknown = rf == RF_UNKNOWN
    print(f"Union mask — {date}, RF + {source}: RF river {area[rf_water].sum():.2f} km², "
          f"RF unknown {area[rf_unknown].sum():.2f} km²")

    if source == "s2":
        variants = s2_variants(patch, date, tf, H, W, thresholds, min_wets)
    else:
        variants = jrc_variants(patch, date, cutoffs)

    prof = dict(profile, dtype="uint8", count=1, nodata=None, compress="deflate")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, "runs", stamp)
    os.makedirs(run_dir)
    rows, src_path = [], None
    for tag, name, wet, params, src_path in variants:
        u = np.zeros((H, W), np.uint8)
        u[rf_unknown] = 9
        u[rf_water & ~wet] = 1
        u[wet & ~rf_water] = 2          # overrides 9: source says water where RF unsure
        u[rf_water & wet] = 3
        base = os.path.join(out_dir, f"{date}_union_{tag}")
        with rasterio.open(base + ".tif", "w", **prof) as d:
            d.write(u, 1)
        open(base + ".qml", "w", encoding="utf-8").write(sources_qml(name))

        water = (u > 0) & (u != 9)
        row = {"variant": tag, **params, "union_km2": area[water].sum(),
               "rf_only_km2": area[u == 1].sum(), "source_only_km2": area[u == 2].sum(),
               "both_km2": area[u == 3].sum(), "unknown_km2": area[u == 9].sum(),
               "source_only_where_rf_unsure_km2": area[(u == 2) & rf_unknown].sum()}
        if nlcd is not None:
            row["union_in_nlcd_pct"] = 100 * area[water & nlcd].sum() / area[water].sum()
            row["nlcd_in_union_pct"] = 100 * area[water & nlcd].sum() / area[nlcd].sum()
        rows.append(row)
        print(f"  {tag:18} union {row['union_km2']:6.2f} km²  (both {row['both_km2']:.2f}, "
              f"RF only {row['rf_only_km2']:.2f}, {name} only {row['source_only_km2']:.2f})")
    open(os.path.join(out_dir, "union_water.qml"), "w", encoding="utf-8").write(WATER_QML)

    with open(os.path.join(run_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in r.items()} for r in rows)
    json.dump({"script": "union_mask.py", "date": date, "source": source,
               "variants": [r["variant"] for r in rows],
               "rf_class": os.path.relpath(rf_path, HERE),
               "second_source": os.path.relpath(src_path, HERE),
               "nlcd_water": os.path.relpath(nlcd_path, HERE) if nlcd is not None else None,
               "rule": "water = RF river OR second source wet"},
              open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"), indent=1)
    print(f"  run recorded in runs/{stamp}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Union of RF river and a second water source.")
    p.add_argument("--date", required=True)
    p.add_argument("--source", choices=["s2", "jrc"], default="s2")
    p.add_argument("--ndwi", type=float, nargs="+", default=[0.0, 0.1],
                   help="s2: Sentinel NDWI thresholds T")
    p.add_argument("--min-wet", type=int, nargs="+", default=[1, 3],
                   help="s2: Sentinel cells out of 9 that must be wet (K)")
    p.add_argument("--cutoffs", type=int, nargs="+", default=[25, 50, 75],
                   help="jrc: occurrence cutoffs in %%")
    a = p.parse_args()
    main(a.date, a.source, a.cutoffs, a.ndwi, a.min_wet)
