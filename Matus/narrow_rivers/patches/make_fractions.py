"""
Hand-drawn polygons -> per-VIIRS-pixel class fractions.

1. Read the QGIS label shapefile (no fiona/GDAL needed — the polygon shapefile
   format is read directly, so the venv stays as it is).
2. Rasterise the polygons on a grid 50x finer than a VIIRS cell, so each cell
   holds exactly 50 x 50 fine cells and the area weighting is exact (no
   12.5-Landsat-pixels-per-side rounding).
3. Aggregate each VIIRS cell into class fractions.

Grid: EPSG:4326 on the pipeline's Alaska grid — see patch_config.py.

Classes (class_id): 1 river, 2 ice, 3 snow, 4 land, 9 = no data (Landsat has no
values there). 0 = not drawn. Cells are only usable when they are labelled edge
to edge: `labelled` reports the covered share and anything marked 9 is excluded,
so a cell containing no-data can never count as finished.

Outputs (in patches/<patch>/fractions/):
  labels_landsat.tif    class per Landsat cell (display/QC)
  fractions_375m.tif    5 bands: river, ice, snow, land, labelled
  fractions_375m.csv    one row per VIIRS cell
  viirs_grid_375m.geojson  the cell grid for QGIS, carrying the fractions
  fractions_qc.png      labels + fraction maps

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\patches\\make_fractions.py --patch nenana_20240622
"""

import os, sys, csv, json, math, struct, argparse
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_config as C

NR_ROOT  = os.path.dirname(HERE)
CLASSES  = C.CLASSES                      # 1 river, 2 ice, 3 snow, 4 land
NODATA   = C.NODATA_CLASS                 # 9
SUB      = 50                             # fine cells per VIIRS cell, per side
FINE_RES = C.SHARED_RES / SUB             # degrees


# ── shapefile reading (polygons only) ────────────────────────────────────────

def _read_dbf(path):
    """dBase table -> list of dicts (values as str, numbers converted)."""
    with open(path, "rb") as f:
        buf = f.read()
    n_rec, hdr_len, rec_len = struct.unpack("<IHH", buf[4:12])
    fields = []
    for off in range(32, hdr_len - 1, 32):
        if buf[off] == 0x0D:
            break
        name = buf[off:off + 11].split(b"\0")[0].decode("latin-1").strip()
        ftype = chr(buf[off + 11])
        flen = buf[off + 16]
        fields.append((name, ftype, flen))
    rows, pos = [], hdr_len
    for _ in range(n_rec):
        rec, pos = buf[pos:pos + rec_len], pos + rec_len
        if not rec:
            continue
        # Deleted records stay in the file (and in the .shp) until it is packed,
        # so keep them in place and flag them — dropping them would shift the
        # attributes relative to the geometries.
        row, o = {"_deleted": rec[:1] == b"*"}, 1
        for name, ftype, flen in fields:
            raw = rec[o:o + flen].decode("latin-1").strip(); o += flen
            if ftype in "NF" and raw not in ("", "-"):
                try:
                    row[name] = float(raw) if "." in raw else int(raw)
                except ValueError:
                    row[name] = None
            else:
                row[name] = raw or None
        rows.append(row)
    return rows


def _ring_area(pts):
    """Signed area; shapefile outer rings are clockwise (negative here)."""
    x = np.asarray([p[0] for p in pts]); y = np.asarray([p[1] for p in pts])
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _read_shp(path):
    """Polygon shapefile -> list of GeoJSON-like Polygon dicts (one per record).

    Record offsets come from the .shx index when it exists: after edits QGIS can
    leave stale record blocks in the .shp, and only the index says which ones
    are live.
    """
    with open(path, "rb") as f:
        buf = f.read()
    shx_path = path[:-4] + ".shx"
    offsets = None
    if os.path.exists(shx_path):
        with open(shx_path, "rb") as f:
            shx = f.read()
        offsets = [struct.unpack(">I", shx[100 + i * 8: 104 + i * 8])[0] * 2
                   for i in range((len(shx) - 100) // 8)]
    shapes, pos = [], 100
    for i in range(len(offsets) if offsets is not None else 10 ** 9):
        if offsets is not None:
            pos = offsets[i]
        elif pos >= len(buf):
            break
        _, clen = struct.unpack(">II", buf[pos:pos + 8])
        body = buf[pos + 8: pos + 8 + clen * 2]
        pos += 8 + clen * 2
        (stype,) = struct.unpack("<I", body[:4])
        if stype == 0:                      # null shape
            shapes.append(None); continue
        if stype not in (5, 15, 25):
            raise ValueError(f"shape type {stype} is not a polygon")
        n_parts, n_points = struct.unpack("<II", body[36:44])
        parts = list(struct.unpack(f"<{n_parts}I", body[44:44 + 4 * n_parts]))
        p0 = 44 + 4 * n_parts
        xy = struct.unpack(f"<{2 * n_points}d", body[p0:p0 + 16 * n_points])
        pts = list(zip(xy[0::2], xy[1::2]))
        rings = [pts[parts[i]: (parts[i + 1] if i + 1 < n_parts else n_points)]
                 for i in range(n_parts)]
        # outer ring = clockwise (negative signed area); others are holes
        polys, current = [], None
        for r in rings:
            if _ring_area(r) < 0:
                if current:
                    polys.append(current)
                current = [r]
            elif current:
                current.append(r)
            else:
                current = [r[::-1]]          # lone CCW ring: treat as outer
        if current:
            polys.append(current)
        shapes.append({"type": "MultiPolygon", "coordinates": polys}
                      if len(polys) > 1 else
                      {"type": "Polygon", "coordinates": polys[0]})
    return shapes


def _read_prj(path):
    return open(path, encoding="utf-8").read().strip() if os.path.exists(path) else None


def _patch_date(patch):
    """'nenana_20240622' -> '2024-06-22'."""
    d = patch.split("_")[-1]
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


# ── main ─────────────────────────────────────────────────────────────────────

def main(patch, class_field, min_labelled):
    patch_dir = os.path.join(NR_ROOT, "patches", patch)
    date = _patch_date(patch)
    shp = os.path.join(patch_dir, "labels", f"labels_{date}.shp")
    if not os.path.exists(shp):                       # older naming
        alt = os.path.join(patch_dir, "labels", f"labels_{patch}.shp")
        shp = alt if os.path.exists(alt) else shp
    if not os.path.exists(shp):
        raise SystemExit(f"ERROR: no label shapefile in {os.path.dirname(shp)}")

    lg, vg = C.landsat_grid(), C.viirs_grid()
    nx, ny = vg["W"], vg["H"]
    fine_tf = from_origin(vg["x0"], vg["y1"], FINE_RES, FINE_RES)
    fine_w, fine_h = nx * SUB, ny * SUB
    ls_tf = from_origin(lg["x0"], lg["y1"], lg["res"], lg["res"])

    print("=" * 62)
    print(f"Fractions — {patch}  ({date})")
    print(f"  labels : {shp}")
    print("  " + C.describe(lg, "Landsat: "))
    print("  " + C.describe(vg, "VIIRS  : "))
    print(f"  rasterised at {FINE_RES:.8f} deg ({SUB}x{SUB} per VIIRS cell)")
    print("=" * 62)

    prj = _read_prj(shp[:-4] + ".prj")
    if prj and not any(k in prj.replace(" ", "_") for k in
                       ("4326", "GCS_WGS_1984", "WGS_84", "WGS_1984")):
        print("  WARNING: the .prj does not look like EPSG:4326 — check the "
              "labels were drawn in the patch CRS")

    shapes = _read_shp(shp)
    attrs  = _read_dbf(shp[:-4] + ".dbf")
    if len(shapes) != len(attrs):
        raise SystemExit("ERROR: .shp and .dbf record counts differ")
    n_del = sum(1 for a in attrs if a.get("_deleted"))
    if n_del:
        print(f"  {n_del} deleted polygon(s) in the file are ignored "
              f"(they disappear when the layer is packed)")
    pairs = [(g, a.get(class_field)) for g, a in zip(shapes, attrs)
             if g is not None and not a.get("_deleted")]
    valid = set(CLASSES) | {NODATA}
    bad = [c for _, c in pairs if c not in valid]
    if bad:
        print(f"  WARNING: {len(bad)} polygon(s) with {class_field} not in "
              f"{sorted(valid)} — ignored: {sorted(set(map(str, bad)))}")
    pairs = [(g, int(c)) for g, c in pairs if c in valid]
    if not pairs:
        raise SystemExit(f"ERROR: no polygons with a valid {class_field}")

    counts = {c: sum(1 for _, k in pairs if k == c) for c in sorted(valid)}
    print("  polygons: " + f"{len(pairs)}  " +
          "  ".join(f"{CLASSES.get(c, 'no data')}={counts[c]}" for c in sorted(valid)))

    # Rasterise. Overlaps: the polygon drawn last wins, and we report them.
    fine = rasterize(pairs, out_shape=(fine_h, fine_w), transform=fine_tf,
                     fill=0, dtype="uint8", all_touched=False)
    per_class = {}
    for c in sorted(valid):
        only = rasterize([(g, 1) for g, k in pairs if k == c],
                         out_shape=(fine_h, fine_w), transform=fine_tf,
                         fill=0, dtype="uint8", all_touched=False) if counts[c] else 0
        per_class[c] = int(np.sum(only))
    drawn = int((fine > 0).sum())
    overlap = sum(per_class.values()) - drawn
    if overlap > 0:
        cell_ha = (FINE_RES * 110_574.0) * (FINE_RES * 111_320.0 *
                   math.cos(math.radians(vg["y1"] - ny * vg["res"] / 2))) / 1e4
        print(f"  WARNING: classes overlap on ~{overlap * cell_ha:.2f} ha "
              f"({100 * overlap / max(1, drawn):.1f} % of the drawn area); "
              f"the polygon drawn last wins")

    # Aggregate to VIIRS cells. Cells containing 'no data' can never be usable.
    blocks   = fine.reshape(ny, SUB, nx, SUB)
    frac     = {c: (blocks == c).sum(axis=(1, 3)) / (SUB * SUB) for c in CLASSES}
    nodata_f = (blocks == NODATA).sum(axis=(1, 3)) / (SUB * SUB)
    labelled = sum(frac.values())                      # excludes class 9 on purpose

    out_dir = os.path.join(patch_dir, "fractions")
    os.makedirs(out_dir, exist_ok=True)

    ls_labels = rasterize(pairs, out_shape=(lg["H"], lg["W"]), transform=ls_tf,
                          fill=0, dtype="uint8", all_touched=False)
    with rasterio.open(os.path.join(out_dir, "labels_landsat.tif"), "w", driver="GTiff",
                       height=lg["H"], width=lg["W"], count=1, dtype="uint8",
                       crs=C.CRS, transform=ls_tf, nodata=0, compress="deflate") as dst:
        dst.write(ls_labels, 1)
        dst.set_band_description(1, "class_id 1=river 2=ice 3=snow 4=land 9=no data")

    viirs_tf = from_origin(vg["x0"], vg["y1"], vg["res"], vg["res"])
    stack = np.stack([frac[c] for c in CLASSES] + [labelled]).astype("float32")
    with rasterio.open(os.path.join(out_dir, "fractions_375m.tif"), "w", driver="GTiff",
                       height=ny, width=nx, count=5, dtype="float32", crs=C.CRS,
                       transform=viirs_tf, nodata=np.nan, compress="deflate") as dst:
        dst.write(stack)
        for i, name in enumerate([CLASSES[c] for c in CLASSES] + ["labelled"], start=1):
            dst.set_band_description(i, name)

    rows = []
    for r in range(ny):
        for c in range(nx):
            if labelled[r, c] <= 0 and nodata_f[r, c] <= 0:
                continue
            rows.append({
                "row": r, "col": c,
                "alaska_row": vg["row0"] + r, "alaska_col": vg["col0"] + c,
                "lat": round(vg["y1"] - (r + 0.5) * vg["res"], 6),
                "lon": round(vg["x0"] + (c + 0.5) * vg["res"], 6),
                **{f"frac_{CLASSES[k]}": round(float(frac[k][r, c]), 4) for k in CLASSES},
                "frac_nodata": round(float(nodata_f[r, c]), 4),
                "labelled": round(float(labelled[r, c]), 4),
                "usable": int(labelled[r, c] >= min_labelled and nodata_f[r, c] == 0),
            })
    if not rows:
        raise SystemExit("ERROR: the polygons do not fall inside the patch grid")
    csv_path = os.path.join(out_dir, "fractions_375m.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_use = sum(r["usable"] for r in rows)
    print(f"\n  cells touched by labels : {len(rows)}")
    print(f"  cells >= {min_labelled:.0%} labelled and free of no-data: {n_use}")
    for c in CLASSES:
        v = np.array([r[f"frac_{CLASSES[c]}"] for r in rows if r["usable"]])
        if v.size:
            print(f"    {CLASSES[c]:6}: mean {v.mean():.3f}  max {v.max():.3f}  "
                  f"cells > 0: {(v > 0).sum()}")
    print(f"\n  {out_dir}")

    # cell grid for QGIS, carrying the fractions so it can be styled
    feats = []
    for r in range(ny):
        for c in range(nx):
            x, y = vg["x0"] + c * vg["res"], vg["y1"] - r * vg["res"]
            feats.append({
                "type": "Feature",
                "properties": {
                    "row": r, "col": c,
                    "labelled": round(float(labelled[r, c]), 4),
                    **{f"frac_{CLASSES[k]}": round(float(frac[k][r, c]), 4) for k in CLASSES},
                    "frac_nodata": round(float(nodata_f[r, c]), 4),
                    "usable": int(labelled[r, c] >= min_labelled and nodata_f[r, c] == 0),
                },
                "geometry": {"type": "Polygon", "coordinates": [[
                    [x, y], [x + vg["res"], y], [x + vg["res"], y - vg["res"]],
                    [x, y - vg["res"]], [x, y]]]},
            })
    grid_path = os.path.join(out_dir, "viirs_grid_375m.geojson")
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection",
                   "crs": {"type": "name",
                           "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
                   "features": feats}, f)
    print(f"  grid for QGIS: {grid_path}")

    # QC figure. Pixels are rectangular on the ground here, so stretch the
    # aspect to show true shape.
    lat_mid = vg["y1"] - ny * vg["res"] / 2
    aspect = 1.0 / math.cos(math.radians(lat_mid))
    fig, ax = plt.subplots(1, 6, figsize=(20, 9))
    cmap = matplotlib.colors.ListedColormap(
        ["#00000000"] + [C.CLASS_COLOURS[c] for c in CLASSES])
    ax[0].imshow(np.where(ls_labels > 4, 0, ls_labels), cmap=cmap, vmin=0, vmax=4,
                 interpolation="nearest", aspect=aspect)
    ax[0].set_title("labels (Landsat grid)")
    for i, c in enumerate(CLASSES, start=1):
        ax[i].imshow(np.where(labelled > 0, frac[c], np.nan), vmin=0, vmax=1,
                     cmap="viridis", interpolation="nearest", aspect=aspect)
        ax[i].set_title(f"fraction {CLASSES[c]}")
    im = ax[5].imshow(labelled, vmin=0, vmax=1, cmap="Greys_r",
                      interpolation="nearest", aspect=aspect)
    ax[5].set_title("labelled fraction")
    for a in ax:
        a.axis("off")
    plt.colorbar(im, ax=ax[5], fraction=0.04)
    fig.suptitle(f"{patch} — {len(pairs)} polygons, {n_use} usable cells", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fractions_qc.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-VIIRS-pixel class fractions from QGIS labels.")
    p.add_argument("--patch", default="nenana_20240622")
    p.add_argument("--class-field", default="class_id")
    p.add_argument("--min-labelled", type=float, default=0.95,
                   help="cell counts as usable when this much of it is labelled")
    a = p.parse_args()
    main(a.patch, a.class_field, a.min_labelled)
