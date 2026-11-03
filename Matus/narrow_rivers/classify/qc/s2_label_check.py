"""
Check one patch's hand labels against Sentinel-2 at 10 m.

DIAGNOSTIC ONLY. Nothing here feeds the pipeline. It reads the patch's
Sentinel-2 NDWI and the sample table and writes a report; it is never a model
feature and never a label source. Only some dates have a near-coincident
cloud-free S2 scene, so promoting it to an input would make the dates
incomparable with each other.

Each Landsat cell holds exactly 3 x 3 Sentinel cells on the same origin, so the
water fraction is a block mean -- no resampling, no interpolation. That also
means the fraction comes in ninths: 0, 0.11, 0.22 ... 1.0. Do not read 0.22 as
meaningfully different from 0.33.

NOT VALID ON A BREAKUP DATE. The S2 scene sits minutes to tens of minutes from
the Landsat one. Ice floes drift at more than a metre per second, so on a
breakup date the two images show the ice in different places and a disagreement
says nothing about the labels. Water extent on a static channel is fine; ice
position is not. Those dates are refused unless --force is passed.

Water here is NDWI > 0 ("the edge sits near 0", per the patch README). That
separates water from land. It does NOT separate water from snow or ice, which
are also bright in green and dark in NIR, so a freeze-up or winter date needs a
visible-brightness term that this script does not implement.

Outputs (in narrow_rivers/classify/qc/):
  s2_label_check_<date>.txt       the report
  s2_label_check_<date>.geojson   the disagreeing pixels, for QGIS
  s2_label_check_<date>.qml       red circles / amber triangles by flag kind;
                                  QGIS picks it up automatically by filename
  s2_waterfrac_<date>.tif         water fraction per Landsat cell, float32

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\qc\\s2_label_check.py --date 2024-06-22
"""

import os, sys, csv, json, argparse
import numpy as np
import rasterio

HERE     = os.path.dirname(os.path.abspath(__file__))
CLASSIFY = os.path.dirname(HERE)
NR_ROOT  = os.path.dirname(CLASSIFY)
PATCHES  = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
import patch_config as C

NDWI_THRESH = 0.0
SENSITIVITY = (-0.10, 0.0, 0.10)
WET_CUT     = 6 / 9          # a land label this wet is suspect
THIN_CUT    = 2 / 9          # a river label this dry is thin, not yet an error
TOP_POLYS   = 10
# The fractions are float32 ninths, so a cut written as 2/9 in double precision
# sits just below the stored value and drops a whole bin. Compare with slack.
EPS         = 1e-6

# The S2/Landsat time gap only matters where the scene moves between the two
# acquisitions. Breakup is the case where it does.
BREAKUP_DATES = {"2023-05-05", "2024-04-21"}


# Flag styling. Distinct in both colour and shape, and deliberately clear of
# the label palette in patch_config (river blue, land brown, ice magenta,
# snow cyan) so a flag can never be mistaken for a class.
FLAG_STYLE = {
    "river_dry": ("circle",   "227,26,28",   "labelled river, no S2 water"),
    "land_wet":  ("triangle", "255,191,0",   "labelled land, >= 0.67 water"),
}


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def _marker(name, shape, rgb):
    """One categorised marker symbol. Both the Option map (QGIS >= 3.20) and
    the legacy prop tags are written, so older QGIS builds still read it."""
    opts = [("name", shape), ("color", rgb + ",255"),
            ("outline_color", "0,0,0,255"), ("outline_style", "solid"),
            ("outline_width", "0.2"), ("outline_width_unit", "MM"),
            ("size", "2.2"), ("size_unit", "MM"), ("angle", "0"),
            ("offset", "0,0"), ("offset_unit", "MM"),
            ("scale_method", "diameter"), ("joinstyle", "bevel"),
            ("cap_style", "square"), ("horizontal_anchor_point", "1"),
            ("vertical_anchor_point", "1")]
    o = "\n".join(f'        <Option name="{k}" type="QString" value="{v}"/>'
                  for k, v in opts)
    p = "\n".join(f'      <prop k="{k}" v="{v}"/>' for k, v in opts)
    return (f'  <symbol type="marker" name="{name}" alpha="1" '
            f'clip_to_extent="1" force_rhr="0">\n'
            f'   <layer class="SimpleMarker" enabled="1" pass="0" locked="0">\n'
            f'    <Option type="Map">\n{o}\n    </Option>\n{p}\n'
            f'   </layer>\n  </symbol>\n')


def flag_qml(kinds):
    """Categorised style on audit_kind, for the disagreement point layer."""
    cats, syms = "", ""
    for i, kind in enumerate(kinds):
        shape, rgb, label = FLAG_STYLE[kind]
        cats += (f'   <category render="true" value="{kind}" symbol="{i}" '
                 f'label="{kind} - {label}"/>\n')
        syms += _marker(str(i), shape, rgb)
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34.0" styleCategories="Symbology">\n'
            ' <renderer-v2 type="categorizedSymbol" attr="audit_kind"\n'
            '              forceraster="0" symbollevels="0" enableorderby="0">\n'
            f'  <categories>\n{cats}  </categories>\n'
            f'  <symbols>\n{syms}  </symbols>\n'
            ' </renderer-v2>\n</qgis>\n')


def water_fraction(ndwi_path, land_shape, thresh):
    """Water fraction per Landsat cell from the 10 m NDWI, as a block mean."""
    with rasterio.open(ndwi_path) as d:
        ndwi = d.read(1)
    h, w = land_shape
    if ndwi.shape != (h * 3, w * 3):
        raise SystemExit(f"ERROR: {ndwi.shape} is not 3x {land_shape}; the "
                         f"grids do not nest, so a block mean is wrong here.")
    valid = np.isfinite(ndwi)
    wet   = (valid & (ndwi > thresh)).reshape(h, 3, w, 3)
    valid = valid.reshape(h, 3, w, 3)
    num = wet.sum(axis=(1, 3)).astype(np.float32)
    den = valid.sum(axis=(1, 3)).astype(np.float32)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan).astype(np.float32)


def load_samples(date):
    path = os.path.join(CLASSIFY, f"samples_{date}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: no sample table at {path}. Run "
                         f"extract_samples.py --dates {date} first.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["date"] == date]
    if not rows:
        raise SystemExit(f"ERROR: {path} holds no rows for {date}.")
    return rows


def ninth_histogram(fracs):
    """Counts per ninth, as a list of 10 ints."""
    idx = np.rint(np.asarray(fracs) * 9).astype(int)
    return [int((idx == k).sum()) for k in range(10)]


def count_flags(frac, rows):
    """(river labels with no water, land labels mostly water)."""
    dry = wet = 0
    for r in rows:
        f = frac[int(r["row"]), int(r["col"])]
        if not np.isfinite(f):
            continue
        cid = int(r["class_id"])
        if cid == 1 and f == 0:
            dry += 1
        elif cid == 4 and f >= WET_CUT - EPS:
            wet += 1
    return dry, wet


def main(date, thresh, force):
    if date in BREAKUP_DATES and not force:
        raise SystemExit(
            f"ERROR: {date} is a breakup date. The S2 and Landsat scenes are "
            f"minutes apart and the ice will have moved between them, so a "
            f"disagreement here measures drift, not labelling. Pass --force "
            f"only if you are asking about water extent on a static channel.")

    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    sr    = os.path.join(patch, f"{date}_sr.tif")
    ndwi  = os.path.join(patch, f"{date}_s2_ndwi_10m.tif")
    for p in (sr, ndwi):
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: missing {p}")

    with rasterio.open(sr) as d:
        shape, transform, crs = d.shape, d.transform, d.crs

    out = []

    def say(line=""):
        print(line)
        out.append(line)

    say("=" * 70)
    say(f"Sentinel-2 label check -- {date}   (diagnostic, not a pipeline step)")
    say("=" * 70)
    say(f"  water      NDWI > {thresh:+.2f} at 10 m, block-averaged 3x3 to 30 m")
    say( "  fractions  in ninths; 0.11 steps are the finest real distinction")
    say()

    frac = water_fraction(ndwi, shape, thresh)
    rows = load_samples(date)

    # -- per-class distribution of the S2 water fraction ---------------------
    by_class = {}
    for r in rows:
        f = frac[int(r["row"]), int(r["col"])]
        if np.isfinite(f):
            by_class.setdefault(int(r["class_id"]), []).append(float(f))

    qa_bad = sum(1 for r in rows if r.get("qa_bad") == "1")
    say(f"  {len(rows):,} labelled pixels, {qa_bad:,} flagged qa_bad "
        f"(kept here; train_rf drops them with --drop-qa)")
    say()
    say("  S2 water fraction by hand-labelled class")
    say("  " + "-" * 66)
    say(f"  {'class':8} {'n':>7} {'mean':>7} {'median':>7} {'frac=0':>8} "
        f"{'frac=1':>8}")
    for cid in sorted(by_class):
        v = np.asarray(by_class[cid])
        say(f"  {C.CLASSES.get(cid, cid):8} {len(v):>7,} {v.mean():>7.3f} "
            f"{np.median(v):>7.3f} {int((v == 0).sum()):>8,} "
            f"{int((v == 1).sum()):>8,}")
    say()
    say("  Histogram in ninths (columns 0/9 .. 9/9)")
    say("  " + "-" * 66)
    for cid in sorted(by_class):
        h = ninth_histogram(by_class[cid])
        say(f"  {C.CLASSES.get(cid, cid):8} " + " ".join(f"{n:>6,}" for n in h))
    say()

    # -- the disagreements ---------------------------------------------------
    flags = []
    for r in rows:
        rr, cc = int(r["row"]), int(r["col"])
        f = frac[rr, cc]
        if not np.isfinite(f):
            continue
        cid = int(r["class_id"])
        kind = None
        if cid == 1 and f == 0:
            kind = "river_dry"          # called river, S2 sees no water at all
        elif cid == 4 and f >= WET_CUT - EPS:
            kind = "land_wet"           # called land, S2 sees mostly water
        if kind:
            flags.append({"audit_kind": kind, "class_id": cid,
                          "poly_id": int(r["poly_id"]), "row": rr, "col": cc,
                          "s2_frac": round(float(f), 3)})

    n_river = len(by_class.get(1, []))
    n_land  = len(by_class.get(4, []))
    n_dry, n_wet = count_flags(frac, rows)
    thin = sum(1 for f in by_class.get(1, []) if 0 < f <= THIN_CUT + EPS)

    say("  Disagreements")
    say("  " + "-" * 66)
    say(f"  river_dry  {n_dry:>6,} of {n_river:,} river pixels "
        f"({pct(n_dry, n_river)}) have no S2 water at all")
    say(f"  land_wet   {n_wet:>6,} of {n_land:,} land pixels "
        f"({pct(n_wet, n_land)}) are >= {WET_CUT:.2f} water")
    say(f"  (thin      {thin:>6,} river pixels at 0 < frac <= {THIN_CUT:.2f}: "
        f"genuinely mixed, not an error)")
    say()

    # -- which polygons to look at first -------------------------------------
    flagged, totals = {}, {}
    for g in flags:
        k = (g["class_id"], g["poly_id"])
        flagged[k] = flagged.get(k, 0) + 1
    for r in rows:
        k = (int(r["class_id"]), int(r["poly_id"]))
        totals[k] = totals.get(k, 0) + 1
    ranked = sorted(flagged.items(), key=lambda kv: -kv[1])[:TOP_POLYS]
    if ranked:
        say(f"  Worst {len(ranked)} polygons by flagged pixel count "
            f"(edit these in QGIS first)")
        say("  " + "-" * 66)
        say(f"  {'class':8} {'poly_id':>8} {'flagged':>8} {'of':>8} {'share':>7}")
        for (cid, pid), n in ranked:
            tot = totals.get((cid, pid), 0)
            say(f"  {C.CLASSES.get(cid, cid):8} {pid:>8} {n:>8,} {tot:>8,} "
                f"{pct(n, tot):>7}")
        say()

    # -- does the answer survive a different threshold? ----------------------
    say("  Threshold sensitivity")
    say("  " + "-" * 66)
    say(f"  {'NDWI >':>8} {'river_dry':>10} {'land_wet':>10}")
    for t in SENSITIVITY:
        fr = frac if t == thresh else water_fraction(ndwi, shape, t)
        d, w = count_flags(fr, rows)
        mark = "  <- used" if t == thresh else ""
        say(f"  {t:>+8.2f} {d:>10,} {w:>10,}{mark}")
    say()

    # -- write ---------------------------------------------------------------
    base = os.path.join(HERE, f"s2_label_check_{date}")

    feats = []
    for g in flags:
        lon, lat = transform * (g["col"] + 0.5, g["row"] + 0.5)
        props = dict(g)
        props["class"] = C.CLASSES.get(g["class_id"], str(g["class_id"]))
        feats.append({"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": props})
    with open(base + ".geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, indent=1)

    kinds = [k for k in FLAG_STYLE if any(g["audit_kind"] == k for g in flags)]
    with open(base + ".qml", "w", encoding="utf-8") as f:
        f.write(flag_qml(kinds))

    tif = os.path.join(HERE, f"s2_waterfrac_{date}.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=shape[0],
                       width=shape[1], count=1, dtype="float32", crs=crs,
                       transform=transform, nodata=float("nan"),
                       compress="deflate") as d:
        d.write(frac, 1)

    say(f"  wrote  {os.path.basename(base)}.txt")
    say(f"         {os.path.basename(base)}.geojson   ({len(feats):,} points)")
    say(f"         {os.path.basename(base)}.qml")
    say(f"         {os.path.basename(tif)}")
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--date", required=True, help="patch date, e.g. 2024-06-22")
    ap.add_argument("--ndwi-thresh", type=float, default=NDWI_THRESH,
                    help="10 m NDWI above this counts as water (default 0.0)")
    ap.add_argument("--force", action="store_true",
                    help="run anyway on a breakup date")
    a = ap.parse_args()
    main(a.date, a.ndwi_thresh, a.force)
