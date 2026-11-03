"""
Check one patch's hand labels against independent water-fraction estimates.

Supersedes s2_label_check.py, which thresholded Sentinel NDWI at 0 and so missed
narrow channels: a 30-50 % water pixel never gets green above NIR, and was
reported completely dry. Of the 658 pixels that version flagged on 2024-06-22,
79 % carry real water. The sensor was never the limit; the binary cut was.

DIAGNOSTIC ONLY. Nothing here feeds the pipeline, and Sentinel-2 is never a
model feature or a label source -- only some dates have a coincident scene, so
using it as an input would make the dates incomparable.

HOW THE FRACTION IS ESTIMATED

By linear spectral unmixing against THREE endmembers, because the scene has
three surfaces, not two. An earlier two-endmember version (water and land)
reported the dry gravel islands in the main channel as 85 % water, because
gravel is almost identical to water in green (0.071 vs 0.074) and only slightly
brighter in NIR, so a water<->land line had nowhere else to put it.

SWIR is what separates them: gravel 0.083 against water 0.012, a factor of
seven. Vegetation is separated from both by NIR. So:

  water        green 0.074  NIR 0.054  SWIR1 0.012   (2024-06-22 medians)
  gravel       green 0.071  NIR 0.088  SWIR1 0.083
  vegetation   green 0.037  NIR 0.289  SWIR1 0.141

  f_mix    the water coefficient of the three-endmember unmix over green, red,
           NIR, SWIR1 and SWIR2, constrained to the simplex. The estimate to use.
  f_gravel the gravel coefficient. Bright dry bars and islands, which the
           two-endmember version silently counted as water.
  resid    what the three endmembers cannot explain. With land split into
           vegetation and gravel this is far smaller than before, so a large
           value now means something: shadow, snow, cloud, or a surface we have
           not modelled.
  f_nir    single-band NIR unmix, kept only to show how badly one band does.
  s2_frac  the same three-endmember unmix at Sentinel's 10 m, averaged over the
           3 x 3 cells that nest exactly in one Landsat cell.

Endmembers come from THIS date's own pixels: water and vegetation from the hand
labels, gravel from the polygons in GRAVEL_POLYS -- bars identified by eye,
since no index finds them on its own. Update that list per date.

NDWI is carried as an independent cross-check, never as the estimate. It is a
ratio, so overall brightness cancels: it was not fooled by the gravel islands
(-0.106 against water's +0.160) when absolute reflectance was. Where the unmix
says water and NDWI disagrees, the pixel is flagged rather than believed.

Sentinel nodata is excluded explicitly: the 8-bit composites store it as zero in
every band, which a naive unmix reads as 100 % water (2.2 % of the 2024-06-22
box, along the top edge where the scene does not reach). Sentinel has no SWIR
band on disk, so B11 is recovered exactly from the MNDWI raster and green:
B11 = B3 (1 - MNDWI) / (1 + MNDWI). B11 is natively 20 m, so the Sentinel
gravel/water split carries 20 m detail at 10 m spacing.

KNOWN LIMITS
  - Shadow detection is spectral, by choice: no DEM hillshade. On 2024-06-22 it
    is moot -- f_nir and f_mix never differ by more than 0.041 over labelled
    pixels, so there is no shadow/water confusion at 48.6 deg sun elevation. At
    October's 21.1 deg it will matter, and skylight is blue- and green-rich, so
    shadowed green is lifted towards water.
  - Ice and snow also darken NIR and brighten green, so this is for an ice-free
    date; they would need their own endmember. Breakup dates are refused: the
    Sentinel scene is minutes to tens of minutes away and floes drift more than
    a metre per second.

Outputs (in narrow_rivers/classify/qc/). One quantity per file, each with a .qml
that QGIS loads automatically, so layers can be toggled individually -- a
multi-band float raster has no sensible default rendering and QGIS paints the
first three bands as red/green/blue, which is meaningless here.

  label_check_<date>.txt        the report
  label_check_<date>.geojson    flagged pixels, with every estimate attached
  label_check_<date>.qml        one colour and shape per flag kind
  f_mix_<date>.tif              water fraction 0-1, 30 m. THE ESTIMATE TO USE:
                                min(f_unmix, f_ndwi), so a pixel is only as wet
                                as both methods allow.
  f_unmix_<date>.tif            the unmix alone -- over-calls dark vegetation
  f_ndwi_<date>.tif             NDWI alone, calibrated against Sentinel
  f_nir_<date>.tif              water fraction from NIR alone, 30 m. For
                                comparison only: it calls anything dark water.
  f_gravel_<date>.tif           gravel fraction 0-1, 30 m -- the dry bright bars
  resid_<date>.tif              what no endmember mixture explains
  s2_frac_<date>.tif            Sentinel water fraction averaged to 30 m
  s2_frac_10m_<date>.tif        the same at Sentinel's native 10 m -- the layer
                                to judge a narrow channel by eye

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\classify\\qc\\label_check.py --date 2024-06-22
"""

import os, sys, csv, json, argparse, warnings
import numpy as np
import rasterio

HERE     = os.path.dirname(os.path.abspath(__file__))
CLASSIFY = os.path.dirname(HERE)
NR_ROOT  = os.path.dirname(CLASSIFY)
PATCHES  = os.path.join(NR_ROOT, "patches")
sys.path.insert(0, PATCHES)
import patch_config as C

S2_SUB     = 3
B8_VMAX    = 0.45     # the 843 composite's 8-bit scaling, from add_sentinel.py
DRY_CUT    = 0.05     # a river label this dry on both estimates is suspect
WET_CUT    = 0.60     # a land label this wet on both estimates is suspect
RESID_CUT  = 0.030    # unexplained by all three endmembers
SHADOW_GAP = 0.30     # f_nir exceeding f_mix by this much suspects shadow
DISAGREE   = 0.35     # any disagreement this large goes to your eye
NDWI_WATER = -0.05    # below this, NDWI denies water however bright the pixel
TOP_POLYS  = 10
BREAKUP_DATES = {"2023-05-05", "2024-04-21"}

# Gravel bars and islands, identified by eye per date: no index finds them, and
# without them as an endmember the unmix calls them water. 2024-06-22: polygons
# 39 and 20 are dry gravel islands inside the main channel, each cut through by
# channels too small to see at 30 m. Most similar islands in the main channel
# are the same, so this list will grow.
GRAVEL_POLYS = {"2024-06-22": {39, 20}}

# Dark dense forest, in band order green, red, NIR, SWIR1, SWIR2. A single
# vegetation endmember cannot span bright tundra and dark spruce, and anything
# darker than the median gets explained by adding water: on 2024-06-22 that was
# 7,531 pixels sitting at f_unmix 0.70 with NDWI -0.75. These 161 clusters are
# concentrated in the north of the box; the largest, 4,541 pixels centred on
# 64.5127, -149.1395, was confirmed as dark dense forest by eye (2026-10-27) and
# is the median spectrum below. Clusters 2, 5 and 6 lie within 0.011-0.025 of
# it, so one endmember covers the bulk of them.
# Hand-drawn dark-forest polygons, same mechanism as GRAVEL_POLYS. These
# replaced a hardcoded spectrum (from the largest ndwi_denies cluster) and a
# threshold rule: the rule kept landing on mildly-dark forest, only 0.067 from
# the vegetation endmember where water is 0.242 away, which left the Sentinel
# unmix ill-conditioned. A drawn polygon gives both sensors a real spectrum.
DARK_VEG_POLYS = {"2024-06-22": {104, 105}}

# Fraction layers share one ramp so they can be compared by eye; the residual
# gets its own, since it is not a fraction.
FRAC_RAMP = [(0.0, "#ffffff", "0.0  no water"), (0.25, "#c6dbef", "0.25"),
             (0.50, "#6baed6", "0.50  half water"), (0.75, "#2171b5", "0.75"),
             (1.0, "#08306b", "1.0  all water")]
GRAVEL_RAMP = [(0.0, "#ffffff", "0.0  no gravel"), (0.5, "#dfc27d", "0.50"),
               (1.0, "#8c510a", "1.0  all gravel")]
RESID_RAMP = [(0.0, "#ffffff", "0.00  explained by the endmembers"),
              (0.02, "#fee6ce", "0.02"), (0.04, "#fdae6b", "0.04"),
              (0.08, "#d94801", "0.08+  not a water/land mix")]

FLAG_STYLE = {
    "river_dry":      ("circle",   "227,26,28",   "labelled river, no water found"),
    "land_wet":       ("triangle", "255,191,0",   "labelled land, mostly water"),
    "shadow_suspect": ("square",   "117,112,179", "dark in NIR and green: shadow, not water"),
    "ndwi_denies":    ("pentagon", "106,61,154",  "unmix says water, NDWI says not"),
    "off_line":       ("star",     "217,95,2",    "no endmember mixture explains it"),
    "disagree":       ("diamond",  "27,158,119",  "the estimates disagree"),
}


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def unmixN(obs, ems):
    """Linear unmix against any number of endmembers, on the simplex.

    obs is (k, H, W) reflectance. ems is a list of length-k endmembers; the
    LAST one is the reference, eliminated by the sum-to-one constraint. Solving
    min ||obs - sum f_i E_i|| with sum f_i = 1 then reduces to unconstrained
    least squares in the first N-1 abundances. The design matrix is the same for
    every pixel, so its pseudo-inverse is computed once and applied as a single
    tensor contraction -- the whole scene in one pass.

    Returns (abundances, resid): a list of N arrays in the order of `ems`, and
    what no mixture explains.

    The simplex projection is a clip to non-negative followed by a rescale when
    the abundances exceed one. That is not the exact Euclidean projection, but
    the difference only shows up for pixels already far off the simplex, which
    the residual flags anyway.
    """
    E = [np.asarray(e, float) for e in ems]
    V = E[-1]
    A = np.stack([e - V for e in E[:-1]])            # (N-1, k) directions
    x = obs - V[:, None, None]
    G = A @ A.T
    if abs(np.linalg.det(G)) < 1e-14:
        raise SystemExit("ERROR: the endmembers are collinear, so the unmix is "
                         "not identifiable. Check GRAVEL_POLYS / DARK_VEG.")
    M = np.linalg.inv(G) @ A                        # (N-1, k)
    f = np.tensordot(M, x, axes=(1, 0))             # (N-1, H, W)

    f = np.clip(f, 0.0, 1.0)
    tot = f.sum(axis=0)
    over = tot > 1.0
    if over.any():
        f[:, over] /= tot[over]
        tot = f.sum(axis=0)

    r = x - np.tensordot(A, f, axes=(0, 0))
    out = [a.astype(np.float32) for a in f]
    out.append((1.0 - tot).clip(0.0, 1.0).astype(np.float32))
    return out, np.sqrt((r * r).sum(axis=0)).astype(np.float32)


def ndwi_of(green, nir):
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ((green - nir) / (green + nir)).astype(np.float32)
    v[~np.isfinite(v)] = np.nan
    return v


def unmix1(a, water, land):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((a - land) / (water - land), 0.0, 1.0).astype(np.float32)


def block_mean(a, k):
    """Mean over k x k blocks, ignoring nan; all-nan blocks come back nan."""
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3)).astype(np.float32)


def main(date, force):
    if date in BREAKUP_DATES and not force:
        raise SystemExit(
            f"ERROR: {date} is a breakup date. The Sentinel and Landsat scenes "
            f"are minutes apart and the ice will have moved between them, so a "
            f"disagreement measures drift, not labelling.")

    patch = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}")
    need = {"sr": f"{date}_sr.tif", "s2ndwi": f"{date}_s2_ndwi_10m.tif",
            "s2_843": f"{date}_s2_843_10m.tif",
            "s2swir": f"{date}_s2_swir_refl_10m.tif",
            "thermal": f"{date}_thermal.tif"}
    path = {k: os.path.join(patch, v) for k, v in need.items()}
    for k, p in path.items():
        if not os.path.exists(p):
            raise SystemExit(f"ERROR: missing {p}" +
                             ("  (run add_sentinel.py --extras)"
                              if k == "s2_843" else ""))

    with rasterio.open(path["sr"]) as d:
        sr = d.read()
        shape, transform, prof = d.shape, d.transform, d.profile
    green, nir = sr[1], sr[3]
    with rasterio.open(path["thermal"]) as d:
        thermal = d.read(1)

    with rasterio.open(path["s2_843"]) as d:
        comp = d.read()                       # band 1 NIR B8, 2 red B4, 3 green B3
        s2prof = d.profile
    nodata = (comp[0] == 0) & (comp[1] == 0) & (comp[2] == 0)
    s2_nir   = comp[0].astype(np.float32) / 255.0 * B8_VMAX
    s2_green = comp[2].astype(np.float32) / 255.0 * B8_VMAX
    with rasterio.open(path["s2ndwi"]) as d:
        s2ndwi = d.read(1)
    with rasterio.open(path["s2swir"]) as d:
        s2sw = d.read()                       # band 1 B11, band 2 B12, float

    # ---- samples -----------------------------------------------------------
    spath = os.path.join(CLASSIFY, f"samples_{date}.csv")
    if not os.path.exists(spath):
        raise SystemExit(f"ERROR: no sample table at {spath}")
    with open(spath, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["date"] == date]
    rr  = np.array([int(r["row"]) for r in rows])
    cc  = np.array([int(r["col"]) for r in rows])
    cid = np.array([int(r["class_id"]) for r in rows])

    def s2cells(arr, r, c):
        return arr[S2_SUB * r:S2_SUB * r + S2_SUB, S2_SUB * c:S2_SUB * c + S2_SUB]

    # unambiguous pixels: every Sentinel cell agrees, so there is no mixing to
    # argue about and the endmembers are not contaminated by edges
    pure_w = np.array([cid[i] == 1 and np.all(s2cells(s2ndwi, rr[i], cc[i]) > 0.0)
                       for i in range(len(rows))])
    pure_l = np.array([cid[i] == 4 and np.all(s2cells(s2ndwi, rr[i], cc[i]) < -0.5)
                       for i in range(len(rows))])
    if pure_w.sum() < 50 or pure_l.sum() < 50:
        raise SystemExit(f"ERROR: {pure_w.sum()} pure water and {pure_l.sum()} "
                         f"pure land pixels; too few to calibrate")

    # gravel: identified by eye, because no index finds a dry bright bar
    pid = np.array([int(r["poly_id"]) for r in rows])
    gset = GRAVEL_POLYS.get(date, set())
    grav = np.isin(pid, list(gset)) if gset else np.zeros(len(rows), bool)
    if gset and grav.sum() < 50:
        raise SystemExit(f"ERROR: GRAVEL_POLYS[{date}] = {sorted(gset)} covers "
                         f"only {int(grav.sum())} pixels; too few to calibrate")
    pure_l = pure_l & ~grav          # keep gravel out of the vegetation endmember
    _dpids = DARK_VEG_POLYS.get(date, set())
    if _dpids:                       # and keep dark forest out of it as well
        pure_l = pure_l & ~np.isin(
            np.array([int(r["poly_id"]) for r in rows]), list(_dpids))

    # green, red, NIR, SWIR1, SWIR2 -- blue is left out, it is the haziest band
    L_BANDS = [green, sr[2], nir, sr[4], sr[5]]
    obs = np.stack(L_BANDS)

    def em(sel):
        return [float(np.median(b[rr[sel], cc[sel]])) for b in L_BANDS]

    EM_W, EM_V = em(pure_w), em(pure_l)
    EM_G = em(grav) if grav.any() else None
    # poly_id is a running counter over kept polygons, so deleting one renumbers
    # every polygon after it and GRAVEL_POLYS can end up pointing at the wrong
    # feature -- silently, because any pixels yield some spectrum. Check that
    # what came back actually looks like a dry bright bar: SWIR well above
    # water, below vegetation, and NDWI negative.
    if EM_G is not None:
        g_ndwi = float(np.nanmedian(ndwi_of(green, nir)[rr[grav], cc[grav]]))
        if not (EM_W[3] * 2 < EM_G[3] < EM_V[3] and g_ndwi < 0.0):
            raise SystemExit(
                f"ERROR: GRAVEL_POLYS[{date}] = {sorted(gset)} does not look "
                f"like gravel: SWIR1 {EM_G[3]:.3f} (water {EM_W[3]:.3f}, "
                f"vegetation {EM_V[3]:.3f}), NDWI {g_ndwi:+.3f}. poly_id is "
                f"positional -- if the labels were edited, re-check which "
                f"polygons these now are.")
    dset = DARK_VEG_POLYS.get(date, set())
    dkpx = np.isin(pid, list(dset)) if dset else np.zeros(len(rows), bool)
    EM_D = em(dkpx) if dkpx.sum() >= 50 else None

    w_th = float(np.median(thermal[rr[pure_w], cc[pure_w]]))
    l_th = float(np.median(thermal[rr[pure_l], cc[pure_l]]))

    # Sentinel: green, red, NIR from the 843 composite, SWIR1 and SWIR2 as real
    # reflectance. Five independent bands, same as Landsat, so Sentinel can now
    # carry a fourth endmember.
    S2_BANDS = [s2_green, comp[1].astype(np.float32) / 255.0 * B8_VMAX,
                s2_nir, s2sw[0], s2sw[1]]
    s2obs = np.stack(S2_BANDS)
    ok = ~nodata

    def s2em(mask):
        return [float(np.median(b[mask])) for b in S2_BANDS]

    sEM_W = s2em((s2ndwi > 0.1) & ok)
    sEM_V = s2em((s2ndwi < -0.6) & ok)
    gmask = np.zeros(s2ndwi.shape, bool)
    for r, c in zip(rr[grav], cc[grav]):
        gmask[S2_SUB * r:S2_SUB * r + S2_SUB, S2_SUB * c:S2_SUB * c + S2_SUB] = True
    sEM_G = s2em(gmask & ok) if gmask.any() else None

    out = []

    def say(line=""):
        print(line)
        out.append(line)

    say("=" * 76)
    say(f"Label check -- {date}   (diagnostic, not a pipeline step)")
    say("=" * 76)
    say(f"  endmembers from this date only ({int(pure_w.sum()):,} water, "
        f"{int(pure_l.sum()):,} vegetation, {int(grav.sum()):,} gravel"
        + (f" from polygons {sorted(gset)}" if gset else " -- NONE DEFINED") + ")")
    say(f"    {'':12} {'green':>7} {'red':>7} {'NIR':>7} {'SWIR1':>7} {'SWIR2':>7}")
    for nm, e in [("water", EM_W), ("gravel", EM_G),
                  ("dark forest", EM_D), ("vegetation", EM_V)]:
        if e is None:
            continue
        say(f"    {nm:12} " + " ".join(f"{v:7.3f}" for v in e))
    say(f"    thermal      water {w_th:.1f} K   vegetation {l_th:.1f} K")
    if EM_G is None:
        say("    WARNING: no gravel endmember for this date. Dry bright bars "
            "will be reported as water. Add them to GRAVEL_POLYS.")
    say(f"  Sentinel nodata excluded: {int(nodata.sum()):,} cells "
        f"({100 * nodata.mean():.2f}% of the 10 m grid)")
    say()

    # NDWI: a ratio, so overall brightness cancels. Never the estimate, always
    # the veto -- it was not fooled by the gravel islands when reflectance was.
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = ((green - nir) / (green + nir)).astype(np.float32)
    ndwi[~np.isfinite(ndwi)] = np.nan

    # ---- the estimates -----------------------------------------------------
    if EM_G is None:                      # degenerate: no gravel to separate
        EM_G = [(a + b) / 2 for a, b in zip(EM_W, EM_V)]
        sEM_G = [(a + b) / 2 for a, b in zip(sEM_W, sEM_V)]
    ems = [EM_W, EM_G, EM_V] if EM_D is None else [EM_W, EM_G, EM_D, EM_V]
    ab, resid = unmixN(obs, ems)
    f_unmix, f_gravel = ab[0], ab[1]
    f_dark = ab[2] if EM_D is not None else np.zeros_like(f_unmix)
    f_nir = unmix1(nir, EM_W[2], EM_V[2])

    # Sentinel stays at THREE endmembers, and this is not a band-count problem:
    # with B11 and B12 as real reflectance it has five, same as Landsat. The
    # obstacle is endmember selection. Dark forest has no polygon at 10 m, so it
    # has to be chosen by rule, and every rule tried landed on mildly-dark
    # forest: 0.067 from the vegetation endmember where water is 0.242 away,
    # giving a condition number of 52 and abundances that wandered (land came
    # out at 0.36-0.61 water, which then flattened the NDWI curve downstream).
    # A hand-drawn dark-forest polygon would fix it; a rule will not.
    sEM_D = None
    if EM_D is not None:
        dmask = np.zeros(s2ndwi.shape, bool)
        for r, c in zip(rr[dkpx], cc[dkpx]):
            dmask[S2_SUB * r:S2_SUB * r + S2_SUB,
                  S2_SUB * c:S2_SUB * c + S2_SUB] = True
        if (dmask & ok).sum() > 200:
            sEM_D = s2em(dmask & ok)
    # Sentinel keeps THREE endmembers, and this is now a measured dead end, not
    # a missing input. With hand-drawn dark-forest polygons (104, 105) the
    # Sentinel dark-forest spectrum sits 0.032 from vegetation while water is
    # 0.242 away: condition number 1142, and s2_frac's land median rose from
    # 0.022 to 0.180. The rule-based version was 0.067 away at cond 52, so a
    # better polygon makes it WORSE, not better. At 10 m, with green/red/NIR and
    # SWIR, dark forest and ordinary vegetation are simply not separable enough
    # to be distinct endmembers. Landsat keeps its fourth endmember, where the
    # same polygons work fine.
    sems = [sEM_W, sEM_G, sEM_V]
    s2ab, _ = unmixN(s2obs, sems)
    # Report the conditioning rather than assume it: the previous rule-based
    # endmember sat too close to vegetation and the abundances wandered.
    _V = np.asarray(sems[-1], float)
    _A = np.stack([np.asarray(e, float) - _V for e in sems[:-1]])
    say(f"  Sentinel unmix: {len(sems)} endmembers, cond "
        f"{np.linalg.cond(_A @ _A.T):.1f}" +
        (f"; |dark-veg| {np.linalg.norm(np.asarray(sEM_D) - _V):.4f} vs "
         f"|water-veg| {np.linalg.norm(np.asarray(sEM_W) - _V):.4f}"
         if sEM_D is not None else " (no dark forest)"))
    s2_f = s2ab[0].astype(np.float32)
    s2_f[nodata] = np.nan
    s2_frac = block_mean(s2_f, S2_SUB)
    if s2_frac.shape != shape:
        raise SystemExit(f"ERROR: Sentinel block mean is {s2_frac.shape}, "
                         f"not {shape}; the grids do not nest")

    # ---- combine the unmix with NDWI --------------------------------------
    # The unmix gives the magnitude but is brightness-sensitive: dark vegetation
    # is explained by adding water (3,869 pixels at f_unmix 0.70 with NDWI
    # -0.75 on 2024-06-22). NDWI is a ratio, so brightness cancels and it is
    # not fooled -- but on its own it misses narrow channels, which never get
    # green above NIR. So take the conservative combination: a pixel is only as
    # wet as BOTH estimates allow.
    #
    # The NDWI->fraction curve is measured against s2_frac, not assumed, so the
    # relationship comes from this scene. s2_frac is gravel-aware, so the curve
    # is not contaminated by bright dry bars.
    # Calibrate against the THRESHOLD Sentinel fraction -- the share of 10 m
    # cells with NDWI > 0 -- not against s2_frac. The threshold definition
    # misses narrow channels, but gravel and dark forest both read NDWI < 0 at
    # 10 m, so it is not polluted by either. It is biased in COVERAGE, not in
    # the relationship, and the relationship is all we are fitting. Calibrating
    # against the unmix fraction instead produced a curve flat at -0.68 across
    # every bin, because the unmix's own mid-range errors went in as truth.
    wet = ((s2ndwi > 0.0) & ok).astype(np.float32)
    wet[nodata] = np.nan
    s2_thresh = block_mean(wet, S2_SUB)
    fs, vs = [], []
    for lo in np.arange(0.0, 1.0, 1.0 / 12):
        hi = lo + 1.0 / 12
        sel = (s2_thresh >= lo) & (s2_thresh < hi + (1e-9 if hi >= 1 else 0))
        sel &= np.isfinite(ndwi)
        if sel.sum() < 200:
            continue
        fs.append(float(lo + hi) / 2)
        vs.append(float(np.median(ndwi[sel])))
    if len(fs) < 3:
        raise SystemExit("ERROR: too few bins to calibrate NDWI")
    vs = np.maximum.accumulate(np.array(vs))
    f_ndwi = np.interp(ndwi, vs, np.array(fs), left=0.0, right=1.0).astype(np.float32)
    f_ndwi[~np.isfinite(ndwi)] = np.nan
    f_mix = np.minimum(f_unmix, f_ndwi).astype(np.float32)

    say("  NDWI calibration (median Landsat NDWI vs the Sentinel threshold "
        "fraction)")
    say("  " + "-" * 72)
    say("    water " + "  ".join(f"{x:5.2f}" for x in fs))
    say("    NDWI  " + "  ".join(f"{x:+5.2f}" for x in vs))
    say()
    say("  whole-box sanity check (the river is roughly 1.5 % of the box)")
    say("  " + "-" * 72)
    say(f"    cells >= 0.90 water   f_mix {100 * np.mean(f_mix >= 0.9):5.2f} %"
        f"   (unmix alone {100 * np.mean(f_unmix >= 0.9):5.2f} %,"
        f" NDWI alone {100 * np.nanmean(f_ndwi >= 0.9):5.2f} %,"
        f" NIR alone {100 * np.mean(f_nir >= 0.9):5.2f} %)")
    say(f"    gravel  f_gravel >= 0.50: "
        f"{100 * np.mean(f_gravel >= 0.5):5.2f} % of the box "
        f"(dry bars; the two-endmember version counted these as water)")
    # Three endmembers cannot describe a whole landscape -- forest, tundra and
    # bare soil are not one vegetation spectrum -- so a fixed residual cut is
    # meaningless. Calibrate it on labelled vegetation: anything beyond its 95th
    # percentile is unusual FOR THIS SCENE.
    resid_cut = float(np.nanpercentile(resid[rr[pure_l], cc[pure_l]], 95))
    say(f"    unexplained: resid cut {resid_cut:.4f} "
        f"(95th percentile of labelled vegetation), "
        f"{100 * np.mean(resid > resid_cut):5.2f} % of the box beyond it")
    say()

    # ---- per-class summary -------------------------------------------------
    say(f"  {len(rows):,} labelled pixels")
    say()
    say("  median estimate by hand-labelled class")
    say("  " + "-" * 72)
    say(f"  {'class':8} {'n':>7} {'f_mix':>8} {'f_gravel':>9} {'f_dark':>7} "
        f"{'s2_frac':>8} {'f_nir':>8} {'resid':>8}")
    for c in sorted(set(cid)):
        s = cid == c
        say(f"  {str(C.CLASSES.get(c, c)):8} {int(s.sum()):>7,} "
            f"{np.nanmedian(f_mix[rr[s], cc[s]]):>8.3f} "
            f"{np.nanmedian(f_gravel[rr[s], cc[s]]):>9.3f} "
            f"{np.nanmedian(f_dark[rr[s], cc[s]]):>7.3f} "
            f"{np.nanmedian(s2_frac[rr[s], cc[s]]):>8.3f} "
            f"{np.nanmedian(f_nir[rr[s], cc[s]]):>8.3f} "
            f"{np.nanmedian(resid[rr[s], cc[s]]):>8.4f}")
    say()

    # ---- flags -------------------------------------------------------------
    flags = []
    for i in range(len(rows)):
        r, c = int(rr[i]), int(cc[i])
        fm, fn = float(f_mix[r, c]), float(f_nir[r, c])
        s2 = float(s2_frac[r, c])
        rs = float(resid[r, c])
        th = float(thermal[r, c])
        th_landish = abs(th - l_th) < abs(th - w_th)
        kind = None
        nd = float(ndwi[r, c])
        if fn - fm > SHADOW_GAP and rs > resid_cut and th_landish:
            kind = "shadow_suspect"
        elif float(f_unmix[r, c]) - float(f_ndwi[r, c]) > DISAGREE:
            # the unmix claims much more water than NDWI allows: dark
            # vegetation or a bright dry surface. f_mix already took the lower
            # of the two, so this is a record of where that mattered.
            kind = "ndwi_denies"
        elif rs > resid_cut and 0.1 < fm < 0.9:
            kind = "off_line"
        elif cid[i] == 1 and fm <= DRY_CUT and (not np.isfinite(s2) or s2 <= DRY_CUT):
            kind = "river_dry"
        elif cid[i] == 4 and fm >= WET_CUT and (not np.isfinite(s2) or s2 >= WET_CUT):
            kind = "land_wet"
        elif np.isfinite(s2) and abs(fm - s2) > DISAGREE:
            kind = "disagree"
        if kind:
            flags.append({"audit_kind": kind, "class_id": int(cid[i]),
                          "poly_id": int(rows[i]["poly_id"]), "row": r, "col": c,
                          "f_mix": round(fm, 3),
                          "f_gravel": round(float(f_gravel[r, c]), 3),
                          "ndwi": round(nd, 3),
                          "s2_frac": None if not np.isfinite(s2) else round(s2, 3),
                          "f_nir": round(fn, 3), "resid": round(rs, 4),
                          "thermal_k": round(th, 1)})

    n_by = {k: sum(1 for f in flags if f["audit_kind"] == k) for k in FLAG_STYLE}
    n_river, n_land = int((cid == 1).sum()), int((cid == 4).sum())
    say("  Flags")
    say("  " + "-" * 72)
    for kind, (_, _, what) in FLAG_STYLE.items():
        base = (n_river if kind == "river_dry" else
                n_land if kind == "land_wet" else len(rows))
        say(f"  {kind:15} {n_by[kind]:>6,} of {base:,} "
            f"({pct(n_by[kind], base)})  -- {what}")
    say()

    # ---- which polygons first ---------------------------------------------
    flagged, totals = {}, {}
    for f in flags:
        k = (f["class_id"], f["poly_id"])
        flagged[k] = flagged.get(k, 0) + 1
    for i in range(len(rows)):
        k = (int(cid[i]), int(rows[i]["poly_id"]))
        totals[k] = totals.get(k, 0) + 1
    ranked = sorted(flagged.items(), key=lambda kv: -kv[1])[:TOP_POLYS]
    if ranked:
        say(f"  Worst {len(ranked)} polygons by flagged pixel count")
        say("  " + "-" * 72)
        say(f"  {'class':8} {'poly_id':>8} {'flagged':>8} {'of':>8} {'share':>7}")
        for (c, pid), n in ranked:
            say(f"  {str(C.CLASSES.get(c, c)):8} {pid:>8} {n:>8,} "
                f"{totals.get((c, pid), 0):>8,} "
                f"{pct(n, totals.get((c, pid), 0)):>7}")
        say()

    # ---- write -------------------------------------------------------------
    base = os.path.join(HERE, f"label_check_{date}")
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

    kinds = [k for k in FLAG_STYLE if n_by[k]]
    with open(base + ".qml", "w", encoding="utf-8") as f:
        f.write(flag_qml(kinds))

    say(f"  wrote  {os.path.basename(base)}.geojson   ({len(feats):,} points)")
    say(f"         {os.path.basename(base)}.qml")

    # one quantity per file: a 4-band float raster has no sensible default
    # rendering, and the bands cannot be toggled as separate layers
    layers = [("f_mix", f_mix, prof, FRAC_RAMP, 0.0, 1.0),
              ("f_nir", f_nir, prof, FRAC_RAMP, 0.0, 1.0),
              ("f_unmix", f_unmix, prof, FRAC_RAMP, 0.0, 1.0),
              ("f_ndwi", f_ndwi, prof, FRAC_RAMP, 0.0, 1.0),
              ("f_gravel", f_gravel, prof, GRAVEL_RAMP, 0.0, 1.0),
              ("resid", resid, prof, RESID_RAMP, 0.0, 0.08),
              ("s2_frac", s2_frac, prof, FRAC_RAMP, 0.0, 1.0),
              ("s2_frac_10m", s2_f, s2prof, FRAC_RAMP, 0.0, 1.0)]
    for name, arr, pr, ramp, lo, hi in layers:
        out_tif = os.path.join(HERE, f"{name}_{date}.tif")
        p = dict(pr, count=1, dtype="float32", nodata=float("nan"),
                 compress="deflate", predictor=2, tiled=True)
        with rasterio.open(out_tif, "w", **p) as d:
            d.write(arr.astype(np.float32), 1)
            d.set_band_description(1, name)
        open(os.path.join(HERE, f"{name}_{date}.qml"), "w",
             encoding="utf-8").write(ramp_qml(ramp, lo, hi))
        say(f"         {os.path.basename(out_tif):28} + .qml   "
            f"median {np.nanmedian(arr):.3f}")

    say(f"         {os.path.basename(base)}.txt")
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def ramp_qml(ramp, vmin, vmax):
    """Interpolated colour ramp for a single-band float raster."""
    items = "\n".join(
        f'      <item value="{v}" color="{c}" alpha="255" label="{lab}"/>'
        for v, c, lab in ramp)
    return ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34" styleCategories="Symbology">\n <pipe>\n'
            '  <rasterrenderer type="singlebandpseudocolor" band="1"'
            f' opacity="1" classificationMin="{vmin}"'
            f' classificationMax="{vmax}" nodataColor="">\n'
            '   <rastershader>\n'
            '    <colorrampshader colorRampType="INTERPOLATED" clip="0"'
            ' classificationMode="1" labelPrecision="2"'
            f' minimumValue="{vmin}" maximumValue="{vmax}">\n'
            f'{items}\n'
            '    </colorrampshader>\n   </rastershader>\n'
            '  </rasterrenderer>\n'
            '  <brightnesscontrast brightness="0" contrast="0"/>\n'
            ' </pipe>\n <blendMode>0</blendMode>\n</qgis>\n')


def _marker(name, shape, rgb):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Label check from a two-band unmix.")
    ap.add_argument("--date", required=True, help="patch date, e.g. 2024-06-22")
    ap.add_argument("--force", action="store_true",
                    help="run anyway on a breakup date")
    a = ap.parse_args()
    main(a.date, a.force)
