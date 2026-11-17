"""
First U-Net on VIIRS: predict per-cell class fractions (river, ice, snow, land)
from the five VIIRS I-bands, trained only on cells you have checked ("done").

THIS IS A PIPELINE TEST. One scene and ~130 checked cells in one corner of the
box are far too little to judge a U-Net; the point is that the chain
VIIRS -> model -> fractions works end to end and is scored honestly.

Training cells = centre inside your done polygon (reference_masks/<date>/done/) AND
  - a usable cell (select_cells.py), or
  - a pure-land cell (no water/ice at all), agreed 2026-11-03: without verified
    land the model never sees "no river"
  and RF-unsure <= --max-unsure, fully inside the box.
Target = the four fractions renormalised over the non-unsure part of the cell.

Split: the grid is cut into blocks of --block x --block cells; 25 % of the blocks
that hold training cells go to TEST (seeded). Neighbouring cells share a block,
so a test cell's neighbours are not in training. The model sees the whole VIIRS
image as input (that is only radiance, no labels); the loss uses train cells only.

Baselines on the same split, so the U-Net numbers can be read:
  mean   always predict the training-set mean fractions
  rf     random forest on the 5 bands of each cell alone (no neighbourhood)

Each run goes to narrow_rivers/unet/runs/<timestamp>_<date>/:
  config.json, metrics.json, log.txt, model.pt,
  pred_<date>.tif (bands f_river f_ice f_snow f_land f_channel) + .qml,
  split_<date>.tif (1 train, 2 test) + .qml,
  pred_<other>.tif for --predict-dates (visual only, no labels there)

Run from Matus/:
    .venv\\Scripts\\python.exe narrow_rivers\\unet\\train_unet.py --date 2023-10-04
"""

import os, sys, json, argparse, datetime, time
import numpy as np
import rasterio
from rasterio.transform import from_origin

HERE = os.path.dirname(os.path.abspath(__file__))
NR_ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(NR_ROOT, "patches")
REF = os.path.join(NR_ROOT, "reference_masks")
sys.path.insert(0, PATCHES)
sys.path.insert(0, os.path.join(NR_ROOT, "classify"))
import patch_config as C
import extract_samples as EX          # label_rasters (read-only)

BANDS = ["I1", "I2", "I3", "I4", "I5"]
CLASS_NAMES = ["river", "ice", "snow", "land"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


# ── data ─────────────────────────────────────────────────────────────────────

def viirs(date):
    p = os.path.join(PATCHES, f"nenana_{date.replace('-', '')}", f"{date}_viirs_375m.tif")
    with rasterio.open(p) as s:
        d = list(s.descriptions)
        x = np.stack([s.read(d.index(b) + 1) for b in BANDS]).astype(np.float32)
    # z-score per image and band (no labels involved). Different dates are
    # therefore each standardised on themselves -- crude, like anchoring.
    m = np.nanmean(x, axis=(1, 2), keepdims=True)
    sd = np.nanstd(x, axis=(1, 2), keepdims=True) + 1e-6
    x = (x - m) / sd
    return np.nan_to_num(x, nan=0.0)


def latest_export(date):
    fdir = os.path.join(REF, date, "final")
    cands = sorted(f for f in os.listdir(fdir)
                   if f.startswith(f"{date}_fractions_375m_") and f.endswith(".tif"))
    if not cands:
        raise SystemExit(f"ERROR: no fractions export in {fdir} — run apply_overrides.py")
    return os.path.join(fdir, cands[-1])


def targets(date, max_unsure, log):
    vg = C.viirs_grid()
    ny, nx, res = vg["H"], vg["W"], vg["res"]
    fpath = latest_export(date)
    with rasterio.open(fpath) as s:
        b = {s.descriptions[i]: s.read(i + 1) for i in range(s.count)}
    with rasterio.open(os.path.join(REF, date, f"{date}_cell_status_375m.tif")) as s:
        status = s.read(1)
    with rasterio.open(os.path.join(REF, date, f"{date}_base_water.tif")) as s:
        shape, tf = s.shape, s.transform
    done_r, _, polys = EX.label_rasters(os.path.join(REF, date, "done", f"done_{date}.shp"),
                                        shape, tf)
    rc = ((vg["y1"] - (np.arange(ny) + 0.5) * res - tf.f) / tf.e).astype(int)
    cc = (((vg["x0"] + (np.arange(nx) + 0.5) * res) - tf.c) / tf.a).astype(int)
    R, Cc = np.meshgrid(np.clip(rc, 0, shape[0] - 1), np.clip(cc, 0, shape[1] - 1), indexing="ij")
    inside = done_r[R, Cc] > 0

    f = np.stack([b["f_river"], b["f_ice"], b["f_snow"], b["f_land"]])
    covered = (f.sum(0) + b["f_unsure"]) > 0.999          # fully inside Landsat
    known = f.sum(0)
    usable = status == 1
    pure_land = (status == 0) & covered
    train = inside & (usable | pure_land) & (b["f_unsure"] <= max_unsure) & (known > 0)
    t = f / np.maximum(known, 1e-9)
    log(f"  targets from {os.path.relpath(fpath, NR_ROOT)}; {len(polys)} done polygon(s)")
    log(f"  checked cells: {int(train.sum())} = {int((train & usable).sum())} with river/ice "
        f"+ {int((train & pure_land).sum())} pure land")
    return t.astype(np.float32), train, usable


def block_split(train, block, seed, test_frac, channel):
    ny, nx = train.shape
    bid = (np.arange(ny)[:, None] // block) * 1000 + (np.arange(nx)[None, :] // block)
    blocks = np.unique(bid[train])
    rng = np.random.default_rng(seed)
    # several seeded draws; keep the one whose test share of channel cells is
    # closest to test_frac (so the test set is not all land by accident)
    best, best_err = None, 9
    for _ in range(200):
        pick = rng.choice(blocks, size=max(1, int(round(test_frac * len(blocks)))), replace=False)
        te = train & np.isin(bid, pick)
        n_ch = (channel & train).sum()
        err = abs((te & channel).sum() / max(n_ch, 1) - test_frac) \
            + abs(te.sum() / train.sum() - test_frac)
        if err < best_err:
            best, best_err = te, err
    return train & ~best, best


# ── model ────────────────────────────────────────────────────────────────────

def build_unet(torch, cin, cout, w=16, p_drop=0.1):
    nn = torch.nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                             nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
            self.d2 = block(4 * w, 2 * w)
            self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
            self.d1 = block(2 * w, w)
            self.drop = nn.Dropout2d(p_drop)
            self.out = nn.Conv2d(w, cout, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.drop(self.e3(self.pool(e2)))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(self.drop(d1))

    return UNet()


# ── scoring ──────────────────────────────────────────────────────────────────

def score(pred, t, m, channel):
    """pred, t: (4, H, W) fractions; m: cells to score."""
    out = {"n_cells": int(m.sum()), "n_channel_cells": int((m & channel).sum())}
    for i, nm in enumerate(CLASS_NAMES):
        out[f"mae_{nm}"] = float(np.abs(pred[i][m] - t[i][m]).mean())
    pc, tc = pred[0] + pred[1], t[0] + t[1]
    out["mae_channel"] = float(np.abs(pc[m] - tc[m]).mean())
    mc = m & channel
    out["mae_channel_on_channel_cells"] = float(np.abs(pc[mc] - tc[mc]).mean()) if mc.any() else None
    ml = m & ~channel
    out["false_channel_on_land_cells"] = float(pc[ml].mean()) if ml.any() else None
    if mc.sum() > 2:
        out["r_channel"] = float(np.corrcoef(pc[m], tc[m])[0, 1])
    return out


def show(log, name, s):
    log(f"    {name:6} channel MAE {s['mae_channel']:.3f} (on river cells "
        f"{s['mae_channel_on_channel_cells'] if s['mae_channel_on_channel_cells'] is None else round(s['mae_channel_on_channel_cells'], 3)}, "
        f"false channel on land {s['false_channel_on_land_cells'] if s['false_channel_on_land_cells'] is None else round(s['false_channel_on_land_cells'], 3)})"
        f"  r {s.get('r_channel', float('nan')):.2f} | MAE river {s['mae_river']:.3f} "
        f"ice {s['mae_ice']:.3f} snow {s['mae_snow']:.3f} land {s['mae_land']:.3f}")


def write_tif(path, arrs, names, dtype="float32"):
    vg = C.viirs_grid()
    with rasterio.open(path, "w", driver="GTiff", height=vg["H"], width=vg["W"], count=len(arrs),
                       dtype=dtype, crs=vg["crs"],
                       transform=from_origin(vg["x0"], vg["y1"], vg["res"], vg["res"]),
                       compress="deflate") as d:
        for i, (a, n) in enumerate(zip(arrs, names), 1):
            d.write(a.astype(dtype), i)
            d.set_band_description(i, n)


PRED_QML = """<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="singlebandpseudocolor" band="5" opacity="1" alphaBand="-1">
      <rastershader><colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
        <item value="0" color="#ffffff" alpha="0" label="0"/>
        <item value="0.05" color="#c6dbef" alpha="200" label="0.05"/>
        <item value="0.2" color="#6baed6" alpha="230" label="0.2"/>
        <item value="0.5" color="#2171b5" alpha="255" label="0.5"/>
        <item value="1" color="#08306b" alpha="255" label="1"/>
      </colorrampshader></rastershader>
    </rasterrenderer>
  </pipe>
</qgis>
"""

SPLIT_QML = """<!DOCTYPE qgis>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
        <paletteEntry value="0" color="#000000" alpha="0" label="not used"/>
        <paletteEntry value="1" color="#2ca02c" alpha="170" label="1 train"/>
        <paletteEntry value="2" color="#ff7f0e" alpha="200" label="2 test"/>
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def main(a):
    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = os.path.join(HERE, "runs", f"{stamp}_{a.date}")
    os.makedirs(run)
    log = Tee(os.path.join(run, "log.txt"))
    log(f"U-Net pipeline test — {a.date}, run {os.path.basename(run)}")

    x = viirs(a.date)
    t, cells, usable = targets(a.date, a.max_unsure, log)
    channel = (t[0] + t[1]) > 0
    tr, te = block_split(cells, a.block, a.seed, 0.25, channel)
    log(f"  split by {a.block}x{a.block}-cell blocks: train {int(tr.sum())} cells "
        f"({int((tr & channel).sum())} with river/ice), test {int(te.sum())} "
        f"({int((te & channel).sum())} with river/ice)")

    # ── baselines ───────────────────────────────────────────────────────────
    from sklearn.ensemble import RandomForestRegressor
    mean_t = t[:, tr].mean(1)
    pred_mean = np.broadcast_to(mean_t[:, None, None], t.shape).copy()
    X_tr = x[:, tr].T
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=a.seed, n_jobs=-1)
    rf.fit(X_tr, t[:, tr].T)
    pred_rf = rf.predict(x.reshape(5, -1).T).T.reshape(t.shape)
    pred_rf = pred_rf / np.maximum(pred_rf.sum(0, keepdims=True), 1e-9)

    # ── U-Net ───────────────────────────────────────────────────────────────
    torch.set_num_threads(min(8, torch.get_num_threads()))
    net = build_unet(torch, 5, 4, w=a.width)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    X = torch.from_numpy(x)[None]
    T = torch.from_numpy(t)[None]
    M = torch.from_numpy(tr.astype(np.float32))[None]
    log(f"  U-Net width {a.width}, {sum(p.numel() for p in net.parameters()):,} parameters, "
        f"{a.epochs} steps, lr {a.lr}")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        xb, tb, mb = X, T, M
        if np.random.rand() < 0.5:                       # flips only: cells are not square,
            xb, tb, mb = xb.flip(-1), tb.flip(-1), mb.flip(-1)   # so no 90-degree rotation
        if np.random.rand() < 0.5:
            xb, tb, mb = xb.flip(-2), tb.flip(-2), mb.flip(-2)
        logp = torch.log_softmax(net(xb), 1)
        loss = -(tb * logp).sum(1)                       # soft cross-entropy per cell
        loss = (loss * mb).sum() / mb.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % max(1, a.epochs // 5) == 0 or ep == 1:
            log(f"    step {ep:5d}  train loss {loss.item():.4f}  ({time.time() - t0:.0f} s)")

    net.eval()
    with torch.no_grad():
        pred_unet = torch.softmax(net(X), 1)[0].numpy()

    # ── scores (test cells only count) ──────────────────────────────────────
    log("\n  TEST cells (held-out blocks) — lower MAE is better; channel = river + ice")
    res = {}
    for nm, p in (("mean", pred_mean), ("rf", pred_rf), ("unet", pred_unet)):
        res[nm] = {"test": score(p, t, te, channel), "train": score(p, t, tr, channel)}
        show(log, nm, res[nm]["test"])
    log("\n  TRAIN cells (for overfitting check only)")
    for nm in ("mean", "rf", "unet"):
        show(log, nm, res[nm]["train"])

    # ── save ────────────────────────────────────────────────────────────────
    torch.save(net.state_dict(), os.path.join(run, "model.pt"))
    names = [f"f_{c}" for c in CLASS_NAMES] + ["f_channel"]
    pp = os.path.join(run, f"pred_{a.date}.tif")
    write_tif(pp, list(pred_unet) + [pred_unet[0] + pred_unet[1]], names)
    open(pp[:-4] + ".qml", "w", encoding="utf-8").write(PRED_QML)
    sp = os.path.join(run, f"split_{a.date}.tif")
    write_tif(sp, [tr.astype(np.uint8) + 2 * te.astype(np.uint8)], ["split"], dtype="uint8")
    open(sp[:-4] + ".qml", "w", encoding="utf-8").write(SPLIT_QML)
    for d in a.predict_dates:
        try:
            with torch.no_grad():
                p = torch.softmax(net(torch.from_numpy(viirs(d))[None]), 1)[0].numpy()
        except Exception as e:                           # missing patch etc.
            log(f"  could not predict {d}: {e}")
            continue
        op = os.path.join(run, f"pred_{d}.tif")
        write_tif(op, list(p) + [p[0] + p[1]], names)
        open(op[:-4] + ".qml", "w", encoding="utf-8").write(PRED_QML)
        log(f"  predicted {d} (no labels there — look at it, do not score it)")
    json.dump(vars(a), open(os.path.join(run, "config.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(run, "metrics.json"), "w"), indent=1)
    log(f"\n  saved in {os.path.relpath(run, NR_ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="U-Net pipeline test on VIIRS.")
    p.add_argument("--date", default="2023-10-04")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--max-unsure", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--predict-dates", nargs="*", default=["2023-04-19", "2024-04-21"])
    main(p.parse_args())
