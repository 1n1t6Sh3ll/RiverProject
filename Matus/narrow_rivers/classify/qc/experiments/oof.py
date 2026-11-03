import sys, os, numpy as np
CL = r"C:\Users\matus\dev\UG Research Assistant\RiverProject\Matus\narrow_rivers\classify"
sys.path.insert(0, CL); os.chdir(CL)
import train_rf as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
import csv
NAMES = {1:"river",2:"ice",3:"snow",4:"land"}
for date in sys.argv[1:]:
    print(f"\n######## {date}")
    rows = [r for r in csv.DictReader(open(f"samples_{date}.csv")) if int(r["class_id"]) in NAMES]
    X = np.array([[float(r[f]) for f in T.EX.FEATURES] for r in rows], np.float32)
    y = np.array([int(r["class_id"]) for r in rows]); g = np.array([int(r["poly_id"]) for r in rows])
    ndwi = X[:, T.EX.FEATURES.index("ndwi")]
    oof = np.zeros_like(y)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
        rf = RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        oof[te] = rf.predict(X[te])
    cls = sorted(set(y))
    print("n =", len(y), " polys =", len(set(g)))
    print("true\pred " + " ".join(f"{NAMES[c]:>7}" for c in cls))
    for a in cls:
        print(f"{NAMES[a]:9} " + " ".join(f"{((y==a)&(oof==b)).sum():7d}" for b in cls)
              + f"   recall {((y==a)&(oof==a)).sum()/(y==a).sum():.3f}  prec {((y==a)&(oof==a)).sum()/max((oof==a).sum(),1):.3f}")
    for a in cls:
        err = (y==a)&(oof!=a)
        if not err.any(): continue
        ps = {}
        for p in set(g[y==a]):
            m = g==p; ps[p] = (m.sum(), (err&m).sum())
        tot = err.sum(); worst = sorted(ps.items(), key=lambda kv:-kv[1][1])
        top5 = sum(v[1] for _,v in worst[:5])
        print(f"\n {NAMES[a]} errors {tot}: worst 5 of {len(ps)} polygons hold {top5/tot:.0%}")
        for p,(n,e) in worst[:6]:
            wrong = oof[(g==p)&err]; to = {NAMES[c]:int((wrong==c).sum()) for c in set(wrong)}
            print(f"   poly {p:4d}: {e:5d}/{n:5d} wrong ({e/n:.0%})  -> {to}  median NDWI wrong px {np.median(ndwi[(g==p)&err]):+.2f}")
        small = [p for p,(n,e) in ps.items() if n<20]; big=[p for p,(n,e) in ps.items() if n>=20]
        rec = lambda P: 1 - sum(ps[p][1] for p in P)/max(sum(ps[p][0] for p in P),1)
        print(f"   recall in polygons <20 px: {rec(small):.2f} ({len(small)} polys)   >=20 px: {rec(big):.2f} ({len(big)} polys)")
    if date == "2024-06-22":
        rv = y==1
        print(f"\n river px NDWI: correct median {np.median(ndwi[rv&(oof==1)]):+.2f}, missed median {np.median(ndwi[rv&(oof!=1)]):+.2f}")
        fp = (y==4)&(oof==1)
        print(f" land called river: {fp.sum()}  median NDWI {np.median(ndwi[fp]) if fp.any() else float('nan'):+.2f}")
