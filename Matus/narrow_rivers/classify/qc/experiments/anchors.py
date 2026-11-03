import sys, numpy as np
exec(open(sys.argv[0].replace('anchors.py','transfer_anchored.py')).read().split("def load")[0])
import glob,re
print(f"{'date':11}"+"".join(f"{f:>8}" for f in EX.FEATURES)+"   sun elev")
for d in TARGETS:
    P=patch(d)
    with rasterio.open(f"{P}/{d}_sr.tif") as s: sr=s.read().astype(np.float32); tags=s.tags()
    th=rasterio.open(f"{P}/{d}_thermal.tif").read(1).astype(np.float32)
    F=EX.features(sr,th); ok=np.isfinite(F).all(0)
    med=[np.median(F[i][ok&dry]) for i in range(len(F))]
    se=next((v for k,v in tags.items() if 'SUN_ELEV' in k.upper()),'?')
    print(f"{d:11}"+"".join(f"{m:8.3f}" for m in med)+f"   {se}")
