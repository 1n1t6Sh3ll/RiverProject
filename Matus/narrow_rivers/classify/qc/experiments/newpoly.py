import pandas as pd, numpy as np, rasterio, glob
d=pd.read_csv('narrow_rivers/classify/samples_2024-06-22.csv')
n=d[d.poly_id>=int(__import__("os").environ.get("FROM","106"))]
p='narrow_rivers/patches/nenana_20240622/'
cls=rasterio.open(p+'rf/2024-06-22_rf_class.tif').read(1)
conf=rasterio.open(p+'rf/2024-06-22_rf_confidence.tif').read(1)
jf=glob.glob(p+'*jrc*occ*.tif')+glob.glob(p+'*occurrence*.tif')
jrc=rasterio.open(jf[0]).read(1) if jf else None
print('jrc file',jf[:1])
n=n.assign(pred=cls[n.row,n.col],conf=conf[n.row,n.col])
if jrc is not None: n=n.assign(jrc=jrc[n.row,n.col])
g=n.groupby('poly_id').agg(cls=('class_id','first'),n=('class_id','size'),
  wrong=('pred',lambda s: None),ndwi=('ndwi','median'),B6=('B6','median'),B5=('B5','median'),conf=('conf','median'),
  **({'jrc':('jrc','median')} if jrc is not None else {}))
g['wrong']=n.groupby('poly_id').apply(lambda x:(x.pred!=x.class_id).mean()).round(3)
print(g.round(3).to_string())
print('pred codes',np.unique(cls,return_counts=True))
