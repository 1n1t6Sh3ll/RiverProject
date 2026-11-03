import sys, os, numpy as np, pandas as pd
sys.argv=[sys.argv[0]]; exec(open(sys.argv[0].replace('gain.py','transfer_anchored.py')).read().split("rows = [r")[0])
from sklearn.ensemble import RandomForestClassifier
old=pd.read_csv(os.path.join(os.path.dirname(__file__),'samples_2024-06-22_v1.csv')); new=pd.read_csv(f"{CL}/samples_{SRC}.csv")
old=old[old.class_id.isin([1,4])]; new=new[new.class_id.isin([1,4])]
keep=list(range(len(EX.FEATURES))); Fs,_,_=load(SRC,keep)
def fit(d): return RandomForestClassifier(300,min_samples_leaf=5,n_jobs=-1,random_state=0).fit(Fs[:,d.row.values,d['col'].values].T,d.class_id.values)
v2=pd.read_csv(os.path.join(os.path.dirname(__file__),'samples_2024-06-22_v2.csv')); v2=v2[v2.class_id.isin([1,4])]
models={'v1':fit(old),'v2':fit(v2),'v3':fit(new)}
oR,oC=old.row.values,old['col'].values; oo=occ[oR,oC]
sr_m=(old.class_id.values==1)&(oo>=80)&(oo!=255); ld_m=old.class_id.values==4
r117=new[new.poly_id==117]
from scipy import ndimage
for d in ["2024-06-22","2025-07-04"]:
    F,ok,_=load(d,keep); m={}
    for k,rf in models.items():
        c=np.zeros((H,W),bool); c[ok]=rf.predict(F[:,ok].T)==1; m[k]=c
    add=m['v3']&~m['v1']; lost=m['v1']&~m['v3']; o=occ[add]
    near=ndimage.binary_dilation(m['v1'],iterations=2)
    print(f"{d}: added {add.sum()*PX_M2/1e6:.2f} km2, lost {lost.sum()*PX_M2/1e6:.2f}; added px JRC>0 {(o>0).mean():.0%}, JRC>=50 {((o>=50)&(o!=255)).mean():.0%}, within 2 px of v1 river {near[add].mean():.0%}")
