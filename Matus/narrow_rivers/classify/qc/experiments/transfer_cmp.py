import sys, os, numpy as np, pandas as pd
sys.argv=[sys.argv[0]]; exec(open(sys.argv[0].replace('transfer_cmp.py','transfer_anchored.py')).read().split("rows = [r")[0])
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
print(f"{'date':11} {'model':5} {'km2':>6} {'land rec':>8} {'stable riv':>10} {'JRC>=90 hit':>11} {'p117 rec':>8}")
for d in TARGETS:
    F,ok,_=load(d,keep)
    for k,rf in models.items():
        cls=np.full((H,W),9,np.uint8); cls[ok]=rf.predict(F[:,ok].T)
        c=cls[oR,oC]
        print(f"{d:11} {k:5} {(cls==1).sum()*PX_M2/1e6:6.2f} {(c[ld_m]==4).mean():8.3f} {(c[sr_m]==1).mean():10.3f} "
              f"{(cls[(occ>=90)&(occ!=255)&ok]==1).mean():11.1%} {(cls[r117.row.values,r117['col'].values]==1).mean():8.3f}")
