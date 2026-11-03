import sys, os, numpy as np, pandas as pd
CL = r"C:\Users\matus\dev\UG Research Assistant\RiverProject\Matus\narrow_rivers\classify"
sys.path.insert(0, CL); os.chdir(CL)
import train_rf as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
F=T.EX.FEATURES; SP=os.path.dirname(os.path.abspath(__file__))
sets={'v1 orig':pd.read_csv(SP+'/samples_2024-06-22_v1.csv'),'v2 +117':pd.read_csv(SP+'/samples_2024-06-22_v2.csv'),'v3 +121-124':pd.read_csv('samples_2024-06-22.csv')}
sets={k:d[d.class_id.isin([1,4])].reset_index(drop=True) for k,d in sets.items()}
ref=sets['v1 orig']; Xr=ref[F].values.astype(np.float32); yr=ref.class_id.values; gr=ref.poly_id.values
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(Xr,yr,gr))
new=sets['v3 +121-124']; NEWR=[117,121,122,123,124]
def rec(y,p,c): return ((y==c)&(p==c)).sum()/(y==c).sum()
def prec(y,p,c): return ((y==c)&(p==c)).sum()/max((p==c).sum(),1)
print(f"{'labels':12} {'old-poly riv rec':>16} {'prec':>6} {'land rec':>8} | wide-channel polys held out (leave-one-out) rec")
for k,d in sets.items():
    X=d[F].values.astype(np.float32); y=d.class_id.values; g=d.poly_id.values
    o=np.zeros_like(yr)
    for tr,te in folds:
        trm=~np.isin(g,np.unique(gr[te]))
        o[te]=RandomForestClassifier(300,min_samples_leaf=5,n_jobs=-1,random_state=0).fit(X[trm],y[trm]).predict(Xr[te])
    # leave-one-polygon-out for each wide-channel polygon (evaluated on v3 geometry)
    loo=[]
    for p in NEWR:
        trm=g!=p
        te=new[new.poly_id==p]
        pr=RandomForestClassifier(300,min_samples_leaf=5,n_jobs=-1,random_state=0).fit(X[trm],y[trm]).predict(te[F].values.astype(np.float32))
        loo.append(f"{p}:{(pr==1).mean():.2f}")
    print(f"{k:12} {rec(yr,o,1):16.3f} {prec(yr,o,1):6.3f} {rec(yr,o,4):8.3f} | "+"  ".join(loo))
