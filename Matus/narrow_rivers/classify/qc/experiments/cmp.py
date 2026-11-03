import pandas as pd, sys
a=pd.read_csv(sys.argv[1]); b=pd.read_csv(sys.argv[2])
print(a.columns.tolist()[:12])
ca=a.groupby('poly_id').agg(n=('poly_id','size'),c=('class_id','first'))
cb=b.groupby('poly_id').agg(n=('poly_id','size'),c=('class_id','first'))
j=ca.join(cb,lsuffix='_old',rsuffix='_new',how='outer')
ch=j[(j.n_old!=j.n_new)|(j.c_old!=j.c_new)]
print("old by class",a.class_id.value_counts().to_dict(),"new",b.class_id.value_counts().to_dict())
print("changed/new polygons:");print(ch.to_string())
new=b[b.poly_id.isin(ch.index)]
print("ndwi of new px by class:");print(new.groupby('class_id').ndwi.describe())
