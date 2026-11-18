"""Record group-fold assignments for platform diagnostics without fitting any model."""
import json,hashlib,platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
from chem_repro.heracleum.data import load
frame,_=load('published_snapshot');groups=frame.generic_murcko_scaffold.fillna('none').to_numpy();folds=np.full(len(frame),-1,int)
for fold,(_,test) in enumerate(GroupKFold(5).split(np.zeros((len(frame),1)),groups=groups)):folds[test]=fold
rows=[{'observation_id':str(i),'group':str(g),'fold':int(f)} for i,g,f in zip(frame.synID,groups,folds)]
result={'platform':platform.platform(),'machine':platform.machine(),'numpy':np.__version__,'sklearn':sklearn.__version__,'rows':rows,'fold_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
from chem_repro.heracleum.cv_registry import reference_fold_assignments
reference=reference_fold_assignments(frame.synID.astype(str).tolist(),groups,'published_snapshot')['scaffold']
reference_rows=[{**row,'fold':int(fold)} for row,fold in zip(rows,reference)]
result['reference_fold_sha256']=hashlib.sha256(json.dumps(reference_rows,sort_keys=True).encode()).hexdigest()
result['reference_rows']=reference_rows
out=Path('validation/generated');out.mkdir(parents=True,exist_ok=True);(out/'cv_folds.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['rows','reference_rows']})
