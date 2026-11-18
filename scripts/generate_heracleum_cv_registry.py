"""Freeze reference implementation fold assignments; no targets or predictions are used."""
import csv,json,hashlib,platform,argparse
from pathlib import Path
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold,KFold
from chem_repro.heracleum.data import load
from chem_repro.common.chemistry import scaffold
from chem_repro.common.datasets import DATA_ROOT
root=DATA_ROOT/'heracleum';rows=[];cohorts={}
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replace-reference',action='store_true');args=parser.parse_args()
if (root/'cv_fold_registry.csv').exists() and not args.replace_reference:
    raise SystemExit('Reference registry exists. Use --replace-reference only for an explicitly documented new reference run.')
def fingerprint(ids,groups):
 codes={};membership=[codes.setdefault(str(g),len(codes)) for g in groups]
 return hashlib.sha256(json.dumps(list(zip(ids,membership)),separators=(',',':')).encode()).hexdigest()
for policy in ['published_snapshot','strict']:
 f,_=load(policy);ids=f.synID.astype(str).tolist();groups=(f.generic_murcko_scaffold.fillna('none').to_numpy() if policy=='published_snapshot' else f.canonical_smiles.map(lambda s:scaffold(s,generic=True,retain_stereo=True)).to_numpy())
 cohorts[policy]={'records':len(ids),'group_membership_sha256':fingerprint(ids,groups)};x=np.zeros((len(ids),1))
 for scheme,partitions in [('random',KFold(5,shuffle=True,random_state=42).split(x)),('scaffold',GroupKFold(5).split(x,groups=groups))]:
  assignments=np.full(len(ids),-1,int)
  for fold,(_,test) in enumerate(partitions):assignments[test]=fold
  assert (assignments>=0).all()
  rows.extend({'identity_policy':policy,'scheme':scheme,'observation_id':i,'fold':int(f)} for i,f in zip(ids,assignments))
path=root/'cv_fold_registry.csv'
with path.open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=['identity_policy','scheme','observation_id','fold'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
manifest=json.loads((root/'manifest.json').read_text())
meta={'origin':'reference implementation execution; not author-supplied fold assignments','reason':'GroupKFold equal-size group tie ordering differs across NumPy platforms; fixed partitions make numeric comparisons portable.','generator':{'python':platform.python_version(),'system':platform.system(),'machine':platform.machine(),'numpy':np.__version__,'scikit-learn':sklearn.__version__},'seed':42,'n_splits':5,'cohorts':cohorts,'input_sha256':{name:manifest['files'][name]['sha256'] for name in ['DS1-DS2_df.csv','DS1_2008.csv','DS2_102.csv']},'targets_or_predictions_used':False}
(root/'cv_fold_registry_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
for p in [path,root/'cv_fold_registry_metadata.json']:
 manifest['files'][p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_url':'https://github.com/chemagents/chem-repro-suite','revision':'reference CV registry v1','parent_sha256':manifest['files']['DS1-DS2_df.csv']['sha256'],'derivation':meta['origin']+'. '+meta['reason']+' Generator versions and cohort/input hashes are in cv_fold_registry_metadata.json.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({'rows':len(rows),'cohorts':cohorts})
