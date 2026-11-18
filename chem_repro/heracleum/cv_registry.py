"""Reference implementation splits, explicitly distinct from author-supplied folds."""
import csv
import hashlib
import json
import numpy as np
from ..common.datasets import DATA_ROOT,verified_manifest


def reference_fold_assignments(ids, groups, policy):
    manifest=verified_manifest('heracleum');root=DATA_ROOT/'heracleum'
    metadata=json.loads((root/'cv_fold_registry_metadata.json').read_text())
    for name,expected in metadata['input_sha256'].items():
        if manifest['files'][name]['sha256']!=expected:raise ValueError('CV registry input changed')
    codes={};membership=[codes.setdefault(str(g),len(codes)) for g in groups]
    digest=hashlib.sha256(json.dumps(list(zip(ids,membership)),separators=(',',':')).encode()).hexdigest()
    if digest!=metadata['cohorts'][policy]['group_membership_sha256']:
        raise ValueError('CV registry cohort or group membership changed')
    with (root/'cv_fold_registry.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['identity_policy']==policy]
    result={}
    for scheme in ['random','scaffold']:
        selected=[r for r in rows if r['scheme']==scheme]
        if [r['observation_id'] for r in selected]!=list(ids):raise ValueError('CV registry record order changed')
        result[scheme]=np.asarray([int(r['fold']) for r in selected],dtype=int)
    return result
