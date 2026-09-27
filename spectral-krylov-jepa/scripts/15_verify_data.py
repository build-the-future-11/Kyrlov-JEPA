#!/usr/bin/env python3
"""Audit corpus-level duplicates, disjoint splits, and nested labeled subsets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spectral_krylov_jepa.utils.provenance import sha256
from spectral_krylov_jepa.utils.io import write_json


def audit(run, manifest_name):
    man=json.loads((run/manifest_name).read_text())
    with h5py.File(run/'labeled.h5','r') as f:
        v=f['potential'][:]
        if not all(np.isfinite(f[k][:]).all() for k in ['potential','psi0','energy','residual_rel']):
            raise ValueError('Nonfinite labeled data')
        if np.max(f['residual_rel'][:])>1e-5: raise ValueError('Rejected labels')
    digest=lambda a:hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
    hashes=[digest(x) for x in v]
    if len(set(hashes))!=len(hashes): raise ValueError('Duplicate labeled potential')
    seen=set()
    for split,ids in man['splits'].items():
        if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Split leakage')
        seen.update(ids)
    if seen!=set(range(len(v))):raise ValueError('Incomplete corpus split coverage')
    previous=set()
    for name,ids in sorted(man['subsets'].items(),key=lambda item:int(item[0][1:])):
        if len(ids)!=int(name[1:]) or not previous<=set(ids)<=set(man['splits']['train_pool']):
            raise ValueError('Invalid nested subsets')
        previous=set(ids)
    with h5py.File(run/'unlabeled.h5','r') as f:
        unlab={digest(x) for x in f['potential'][:]}
        if len(unlab)!=len(f['potential']):raise ValueError('Duplicate unlabeled potentials')
    if unlab.intersection(hashes):raise ValueError('Pretraining overlaps labeled corpus')
    return dict(status='PASS',labeled_count=len(v),unlabeled_count=len(unlab),
        split_counts={k:len(v) for k,v in man['splits'].items()},
        hashes={name:sha256(run/name) for name in ['labeled.h5','unlabeled.h5',manifest_name]})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--manifest',default='labeled.manifest.json');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();result=audit(a.run_dir,a.manifest);write_json(result,a.output);print(json.dumps(result))
