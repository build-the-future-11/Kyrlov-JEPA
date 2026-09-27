#!/usr/bin/env python3
"""Audit existing smoke corpus against simple baselines; no retraining or retuning."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spectral_krylov_jepa.evaluation.baselines import sine_ritz
from spectral_krylov_jepa.evaluation.metrics import evaluate_example, summarize_metrics
from spectral_krylov_jepa.evaluation.bootstrap import paired_bootstrap_ci
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.utils.io import write_json
from spectral_krylov_jepa.utils.provenance import sha256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    run=args.run_dir.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'labeled.manifest.json').read_text())
    grid=GridSpec.from_dict(manifest['grid']); ca=cell_area(grid)
    with h5py.File(run/'labeled.h5','r') as f:
        v=f['potential'][:]; psi=f['psi0'][:]; energy=f['energy'][:]
    rows=[]; t0=time.perf_counter()
    for split,indices in manifest['splits'].items():
        if not split.startswith('test'): continue
        for i in indices:
            for name,modes in [('free_box',1),('sine_ritz_9',3)]:
                start=time.perf_counter(); ehat,phat=sine_ritz(v[i],grid,modes)
                elapsed=time.perf_counter()-start
                rows.append(dict(method=name,subset='no_labels',split=split,index=i,
                    elapsed_sec=elapsed,**evaluate_example(v[i],psi[i],float(energy[i]),phat,ehat,grid)))
        for subset,train in manifest['subsets'].items():
            vt=v[train].reshape(len(train),-1).astype(float)
            pt=psi[train].reshape(len(train),-1).astype(float)
            # Align signs to first training state only, never a test state.
            pt*=np.where((pt@pt[0])[:,None]<0,-1.,1.)
            y=np.column_stack([pt,energy[train]]); xmean=vt.mean(0); ymean=y.mean(0)
            xc=vt-xmean; coef=np.linalg.solve(xc@xc.T+np.eye(len(train)),y-ymean)
            for i in indices:
                for name in ['train_mean','linear_ridge']:
                    start=time.perf_counter()
                    pred=ymean if name=='train_mean' else (v[i].ravel()-xmean)@xc.T@coef+ymean
                    phat=pred[:-1]; phat=phat/np.sqrt((phat**2).sum()*ca)
                    rows.append(dict(method=name,subset=subset,split=split,index=i,
                        elapsed_sec=time.perf_counter()-start,
                        **evaluate_example(v[i],psi[i],float(energy[i]),phat,float(pred[-1]),grid)))
    groups={}
    for r in rows: groups.setdefault((r['method'],r['subset'],r['split']),[]).append(r)
    summaries=[]
    for (method,subset,split),items in groups.items():
        summaries.append(dict(method=method,subset=subset,split=split,
            **summarize_metrics([{k:v for k,v in r.items() if k not in ['method','subset','split']} for r in items])))
    # Comparisons by explicit sample identity, conditional on the historical one-seed fit.
    comparisons=[]
    for path in sorted(run.glob('eval*/**/metrics.json')):
        ev=json.loads(path.read_text())
        if 'per_example' not in ev or 'split' not in ev: continue
        split=ev['split']; ids=[r['index'] for r in ev['per_example']]
        a=np.array([r['fidelity'] for r in ev['per_example']])
        for base in ['free_box','sine_ritz_9']:
            lookup={r['index']:r['fidelity'] for r in rows if r['method']==base and r['split']==split}
            if not set(ids)<=lookup.keys(): raise ValueError('unmatched test examples')
            b=np.array([lookup[i] for i in ids])
            comparisons.append(dict(neural_metrics=str(path.relative_to(run)),baseline=base,
                interpretation='neural minus baseline; conditional on one fitted model; exploratory',
                **paired_bootstrap_ci(a,b,n_resamples=2000,seed=2026)))
    result=dict(evidence_class='exploratory_historical_smoke_audit',data_sha256=sha256(run/'labeled.h5'),
        manifest_sha256=sha256(run/'labeled.manifest.json'),source_run=str(run),rows=rows,
        summaries=summaries,comparisons=comparisons,elapsed_sec=time.perf_counter()-t0)
    write_json(result,out/'baseline_audit.json')
    with (out/'baseline_summary.csv').open('w',newline='') as f:
        keys=['method','subset','split','n','fidelity_mean','rel_energy_error_mean','residual_true_e_mean','elapsed_sec_mean']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); w.writeheader(); w.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    idrows=[r for r in summaries if r['split']=='test_ID']
    fig,ax=plt.subplots(figsize=(8,4))
    ax.barh([r['method']+' '+r['subset'] for r in idrows],[1-r['fidelity_mean'] for r in idrows])
    ax.set_xlabel('Mean infidelity (lower is better)'); ax.set_title('Simple baselines on historical smoke test-ID (exploratory)')
    fig.tight_layout(); fig.savefig(out/'baseline_infidelity.png',dpi=180); plt.close(fig)
    text=['# Baseline audit — historical smoke only','',
          '| Method | Labels | Split | Fidelity | Relative energy error | Residual at true E |',
          '|---|---|---|---:|---:|---:|']
    for r in summaries:
        text.append(f"| {r['method']} | {r['subset']} | {r['split']} | {r['fidelity_mean']:.6f} | {r['rel_energy_error_mean']:.6f} | {r['residual_true_e_mean']:.6f} |")
    (out/'BASELINE_RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'rows':len(rows),'summaries':len(summaries),'paired_comparisons':len(comparisons)}))
if __name__=='__main__': main()
