#!/usr/bin/env python3
"""Verify control-matrix completeness, then regenerate paired effects and costs."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spectral_krylov_jepa.evaluation.bootstrap import paired_bootstrap_ci
from spectral_krylov_jepa.utils.io import write_json
from spectral_krylov_jepa.utils.provenance import sha256


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    run=a.run_dir.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((run/'identity.json').read_text()); summary=json.loads((run/'summary.json').read_text())
    splits=['test_ID','test_OOD_narrow','test_OOD_strong','test_OOD_double']
    sizes=[10,25,50,100] if identity['scale']=='full' else [4]
    expected={(v,s,n,t) for v in identity['variants'] for s in identity['seeds'] for n in sizes for t in splits}
    actual=[(r['variant'],r['seed'],r['n_labels'],r['split']) for r in summary['rows']]
    if not summary['complete'] or set(actual)!=expected or len(actual)!=len(expected):
        raise ValueError('Incomplete or duplicate control cells; no outcome analysis allowed')
    for name,digest in json.loads((run/'data_hashes.json').read_text()).items():
        if sha256(run/name)!=digest: raise ValueError(f'Changed data: {name}')
    evaluations={}; hashes={}
    for row in summary['rows']:
        p=run/row['metrics']; hashes[row['metrics']]=sha256(p); ev=json.loads(p.read_text())
        data={r['index']:r['fidelity'] for r in ev['per_example']}
        if len(data)!=ev['n'] or not np.isfinite(list(data.values())).all():
            raise ValueError(f'Invalid evaluation: {p}')
        evaluations[(row['variant'],row['seed'],row['n_labels'],row['split'])]=data
    comparisons=[]
    for name in identity['variants']:
        if name=='k2': continue
        for n in sizes:
            for split in splits:
                diffs=[]
                for seed in identity['seeds']:
                    k=evaluations[('k2',seed,n,split)]; b=evaluations[(name,seed,n,split)]
                    if k.keys()!=b.keys(): raise ValueError('Pairing mismatch')
                    diffs.append([k[i]-b[i] for i in sorted(k)])
                arr=np.asarray(diffs)
                comparisons.append(dict(comparator=name,n_labels=n,split=split,
                    mean_per_seed=arr.mean(1).tolist(),
                    ci_interpretation='paired potentials after averaging specified seeds; excludes new-training-seed uncertainty; unadjusted exploratory intervals',
                    **paired_bootstrap_ci(arr.mean(0),n_resamples=2000,seed=2026)))
    costs=[]
    for p in sorted(run.glob('pre_*/metrics.json')):
        m=json.loads(p.read_text()); costs.append(dict(run=p.parent.name,**{k:m[k] for k in
            ['steps','elapsed_sec','n_params_total','n_params_trainable','encoder_parameters']}))
    write_json(dict(evidence_class=summary['evidence_class'],verified_cells=len(actual),
        comparisons=comparisons,costs=costs,artifact_hashes=hashes,
        verdict='NO_EFFICACY_INFERENCE_FROM_VERIFICATION_SCALE' if identity['scale']=='verification' else 'DESCRIPTIVE_SUPPLEMENT_REQUIRES_REVIEW'),out/'analysis.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for p in sorted(run.glob('pre_*/metrics.json')):
        m=json.loads(p.read_text()); ax.plot(range(1,len(m['loss_history'])+1),m['loss_history'],label=p.parent.name.replace('pre_',''))
    ax.set_xlabel('Training step'); ax.set_ylabel('Training loss (objectives differ)')
    ax.set_title('Control pipeline convergence check — not a performance ranking')
    ax.legend(fontsize=6,ncol=3); fig.tight_layout(); fig.savefig(out/'convergence.png',dpi=160); plt.close(fig)
    print(f'Verified {len(actual)} cells; {len(comparisons)} descriptive comparisons')
if __name__=='__main__':main()
