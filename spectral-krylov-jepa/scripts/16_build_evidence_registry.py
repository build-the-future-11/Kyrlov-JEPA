#!/usr/bin/env python3
"""Bind reported claims and artifacts to actual files, without a release upgrade."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from spectral_krylov_jepa.utils.provenance import sha256, source_hashes
from spectral_krylov_jepa.utils.io import write_json, environment_info


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    base=ROOT/'results/astra_baselines_20260927/baseline_audit.json'
    audit=json.loads(base.read_text())
    wanted={('free_box','no_labels'):['fidelity_mean','residual_true_e_mean'],
            ('sine_ritz_9','no_labels'):['fidelity_mean','residual_true_e_mean'],
            ('linear_ridge','n20'):['fidelity_mean','residual_true_e_mean']}
    metrics=[]
    for r in audit['summaries']:
        if r['split']=='test_ID' and (r['method'],r['subset']) in wanted:
            for k in wanted[r['method'],r['subset']]:metrics.append((r['method']+' '+k,r[k]))
    historical=ROOT/'experiments/summaries/smoke_latest.json'
    old=json.loads(historical.read_text())['steps']
    for method in ['scratch','krylov']:
        for key in ['fidelity_mean','residual_true_e_mean']:
            metrics.append((method+' '+key,old['eval_'+method+'_n20'][key]))
    papers=[ROOT/'paper/TECHNICAL_REPORT.md',ROOT/'paper/submission/CJSJ_DRAFT.md']
    for paper in papers:
        content=paper.read_text()
        for name,value in metrics:
            if f'{value:.6f}' not in content:raise ValueError(f'{paper.name}: missing metric {name}')
    files=[base,historical,*papers,*ROOT.glob('results/astra_baselines_20260927/*'),
           *ROOT.glob('results/astra_controls_20260927/*'),*ROOT.glob('paper/submission/*.docx'),
           *ROOT.glob('paper/submission/*.pptx'),ROOT/'paper/TECHNICAL_REPORT.pdf']
    unique={str(p.relative_to(ROOT)):sha256(p) for p in files if p.is_file()}
    write_json(dict(environment=environment_info(),source_hashes=source_hashes(ROOT),
        artifacts=unique,verified_manuscript_metrics=dict(metrics),
        engineering='bounded workflows tested; consult test receipt',
        science='negative historical smoke audit; full efficacy untested',
        release='BLOCKED',paper='DRAFT',submission='BLOCKED',independent_reproduction='NOT_RUN'),a.output)
    print(f'Bound {len(unique)} artifacts and {len(metrics)} manuscript metrics')
if __name__=='__main__':main()
