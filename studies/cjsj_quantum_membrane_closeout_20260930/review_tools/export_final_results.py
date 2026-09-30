"""Export a canonical comparison table from preserved per-case measurements.

Never writes under results/. Outputs derived files to an explicit new directory.
Archived timing observations, not verification-replay timings, are claim bearing.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

def build(root):
    rows=[]
    for phase,folder in [('initial',root/'results'),('reference_sensitivity',root/'results/reference_sensitivity')]:
        for c in json.loads((folder/'comparisons.json').read_text()):
            domain=c.get('domain','membrane')
            f=folder/('test.csv' if phase!='initial' else f'{domain}_test.csv')
            d=pd.read_csv(f);d=d[(d.n==c['n'])&(d.family==c['family'])]
            a=d[d.method==c['numerator']].copy();b=d[d.method==c['denominator']].copy()
            keys=['potential_seed']+(['training_seed'] if domain=='quantum' else [])
            a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
            if not a.index.equals(b.index) or a.index.has_duplicates:raise ValueError('Unpaired or duplicate observations')
            av=a.capped_seconds.to_numpy();bv=b.capped_seconds.to_numpy()
            ratio=float(av.mean()/bv.mean())
            if not np.isclose(ratio,c['ratio'],rtol=1e-12):raise ValueError('Stored ratio disagrees with measurements')
            rows.append(dict(study='CJSJ-QM-20260930',phase=phase,domain=domain,grid=c['n'],family=c['family'],numerator=c['numerator'],baseline=c['denominator'],primary=c['primary'],independent_physical_cases=d.potential_seed.nunique(),training_seeds=a.reset_index().training_seed.nunique() if domain=='quantum' else 0,numerator_mean_ms=1000*av.mean(),baseline_mean_ms=1000*bv.mean(),absolute_difference_ms=1000*(av-bv).mean(),time_ratio=ratio,relative_difference_pct=100*(ratio-1),ci_coverage=c['coverage'],ci_low=c['ci'][0],ci_high=c['ci'][1],bootstrap_draws=c['draws'],source=str(f.relative_to(root)),source_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    t=build(a.root);t.to_csv(a.output/'final_results.csv',index=False)
    (a.output/'README.md').write_text('# Canonical comparison table\n\n64 comparisons from the retained initial and reference-sensitivity studies. CIs are paired bootstrap intervals; grid resolutions reuse physical cases. Timing repetitions and model seeds do not multiply independent physical sample counts. Primary interval coverage differs by study. Secondary intervals are descriptive and not jointly adjusted.\n')
    print(json.dumps({'status':'PASS','comparisons':len(t),'output':str(a.output/'final_results.csv')}))
if __name__=='__main__':main()
