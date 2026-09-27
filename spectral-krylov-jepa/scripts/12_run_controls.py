#!/usr/bin/env python3
"""Frozen supplementary controls; verification scale is never scientific evidence."""
from __future__ import annotations
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spectral_krylov_jepa.data.generate_labeled import generate_labeled_dataset
from spectral_krylov_jepa.data.generate_unlabeled import generate_unlabeled_dataset
from spectral_krylov_jepa.evaluation.evaluate import evaluate_checkpoint
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.training.pretrain import pretrain
from spectral_krylov_jepa.training.finetune import finetune
from spectral_krylov_jepa.utils.io import write_json
from spectral_krylov_jepa.utils.provenance import freeze_or_check, source_hashes, sha256


def variants(steps, dim):
    return {
        'scratch': {'method': 'scratch'},
        'field': {'method': 'field'},
        'operator': {'method': 'operator'},
        'k1': {'context_steps': 1},
        'k2': {},
        'k3': {'context_steps': 3},
        'shuffle': {'data_variant': 'shuffle'},
        'remove_v': {'remove_v': True},
        'coeff': {'lambda_coeff': 1e-4},
        'ema4': {'target_update_frequency': 4},
        'powers': {'data_variant': 'powers'},
        'no_latent_norm': {'normalize_latents': False},
        'half_steps': {'max_steps': max(1, steps // 2)},
        'half_dimension': {'encoder_overrides': {'embed_dim': dim // 2}},
    }


def transform_corpus(source, dest, kind):
    """Deterministic donor derangement or normalized powers; never alter source."""
    import shutil
    shutil.copyfile(source, dest)
    with h5py.File(source, 'r') as original, h5py.File(dest, 'r+') as out:
        n = len(original['potential'])
        if kind == 'shuffle':
            # Cyclic offset: every trajectory has a different potential donor.
            for i in range(n):
                for key in ('q', 'alpha', 'beta'):
                    out[key][i] = original[key][(i + 1) % n]
            out.attrs['control'] = 'cyclic donor derangement; metadata refers to original corpus'
        else:
            grid = GridSpec(int(original.attrs['ny']))
            for i in range(n):
                ham = build_hamiltonian(grid, original['potential'][i])
                q = original['q'][i]
                for st in range(q.shape[0]):
                    for j in range(1, q.shape[1]):
                        hq = ham.matvec(q[st, j-1])
                        q[st, j] = hq / np.linalg.norm(hq)
                out['q'][i] = q
            out.attrs['control'] = 'normalized powers; stored coefficients unused'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', choices=['verification', 'full'], default='verification')
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    full = args.scale == 'full'
    cfg = dict(img_size=32 if full else 8, model_size='default' if full else 'smoke',
               device='auto' if full else 'cpu', batch_size=8 if full else 2,
               max_steps=2000 if full else 4, context_steps=2, num_workers=0)
    seeds = [11, 23, 47] if full else [11]
    matrix = variants(cfg['max_steps'], 128 if full else 64)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    identity = {'scale': args.scale, 'config': cfg, 'variants': matrix, 'seeds': seeds,
                'sources': source_hashes(ROOT), 'evidence_class': 'supplementary' if full else 'pipeline_only'}
    freeze_or_check(run / 'identity.json', identity)
    lock = run / 'RUNNING.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        execute(run, cfg, matrix, seeds, full)
    finally:
        lock.unlink()  # only the lock acquired by this invocation


def execute(run, cfg, matrix, seeds, full):
    grid = GridSpec(cfg['img_size'])
    if not (run / 'data.done').exists():
        generate_unlabeled_dataset(output_path=run/'unlabeled.h5', grid=grid,
            n_examples=1000 if full else 8, n_starts=2 if full else 1, depth=3,
            base_seed=10000 if full else 60000)
        generate_labeled_dataset(output_path=run/'labeled.h5', grid=grid,
            n_train=120 if full else 6, n_val=40 if full else 2,
            n_test_id=40 if full else 2, n_ood_each=40 if full else 2,
            subset_sizes=[10,25,50,100] if full else [4], split_seed=2026,
            id_base_seed=20000 if full else 70000, ood_base_seed=30000 if full else 80000,
            manifest_name='splits', manifest_directory=run)
        for kind in ['shuffle', 'powers']:
            transform_corpus(run/'unlabeled.h5', run/f'{kind}.h5', kind)
        (run/'data.done').write_text('ok\n')
    freeze_or_check(run/'data_hashes.json', {p.name: sha256(p) for p in
        [run/'unlabeled.h5', run/'labeled.h5', run/'splits.json', run/'shuffle.h5', run/'powers.h5']})
    rows = []
    for name, override in matrix.items():
        pre_cfg = dict(cfg, **override)
        method = pre_cfg.pop('method', 'krylov')
        data_variant = pre_cfg.pop('data_variant', 'unlabeled')
        for seed in seeds:
            pre_cfg['seed'] = seed  # independent pretraining as well as downstream seeds
            pre_dir = run/f'pre_{name}_s{seed}'
            encoder = None
            if method != 'scratch':
                if not (pre_dir/'status.json').exists():
                    pretrain(method=method, data_path=run/f'{data_variant}.h5', run_dir=pre_dir, config=pre_cfg)
                encoder = pre_dir/'encoder.pt'
            for n in ([10,25,50,100] if full else [4]):
                ft_cfg = dict(cfg, seed=seed, epochs=80 if full else 2,
                    early_stopping_patience=15, lambda_psi=1., lambda_e=1., lambda_psi_mse=.1,
                    encoder_overrides=pre_cfg.get('encoder_overrides', {}))
                ft_dir = run/f'ft_{name}_n{n}_s{seed}'
                if not (ft_dir/'metrics.json').exists():
                    finetune(data_path=run/'labeled.h5', manifest_path=run/'splits.json',
                        subset=f'n{n}', encoder_path=encoder, method_name=name, run_dir=ft_dir, config=ft_cfg)
                for split in ['test_ID','test_OOD_narrow','test_OOD_strong','test_OOD_double']:
                    ev_dir = run/f'eval_{name}_n{n}_s{seed}'/split
                    if not (ev_dir/'metrics.json').exists():
                        evaluate_checkpoint(checkpoint=ft_dir/'checkpoint_best.pt',
                            data_path=run/'labeled.h5', manifest_path=run/'splits.json', split=split,
                            config=ft_cfg, output_dir=ev_dir, n_bootstrap=2000 if full else 50)
                    ev = json.loads((ev_dir/'metrics.json').read_text())
                    rows.append({'variant':name,'seed':seed,'n_labels':n,'split':split,
                                 'metrics':str((ev_dir/'metrics.json').relative_to(run)),
                                 'fidelity':ev['summary']['fidelity_mean']})
        write_json({'complete':False,'rows':rows}, run/'progress.json')
    write_json({'complete':True,'evidence_class':'supplementary' if full else 'pipeline_only',
                'rows':rows,'expected_rows':len(matrix)*len(seeds)*(4 if full else 1)*4,
                'process_peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'peak_rss_unit':'bytes on macOS; KiB on Linux; whole process, not per model'}, run/'summary.json')

if __name__ == '__main__':
    main()
