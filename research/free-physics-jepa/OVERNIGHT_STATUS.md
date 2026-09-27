# Free-Physics JEPA / LMOP-JEPA — Overnight Status

**Claim gate:** ENGINEERING ONLY (init)

| Phase | Status |
|-------|--------|
| 0 Repository/protocol audit | IN PROGRESS |
| 1 Darcy physics | NOT STARTED |
| 2 Data contract/manifests | NOT STARTED |
| 3 Genuine data | NOT STARTED |
| 4 Manufactured data | NOT STARTED |
| 5 Synthetic→real gap | NOT STARTED |
| 6 Shared backbone | NOT STARTED |
| 7 MML-direct | NOT STARTED |
| 8 LMOP-JEPA | NOT STARTED |
| 9 Shuffled control | NOT STARTED |
| 20 Smoke gate | NOT STARTED |
| 21 Confirmatory matrix | NOT STARTED |

Amendment: `amendments/PROTOCOL_AMENDMENT_0001_initial_freeze.md` (protocol files were absent; frozen from session brief).

Krylov-JEPA is unrelated and not used.

## Run smoke_20260926T150135Z
- mode: smoke
- device: mps
- grid: 32

- physics: PASSED cond≈2.27e+03
- manifests: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/smoke_splits.json sha=907a3be89dc2
- pretrain MML-direct...
  MML done loss=0.4900 params=66417
- pretrain LMOP-JEPA...
  LMOP done loss=0.0116 var=1.538e-04
- shuffled-physics LMOP control...
- finetune scratch_n16_s11
- finetune mml_direct_n16_s11
- finetune lmop_jepa_n16_s11
- finetune scratch_n32_s11
- finetune mml_direct_n32_s11
- finetune lmop_jepa_n32_s11
- table: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/smoke_metrics.csv
- DONE smoke

## Run confirmatory_20260926T150251Z
- mode: confirmatory
- device: mps
- grid: 64

- physics: PASSED cond≈2.2e+04
- manifests: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/confirmatory_splits.json sha=0610c3d24eab
- pretrain MML-direct...
  MML done loss=0.1288 params=1185057
- pretrain LMOP-JEPA...

## Run confirmatory_20260926T150845Z
- mode: confirmatory
- device: mps
- grid: 64

- physics: PASSED cond≈2.2e+04
- data: RESUME existing HDF5
- manifests: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/confirmatory_splits.json sha=0610c3d24eab
- pretrain MML-direct...
  MML reused from prior confirmatory attempt
  MML done loss=0.12881408631801605 params=1185057
- pretrain LMOP-JEPA...
  LMOP done loss=0.8241 std=0.2936771755358204
- finetune scratch_n25_s11
- finetune mml_direct_n25_s11
- finetune lmop_jepa_n25_s11
- finetune scratch_n100_s11
- finetune mml_direct_n100_s11
- finetune lmop_jepa_n100_s11
- finetune scratch_n25_s23
- finetune mml_direct_n25_s23
- finetune lmop_jepa_n25_s23
- finetune scratch_n100_s23
- finetune mml_direct_n100_s23

## Run confirmatory_20260926T150845Z
- mode: confirmatory
- device: mps
- grid: 64
- resume: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_20260926T150845Z
- git_sha: f5ecaa72e8ae9b88820fc2e771c6d5e4bdfc6ceb

- physics: PASSED cond≈2.2e+04
- data: RESUME existing HDF5
- manifests: RESUME /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/manifests/confirmatory_splits.json sha=0610c3d24eab
- pretrain MML-direct...
  mml resume existing
  MML done loss=0.12881408631801605 params=1185057
- pretrain LMOP-JEPA...
  lmop resume existing
  LMOP done loss=0.8241 std=0.2936771755358204
- shuffled-physics LMOP control...
  shuffle loss=0.7795073986053467 vs correct=0.8240561485290527 supported=False
- skip complete scratch_n25_s11
- skip complete mml_direct_n25_s11
- skip complete lmop_jepa_n25_s11
- skip complete scratch_n100_s11
- skip complete mml_direct_n100_s11
- skip complete lmop_jepa_n100_s11
- skip complete scratch_n25_s23
- skip complete mml_direct_n25_s23
- skip complete lmop_jepa_n25_s23
- skip complete scratch_n100_s23
- finetune mml_direct_n100_s23
- finetune lmop_jepa_n100_s23
- finetune scratch_n25_s47
- finetune mml_direct_n25_s47
- finetune lmop_jepa_n25_s47
- finetune scratch_n100_s47
- finetune mml_direct_n100_s47
- finetune lmop_jepa_n100_s47
- table: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/confirmatory.csv
- claim_gate: FALSIFIES_HYPOTHESIS -> /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/claim_gate.json
- claim_detail: MML ≥ LMOP on relative L2 across seeds under this protocol.
- DONE confirmatory

## Run smoke_s2_v2_20260926T162507Z
- mode: smoke
- study: 2 data_version=v2
- amendment: 0003_matched_block_jepa
- device: mps
- grid: 32
- resume: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/smoke_s2_v2_20260926T162507Z
- git_sha: f5ecaa72e8ae9b88820fc2e771c6d5e4bdfc6ceb

- physics: PASSED cond≈2.27e+03
- data: GENERATE matched MMS under smoke_v2
- manifests: smoke_v2_splits.json sha=7492aa94ba50
- norm_ratio mfg/genuine=1.980
- pretrain MML-direct...
  MML done loss=0.22966624796390533 params=66417
- pretrain LMOP-JEPA...
  LMOP done loss=15.2118 std=0.33364007985219357
- shuffled-physics LMOP control (u_vs_af)...
  shuffle loss=15.283977508544922 vs correct=15.211813926696777 supported=True
- finetune scratch_n16_s11 (freeze_epochs=5)
- finetune mml_direct_n16_s11 (freeze_epochs=5)
- finetune lmop_jepa_n16_s11 (freeze_epochs=5)
- finetune scratch_n32_s11 (freeze_epochs=5)
- finetune mml_direct_n32_s11 (freeze_epochs=5)
- finetune lmop_jepa_n32_s11 (freeze_epochs=5)
- table: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/smoke_v2_metrics.csv
- DONE smoke study2

## Run confirmatory_s2_v2_20260926T162730Z
- mode: confirmatory
- study: 2 data_version=v2
- amendment: 0003_matched_block_jepa
- device: mps
- grid: 64
- resume: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s2_v2_20260926T162730Z
- git_sha: 0256cb6ade89d94c96e0907bd1629f33a3b89f27

- physics: PASSED cond≈2.2e+04
- data: GENERATE matched MMS under confirmatory_v2

## Run confirmatory_s2_v2_20260926T162730Z
- mode: confirmatory
- study: 2 data_version=v2
- amendment: 0003_matched_block_jepa
- device: mps
- grid: 64
- resume: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s2_v2_20260926T162730Z
- git_sha: 0256cb6ade89d94c96e0907bd1629f33a3b89f27

- physics: PASSED cond≈2.2e+04
- data: RESUME existing HDF5 under confirmatory_v2

## Run confirmatory_s2_v2_20260926T162730Z
- mode: confirmatory
- study: 2 data_version=v2
- amendment: 0003_matched_block_jepa
- device: mps
- grid: 64
- resume: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s2_v2_20260926T162730Z
- git_sha: 815c31011cf8221486c76f6d1352f9e79cec172e

- physics: PASSED cond≈2.2e+04
- data: ensure corpus under confirmatory_v2
- manifests: confirmatory_v2_splits.json sha=d352e8df4392
- norm_ratio mfg/genuine=0.978
- pretrain MML-direct...
  MML done loss=0.12817710638046265 params=1185057
- pretrain LMOP-JEPA...
  LMOP done loss=14.5752 std=0.5237123947429937
- shuffled-physics LMOP control (u_vs_af)...
  shuffle loss=15.827692031860352 vs correct=14.575153350830078 supported=True
- finetune scratch_n25_s11 (freeze_epochs=10)
- finetune mml_direct_n25_s11 (freeze_epochs=10)
- finetune lmop_jepa_n25_s11 (freeze_epochs=10)
- finetune scratch_n100_s11 (freeze_epochs=10)
- finetune mml_direct_n100_s11 (freeze_epochs=10)
- finetune lmop_jepa_n100_s11 (freeze_epochs=10)
- finetune scratch_n25_s23 (freeze_epochs=10)
- finetune mml_direct_n25_s23 (freeze_epochs=10)
- finetune lmop_jepa_n25_s23 (freeze_epochs=10)
- finetune scratch_n100_s23 (freeze_epochs=10)
- finetune mml_direct_n100_s23 (freeze_epochs=10)
- finetune lmop_jepa_n100_s23 (freeze_epochs=10)
- finetune scratch_n25_s47 (freeze_epochs=10)
- finetune mml_direct_n25_s47 (freeze_epochs=10)
- finetune lmop_jepa_n25_s47 (freeze_epochs=10)
- finetune scratch_n100_s47 (freeze_epochs=10)
- finetune mml_direct_n100_s47 (freeze_epochs=10)
- finetune lmop_jepa_n100_s47 (freeze_epochs=10)
- table: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/confirmatory_v2.csv
- claim_gate: FALSIFIES_HYPOTHESIS -> /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/claim_gate.json
- claim_detail: MML ≥ LMOP on relative L2 across seeds under this protocol.
- DONE confirmatory study2

## Study-3 run confirmatory_s3_v2_20260926T185248Z
- amendment: 0004_preserve_ssl
- pretrain_source: confirmatory_s2_v2_20260926T162730Z
- adaptations: ['zero_shot']
- device: mps
- git_sha: 1f0430da5c8cc11e24ca94fad8d102eb08a4402f

- reuse LMOP ckpt /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s2_v2_20260926T162730Z/pretrain_lmop/checkpoint.pt

### adaptation=zero_shot
- zero_shot zero_shot_scratch_n25_s11 id_l2=2.1218
- zero_shot zero_shot_mml_direct_n25_s11 id_l2=0.7778
- zero_shot zero_shot_lmop_jepa_n25_s11 id_l2=0.6857
- zero_shot zero_shot_scratch_n100_s11 id_l2=3.2797
- zero_shot zero_shot_mml_direct_n100_s11 id_l2=0.7778
- zero_shot zero_shot_lmop_jepa_n100_s11 id_l2=0.6857
- zero_shot zero_shot_scratch_n25_s23 id_l2=2.6501
- zero_shot zero_shot_mml_direct_n25_s23 id_l2=0.7778
- zero_shot zero_shot_lmop_jepa_n25_s23 id_l2=0.6857
- zero_shot zero_shot_scratch_n100_s23 id_l2=4.4533
- zero_shot zero_shot_mml_direct_n100_s23 id_l2=0.7778
- zero_shot zero_shot_lmop_jepa_n100_s23 id_l2=0.6857
- zero_shot zero_shot_scratch_n25_s47 id_l2=1.5182
- zero_shot zero_shot_mml_direct_n25_s47 id_l2=0.7778
- zero_shot zero_shot_lmop_jepa_n25_s47 id_l2=0.6857
- zero_shot zero_shot_scratch_n100_s47 id_l2=1.8519
- zero_shot zero_shot_mml_direct_n100_s47 id_l2=0.7778
- zero_shot zero_shot_lmop_jepa_n100_s47 id_l2=0.6857
- zero_shot claim_gate: SUPPORTS_HYPOTHESIS (LMOP beats MML with non-overlapping seed CIs on ['N=25/id', 'N=25/ood', 'N=100/id', 'N=100/ood']; shuffled-physics control worse than correct.)
- DONE study3 -> /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/STUDY3_RESULTS.md

## Study-3 run confirmatory_s3_v2_20260926T185418Z
- amendment: 0004_preserve_ssl
- pretrain_source: confirmatory_s2_v2_20260926T162730Z
- adaptations: ['probe', 'low_lr_ft']
- device: mps
- git_sha: 1f0430da5c8cc11e24ca94fad8d102eb08a4402f

- retrain LMOP with λ_var=1, λ_u=1...
  LMOP retrain done loss=1.3985 std=0.14841447416413575
- retrain shuffled LMOP (u_vs_af)...
  shuffle 2.2258 vs correct 1.3985 supported=True

### adaptation=probe
- finetune probe_scratch_n25_s11
- finetune probe_mml_direct_n25_s11
- finetune probe_lmop_jepa_n25_s11
- finetune probe_scratch_n100_s11
- finetune probe_mml_direct_n100_s11
- finetune probe_lmop_jepa_n100_s11
- finetune probe_scratch_n25_s23
- finetune probe_mml_direct_n25_s23
- finetune probe_lmop_jepa_n25_s23
- finetune probe_scratch_n100_s23
- finetune probe_mml_direct_n100_s23
- finetune probe_lmop_jepa_n100_s23
- finetune probe_scratch_n25_s47
- finetune probe_mml_direct_n25_s47
- finetune probe_lmop_jepa_n25_s47
- finetune probe_scratch_n100_s47
- finetune probe_mml_direct_n100_s47
- finetune probe_lmop_jepa_n100_s47
- probe claim_gate: FALSIFIES_HYPOTHESIS (MML ≥ LMOP on relative L2 across seeds under this protocol.)

### adaptation=low_lr_ft
- finetune low_lr_ft_scratch_n25_s11
- finetune low_lr_ft_mml_direct_n25_s11
- finetune low_lr_ft_lmop_jepa_n25_s11
- finetune low_lr_ft_scratch_n100_s11
- finetune low_lr_ft_mml_direct_n100_s11
- finetune low_lr_ft_lmop_jepa_n100_s11
- finetune low_lr_ft_scratch_n25_s23
- finetune low_lr_ft_mml_direct_n25_s23
- finetune low_lr_ft_lmop_jepa_n25_s23
- finetune low_lr_ft_scratch_n100_s23
- finetune low_lr_ft_mml_direct_n100_s23
- finetune low_lr_ft_lmop_jepa_n100_s23
- finetune low_lr_ft_scratch_n25_s47

## Study-3 run confirmatory_s3_v2_20260926T185418Z
- amendment: 0004_preserve_ssl
- pretrain_source: confirmatory_s2_v2_20260926T162730Z
- adaptations: ['low_lr_ft']
- device: mps
- git_sha: 22a06963cb940753b00e61e7df136be4ce7773df
- resume: /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s3_v2_20260926T185418Z

- resume LMOP ckpt /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/runs/confirmatory_s3_v2_20260926T185418Z/pretrain_lmop/checkpoint.pt

### adaptation=low_lr_ft
- skip complete low_lr_ft_scratch_n25_s11
- skip complete low_lr_ft_mml_direct_n25_s11
- skip complete low_lr_ft_lmop_jepa_n25_s11
- skip complete low_lr_ft_scratch_n100_s11
- skip complete low_lr_ft_mml_direct_n100_s11
- skip complete low_lr_ft_lmop_jepa_n100_s11
- skip complete low_lr_ft_scratch_n25_s23
- skip complete low_lr_ft_mml_direct_n25_s23
- skip complete low_lr_ft_lmop_jepa_n25_s23
- skip complete low_lr_ft_scratch_n100_s23
- skip complete low_lr_ft_mml_direct_n100_s23
- skip complete low_lr_ft_lmop_jepa_n100_s23
- skip complete low_lr_ft_scratch_n25_s47
- finetune low_lr_ft_mml_direct_n25_s47
- finetune low_lr_ft_lmop_jepa_n25_s47
- finetune low_lr_ft_scratch_n100_s47
- finetune low_lr_ft_mml_direct_n100_s47
- finetune low_lr_ft_lmop_jepa_n100_s47
- low_lr_ft claim_gate: FALSIFIES_HYPOTHESIS (MML ≥ LMOP on relative L2 across seeds under this protocol.)
- DONE study3 -> /Volumes/PRO-BLADE/Kyrlov-JEPA/research/free-physics-jepa/results/tables/STUDY3_RESULTS.md
