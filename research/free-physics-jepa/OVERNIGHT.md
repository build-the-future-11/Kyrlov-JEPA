# Overnight publish path (LMOP-JEPA)

Everything for tomorrow’s writeup is wired. **You only need to start the overnight job.**

## One command

```bash
cd /Volumes/PRO-BLADE/Kyrlov-JEPA
nohup ./research/free-physics-jepa/scripts/overnight_confirmatory.sh \
  > research/free-physics-jepa/runs/nohup_overnight.out 2>&1 &
```

Or in a dedicated terminal (logs to `runs/overnight_confirmatory.log`):

```bash
cd /Volumes/PRO-BLADE/Kyrlov-JEPA
./research/free-physics-jepa/scripts/overnight_confirmatory.sh
```

## What it does

1. Resumes `runs/confirmatory_20260926T150845Z` (skips completed cells)
2. Scrubs incomplete `mml_direct_n100_s23` and retrains it
3. Runs **confirmatory shuffled-physics LMOP** control
4. Finishes remaining matrix cells (incl. seed **47**)
5. Writes:
   - `results/tables/confirmatory.csv`
   - `results/tables/claim_gate.json` + `claim_gate.md`
   - `paper/RESULTS_AUTO.md`
   - `figures/confirmatory_label_efficiency.png`
6. Appends progress to `OVERNIGHT_STATUS.md`

## Morning checklist (you)

1. Confirm log ends with `DONE confirmatory`
2. Open `results/tables/claim_gate.md` — note the **verdict**
3. Paste `paper/RESULTS_AUTO.md` into `paper/DRAFT.md` §Results
4. Publish with protocol-honest language (support / falsify / inconclusive)

## Do not

- Use smoke metrics as evidence
- Cite `spectral-krylov-jepa` as LMOP support
- Redesign data after seeing outcomes
