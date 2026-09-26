# Pre-overnight publish checklist

Prep status before you start the overnight job.

## Done (you do not need to redo)

- [x] Protocol frozen (`protocol.yaml`, amendments 0001 + 0002)
- [x] Physics validation path + Darcy tests
- [x] Confirmatory data + manifests on disk
- [x] Smoke gate passed (pipeline only)
- [x] Partial confirmatory matrix (10/18 finetunes complete)
- [x] Runner resumes incomplete run & skips finished cells
- [x] Incomplete `mml_direct_n100_s23` scrubbed for clean retrain
- [x] Confirmatory shuffled-physics control wired
- [x] Claim-gate + `paper/RESULTS_AUTO.md` auto-write
- [x] Paper draft skeleton (`paper/DRAFT.md`)
- [x] Git repo initialized (SHA recorded by overnight run)
- [x] One-command overnight script (`scripts/overnight_confirmatory.sh`)

## Your only action tonight

```bash
cd /Volumes/PRO-BLADE/Kyrlov-JEPA
./research/free-physics-jepa/scripts/overnight_confirmatory.sh
```

See `OVERNIGHT.md`.

## Morning (after DONE)

- [ ] Log shows `DONE confirmatory`
- [ ] `results/tables/confirmatory.csv` has 36 rows
- [ ] Read `results/tables/claim_gate.md` verdict
- [ ] Paste `paper/RESULTS_AUTO.md` into draft §Results
- [ ] Write abstract with honest support / falsify / inconclusive language
