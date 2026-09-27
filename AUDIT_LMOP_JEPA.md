# AUDIT: Free-Physics JEPA / LMOP-JEPA

> **SUPERSEDED (2026-09-27)** by `RESEARCH_AUDIT.md`. This audit predates the LMOP code in `research/free-physics-jepa/`; its "absent" findings are historical.

**Audit date:** 2026-09-26  
**Workspace audited:** `/Volumes/PRO-BLADE/Kyrlov-JEPA`  
**Auditor posture:** skeptical ML reviewer + numerical PDE + reproducibility  

**This audit does not invent LMOP-JEPA results, Darcy code, or confirmatory metrics.**

---

## 1. Executive verdict

### What this workspace actually contains

| Expected (LMOP-JEPA brief) | Present in workspace |
|---|---|
| Frozen Darcy reference | **ABSENT** |
| Manufactured-pair generation | **ABSENT** |
| `research/free-physics-jepa/RESEARCH_PROTOCOL.md` | **ABSENT** |
| `LITERATURE_COLLISION_MAP.md` | **ABSENT** |
| `protocol.yaml` | **ABSENT** |
| Scratch / MML-direct / LMOP-JEPA | **ABSENT** |
| Genuine Darcy solve corpus | **ABSENT** |
| ID/OOD Darcy evaluation | **ABSENT** |

**What is present:** a single project, `spectral-krylov-jepa/`, implementing **Spectral Krylov-JEPA** for the **2D Schrödinger eigenvalue problem** (Lanczos/Krylov self-supervised pretraining → few-shot ground-state prediction). That is a **different PDE, different supervision source, different critical baseline, and different scientific question**.

### Completeness for Free-Physics JEPA / LMOP-JEPA

**Not complete. Not partial. Missing as a research program in this workspace.**

### Is a confirmatory LMOP-JEPA run scientifically defensible right now?

**No.**

There is nothing to run that answers:

> Does joint-embedding latent pretraining on solver-free manufactured Darcy examples transfer to genuine numerical Darcy solutions more efficiently than direct manufactured-solution regression when only a small number of genuine PDE solves are available?

The critical comparison **MML-direct vs LMOP-JEPA** cannot be executed because neither baseline nor the Darcy stack exists here.

### Claim gate (LMOP-JEPA program)

| Gate | Justified? |
|---|---|
| engineering only | **No** (no LMOP engineering present) |
| protocol ready | **No** |
| smoke-run ready | **No** |
| confirmatory-run ready | **No** |
| results available | **No** |
| CJSJ evidence ready | **No** |

### Side note (do not confuse programs)

`spectral-krylov-jepa/` is a separate Schrödinger/Krylov research stack that is **smoke-verified** for *its own* question. Smoke metrics there are **not** LMOP-JEPA evidence and must not be reused as Darcy/CJSJ results.

---

## 2. Critical findings (P0 / P1)

### P0-1 — Wrong research program / total absence of LMOP-JEPA

- **Path:** workspace root; expected `research/free-physics-jepa/`
- **Why it matters:** Every phase of the requested audit (Darcy physics, manufactured round-trip, MML-direct fairness, LMOP latent objective, synthetic-to-real transfer) presupposes code and frozen protocol that are not present.
- **Fix:** Locate the correct Free-Physics JEPA repository, or intentionally scaffold LMOP-JEPA as a **new** program with frozen protocol *before* any confirmatory run. Do not retrofit Krylov-JEPA into Darcy claims.

### P0-2 — Critical baseline MML-direct missing

- **Path:** N/A (no implementation)
- **Why it matters:** Protocol states JEPA vs Scratch is insufficient; **MML-direct vs LMOP-JEPA** is the scientific comparison.
- **Fix:** Implement MML-direct on the **same** manufactured corpus before any JEPA claim.

### P0-3 — No frozen Darcy discrete operator / manufactured consistency contract

- **Path:** N/A
- **Why it matters:** Without identical discrete \(A_a\) for manufacture \(\tilde f=A_a\tilde u\) and genuine solves, the transfer experiment is invalid.
- **Fix:** Freeze one Darcy FD/FV contract; prove manufactured round-trip to FP tolerance before ML.

### P0-4 — Risk of silent question substitution via Krylov-JEPA presence

- **Path:** `spectral-krylov-jepa/`
- **Why it matters:** A polished, runnable JEPA-for-physics repo in the same volume can be mistaken for Free-Physics JEPA progress. Krylov compares Field/Operator/Krylov vs Scratch on Schrödinger eigenpairs—not manufactured Darcy vs MML-direct.
- **Fix:** Keep programs physically and documentarily separated; never cite Krylov smoke tables as LMOP evidence.

### P1-1 — No LMOP protocol / literature collision map / protocol.yaml

- **Paths expected:** `research/free-physics-jepa/RESEARCH_PROTOCOL.md`, `LITERATURE_COLLISION_MAP.md`, `protocol.yaml`
- **Fix:** Author and freeze before experiments.

### P1-2 — No CI, no git history for reproducibility of LMOP (or Krylov root)

- **Evidence:** no `.github/`; `spectral-krylov-jepa` is not a git repo (`git rev-parse` fails)
- **Fix:** Initialize version control; pin commits in run artifacts for any confirmatory program.

### P1-3 — No LMOP tests / entrypoints / manifests / result schema

- **Fix:** Build per phases 15–17 of the brief *before* confirmatory training.

---

## 3. Scientific integrity assessment (LMOP-JEPA)

| Dimension | Assessment |
|---|---|
| Novelty boundary | **Cannot assess empirically**—no manuscript/protocol present. Candidate novelty in the brief is narrow (latent vs direct manufactured transfer); nothing here claims or tests it. |
| Fairness (MML vs JEPA) | **N/A — baselines absent** |
| Leakage | **N/A — no Darcy splits** |
| Hypothesis validity | Brief hypothesis is coherent; **unimplemented** |
| Baseline strength | MML-direct **MISSING** → any future JEPA>Scratch result would be scientifically weak by the brief’s own standard |
| OOD validity | **MISSING** |
| Statistical design | **MISSING** for LMOP; Krylov has bootstrap utilities for a different task |

**Overclaim flag:** None found for LMOP (nothing to overclaim). **Do not** treat Krylov README/CJSJ outline as Free-Physics JEPA documentation.

---

## 4. Implementation matrix (LMOP-JEPA required components)

| Component | Status | Evidence | Required action |
|---|---|---|---|
| `research/free-physics-jepa/RESEARCH_PROTOCOL.md` | **MISSING** | Path does not exist | Create & freeze |
| `LITERATURE_COLLISION_MAP.md` | **MISSING** | Path does not exist | Create |
| `protocol.yaml` | **MISSING** | Path does not exist | Create & freeze |
| Darcy discrete operator | **MISSING** | No Darcy code (`rg` empty) | Implement + analytic tests |
| Genuine Darcy solver dataset | **MISSING** | — | Implement |
| Manufactured-pair generator | **MISSING** | — | Implement + round-trip tests |
| Immutable split manifests + hashes | **MISSING** | — | Implement |
| Scratch baseline (Darcy) | **MISSING** | — | Implement |
| **MML-direct** | **MISSING** | — | **Implement first** |
| LMOP-JEPA (context/target/EMA/predictor/mask) | **MISSING** | — | Implement after MML |
| Collapse diagnostics | **MISSING** | — | Implement |
| ID/OOD eval + rel-L2 + residual | **MISSING** | — | Implement |
| Paired bootstrap | **MISSING** (for LMOP) | — | Implement |
| Compute-fairness logging | **MISSING** | — | Implement |
| Confirmatory configs (N=25,100; seeds 11/23/47) | **MISSING** | — | Freeze |
| CJSJ paper with pending results | **MISSING** | — | Draft only after protocol freeze |
| CI | **MISSING** | No `.github` | Add |
| External Darcy datasets/checkpoints | **MISSING** | — | N/A until created |

### Compact map of what *is* in the workspace (unrelated program)

```
Kyrlov-JEPA/
└── spectral-krylov-jepa/     # Schrödinger Krylov-JEPA — NOT LMOP-JEPA
    ├── src/spectral_krylov_jepa/{physics,data,models,training,evaluation,...}
    ├── scripts/00–11
    ├── configs/, tests/, paper/, docs/, experiments/, results/, figures/
    └── .venv/
```

Classification for Krylov stack (informational only): physics/Lanczos **COMPLETE AND VERIFIED** (tests+smoke); full decisive Krylov experiment **IMPLEMENTED BUT UNVERIFIED** / not executed; LMOP components **MISSING**.

---

## 5. Tests executed (this audit)

Commands actually run during inventory:

```bash
find /Volumes/PRO-BLADE/Kyrlov-JEPA ...
rg 'Darcy|LMOP|manufactured|free-physics|MML-direct'   # no project hits outside .venv noise
test -d research/free-physics-jepa                     # absent
test -f RESEARCH_PROTOCOL.md                           # absent
```

**LMOP-JEPA test suite:** not present → **not run**.

Krylov suite was previously run in this workspace history (`23 passed`, smoke OK) but that **does not constitute** LMOP verification and is not re-asserted here as evidence for Free-Physics JEPA.

---

## 6. Changes made

**None.**  

No LMOP code, protocol, or results were fabricated. No frozen Krylov protocol was silently rewritten into a Darcy study. No “demo success” metrics were inserted.

---

## 7. Remaining blockers (real)

1. **Obtain or create** the Free-Physics JEPA / LMOP-JEPA codebase and frozen protocol.
2. Implement and verify **identical** Darcy discrete operator for manufacture and genuine solves.
3. Implement **MML-direct** on the manufactured corpus **before** LMOP claims.
4. Freeze splits/manifest hashes, OOD definition, label budgets (25/100), seeds (11/23/47).
5. Run smoke → then confirmatory **without** test-informed tuning.
6. Separate this workspace’s Krylov-JEPA artifacts from any LMOP reporting.

---

## 8. Exact next commands

**If the LMOP repo lives elsewhere:**

```bash
# open/clone the correct Free-Physics JEPA repository, then:
ls research/free-physics-jepa/RESEARCH_PROTOCOL.md
ls research/free-physics-jepa/LITERATURE_COLLISION_MAP.md
ls research/free-physics-jepa/protocol.yaml
```

**If this workspace is intended to host LMOP-JEPA (greenfield):** do **not** start from Krylov smoke metrics. Minimum sequence:

```bash
# 1) Author and freeze protocol files under research/free-physics-jepa/
# 2) Implement Darcy operator + manufactured round-trip tests
# 3) Generate genuine + manufactured data + hashed manifests
# 4) Train MML-direct on manufactured corpus
# 5) Pretrain LMOP-JEPA on SAME manufactured corpus
# 6) Fine-tune both + Scratch on identical genuine IDs at N=25,100; seeds 11,23,47
# 7) Evaluate ID/OOD rel-L2 + residual + paired bootstrap
```

No legitimate confirmatory command exists in this workspace today.

---

## 9. Claim gate (restated)

For **Free-Physics JEPA / LMOP-JEPA** in `/Volumes/PRO-BLADE/Kyrlov-JEPA`:

> **Not engineering-ready. Not protocol-ready. Not smoke-ready. Not confirmatory-ready. No results. Not CJSJ-ready.**

The end state of this audit is not that the repo looks incomplete—it is that **the audited research program is not present**, and therefore **no result from this workspace can currently answer the manufactured-latent vs MML-direct Darcy transfer question**.
