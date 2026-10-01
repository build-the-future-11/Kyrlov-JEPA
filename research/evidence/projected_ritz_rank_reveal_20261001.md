# Projected Rayleigh-Ritz rank-reveal receipt — 2026-10-01

Source: `build-the-future-11/Kyrlov-JEPA`
Canonical main: `41d989cc17bae906fd2d59f5417998edf255ed28`
Frozen research boundary: `6c91197c3cc8287c0ab3c4205b2114093377af41`
Base numerical branch: `numerics/adaptive-ritz-reorth-20261001`
Base-branch solver blob: `c6e15c73ca0c0ebab96a57c4b9f18f5b1b28b089`

## Defect
For `projected_ritz(..., assume_orthonormal=False)`, canonical code uses
`np.linalg.qr(b)[0]`. Rank-deficient bases therefore receive one Q column per
input column, allowing duplicate/zero columns to inject arbitrary directions
outside the requested trial span.

A deterministic CPU fixture with H=diag(1,10,20,30) and requested span e2
reproduced the failure:
- one e2 column -> Ritz energy 10
- duplicated [e2,e2] under the old QR path -> basis_dim 2, Ritz energy 1
- rank-revealing SVD path -> basis_dim 1, Ritz energy 10
- zero basis -> old QR manufactures a two-dimensional space; proposed path
  rejects numerical rank zero
- random full-rank 8x3 fixture: projected spectrum differs by only
  2.6645352591003757e-15

## Proposed patch
Use compact SVD for arbitrary bases, determine numerical rank with
`max(shape) * eps * largest_singular_value`, reject rank-zero bases, and pass
only the actual column space to the projected eigensolve. The orthonormal fast
path remains unchanged.

Two regression tests are specified:
1. duplicate columns preserve trial-space dimension and energy;
2. zero-rank bases fail closed.

## Execution boundary
Executed only the self-contained deterministic CPU regression described above.
Repository-wide pytest, package build and CI were UNRUN. No outcome study,
protected evaluation, workflow dispatch, merge or submission occurred.

The executable source write was blocked by connector safety checks, so this
branch receipt does not claim that the solver patch landed. The exact patch and
standalone regression were retained in the automation artifact for handoff.
