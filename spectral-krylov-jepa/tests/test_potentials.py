"""Potential generator tests."""

import numpy as np

from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.potentials import FAMILY_CONFIGS, generate_potential


def test_reproducible_generation():
    g = GridSpec(n_interior=16)
    v1, s1 = generate_potential("id_gaussian_mixture", 123, grid=g)
    v2, s2 = generate_potential("id_gaussian_mixture", 123, grid=g)
    assert np.allclose(v1, v2)
    assert s1.wells == s2.wells


def test_ood_families_exist():
    g = GridSpec(n_interior=12)
    for name in ("ood_narrow", "ood_strong", "ood_double", "ood_rough"):
        assert name in FAMILY_CONFIGS
        v, spec = generate_potential(name, 5, grid=g)
        assert v.shape == (12, 12)
        assert np.all(np.isfinite(v))
        assert spec.family == name


def test_different_seeds_differ():
    g = GridSpec(n_interior=12)
    v1, _ = generate_potential("id_gaussian_mixture", 1, grid=g)
    v2, _ = generate_potential("id_gaussian_mixture", 2, grid=g)
    assert not np.allclose(v1, v2)
