import json

import numpy as np
import pytest

from spectral_krylov_jepa.evaluation.baselines import sine_ritz
from spectral_krylov_jepa.physics.grid import GridSpec, cell_area
from spectral_krylov_jepa.physics.hamiltonian import build_hamiltonian
from spectral_krylov_jepa.utils.provenance import freeze_or_check


def test_resume_rejects_changed_identity(tmp_path):
    path = tmp_path / "identity.json"
    freeze_or_check(path, {"seed": 11})
    freeze_or_check(path, {"seed": 11})
    with pytest.raises(ValueError, match="incompatible"):
        freeze_or_check(path, {"seed": 23})
    assert json.loads(path.read_text()) == {"seed": 11}


def test_ritz_constant_potential_exact_and_nested_variational():
    grid = GridSpec(8)
    v = np.full((8, 8), 2.0)
    energy, psi = sine_ritz(v, grid)
    exact = 4 / grid.h**2 * np.sin(np.pi / 18)**2 + 2
    assert abs(energy - exact) < 1e-10
    assert abs((psi**2).sum() * cell_area(grid) - 1) < 1e-12
    assert np.linalg.norm(build_hamiltonian(grid, v).matvec(psi.ravel()) - energy * psi.ravel()) < 1e-10
    v = np.random.default_rng(8).normal(size=(8, 8))
    energies = [sine_ritz(v, grid, m)[0] for m in [1, 2, 3, 8]]
    assert np.all(np.diff(energies) <= 1e-10)


def test_invalid_update_frequency():
    from spectral_krylov_jepa.training.pretrain import pretrain
    with pytest.raises(ValueError, match="frequency"):
        pretrain(method="krylov", data_path="unused", config={"target_update_frequency": 0})
