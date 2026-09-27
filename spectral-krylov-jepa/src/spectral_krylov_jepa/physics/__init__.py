"""Physics package: grid, operators, potentials, eigensolver, Lanczos."""

from spectral_krylov_jepa.physics.eigensolver import EigenpairResult, solve_ground_state
from spectral_krylov_jepa.physics.grid import GridSpec
from spectral_krylov_jepa.physics.hamiltonian import Hamiltonian, build_hamiltonian
from spectral_krylov_jepa.physics.lanczos import LanczosResult, run_lanczos
from spectral_krylov_jepa.physics.potentials import generate_potential

__all__ = [
    "GridSpec",
    "Hamiltonian",
    "build_hamiltonian",
    "generate_potential",
    "solve_ground_state",
    "EigenpairResult",
    "run_lanczos",
    "LanczosResult",
]
