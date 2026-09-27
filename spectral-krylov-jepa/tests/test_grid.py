"""Grid tests."""

from spectral_krylov_jepa.physics.grid import GridSpec


def test_grid_dof_and_spacing():
    g = GridSpec(n_interior=32)
    assert g.n_dof == 32 * 32
    assert abs(g.h - 1.0 / 33) < 1e-12
    X, Y = g.meshgrid()
    assert X.shape == (32, 32)
    assert Y.shape == (32, 32)


def test_flatten_reshape_roundtrip():
    g = GridSpec(n_interior=8)
    X, Y = g.meshgrid()
    field = X + Y
    vec = g.flatten_field(field)
    assert vec.shape == (64,)
    back = g.reshape_vector(vec)
    assert (back == field).all()
