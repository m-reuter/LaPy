"""Tests for Solver.eigs and Solver.poisson — parameters and 2-D rhs support."""

import numpy as np
import pytest

from ...solver import Solver
from ...tria_mesh import TriaMesh


@pytest.fixture
def tria_mesh():
    return TriaMesh.read_off("data/square-mesh.off")


# ---------------------------------------------------------------------------
# eigs — new parameter and sorting tests
# ---------------------------------------------------------------------------


def test_eigs_rng_int_reproducible(tria_mesh):
    """Two calls with the same integer rng seed must return identical results."""
    fem = Solver(tria_mesh, lump=True)
    evals1, evecs1 = fem.eigs(k=4, rng=42)
    evals2, evecs2 = fem.eigs(k=4, rng=42)
    np.testing.assert_array_equal(evals1, evals2)
    np.testing.assert_array_equal(evecs1, evecs2)


def test_eigs_v0_takes_precedence_over_rng(tria_mesh):
    """Explicit v0 must take precedence over rng."""
    fem = Solver(tria_mesh, lump=True)
    v0 = np.random.default_rng(0).standard_normal(len(tria_mesh.v))
    # Same v0 with different rng seeds must give identical results.
    evals1, evecs1 = fem.eigs(k=4, v0=v0, rng=0)
    evals2, evecs2 = fem.eigs(k=4, v0=v0, rng=99)
    np.testing.assert_array_equal(evals1, evals2)
    np.testing.assert_array_equal(evecs1, evecs2)


def test_poisson_scalar_and_1d_return_1d(tria_mesh):
    """Scalar and 1-D rhs must return a 1-D array (backward compatibility)."""
    fem = Solver(tria_mesh, lump=True)
    _, evec = fem.eigs(k=3)

    assert fem.poisson(0.0).ndim == 1
    assert fem.poisson(evec[:, 1]).ndim == 1


def test_poisson_2d_rhs_matches_1d(tria_mesh):
    """2-D rhs must give the same result as repeated independent 1-D solves."""
    fem = Solver(tria_mesh, lump=True)
    _, evec = fem.eigs(k=5)
    rhs = evec[:, 1:5]  # (n_vertices, 4)

    x_batch = fem.poisson(rhs)

    assert x_batch.shape == (len(tria_mesh.v), 4)
    for k in range(4):
        np.testing.assert_allclose(
            x_batch[:, k],
            fem.poisson(rhs[:, k]),
            rtol=1e-6, atol=1e-9,
            err_msg=f"poisson 2-D mismatch at column {k}",
        )


def test_poisson_2d_rhs_with_dirichlet(tria_mesh):
    """2-D rhs with Dirichlet BC must match repeated 1-D solves."""
    fem = Solver(tria_mesh, lump=True)
    _, evec = fem.eigs(k=5)
    rhs = evec[:, 1:4]  # (n_vertices, 3)
    dtup = (np.array([0, 1]), np.array([0.0, 0.0]))

    x_batch = fem.poisson(rhs, dtup=dtup)

    assert x_batch.shape == (len(tria_mesh.v), 3)
    for k in range(3):
        np.testing.assert_allclose(
            x_batch[:, k],
            fem.poisson(rhs[:, k], dtup=dtup),
            rtol=1e-6, atol=1e-9,
            err_msg=f"poisson 2-D Dirichlet mismatch at column {k}",
        )

