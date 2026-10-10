import numpy as np

from ...intersect import self_intersections, triangles_intersect, undo_intersections
from ...tria_mesh import TriaMesh
from .test_tria_mesh import _torus


def _pushed_through_torus():
    """Torus with a vertex of the inner equator moved across the hole into the opposite tube."""
    v, t = _torus(30, 12, 3.0, 1.0)
    i = int(np.argmin(np.linalg.norm(v - [2.0, 0.0, 0.0], axis=1)))
    moved = v.copy()
    moved[i] = [-3.0, 0.0, 0.0]
    return v, moved, t, i


def test_triangle_pairs():
    """
    A triangle against one that pierces it, lies above it, touches a corner,
    lies beside it and lies in the same plane.
    """
    a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    piercing = np.array([[0.5, 0.5, -1], [0.5, 0.5, 1], [3, 3, 0]], dtype=float)
    above = piercing + [0, 0, 5]
    corner_touch = np.array([[2, 0, 0], [3, 0, 1], [3, 0, -1]], dtype=float)
    beside = np.array([[3, 0, -1], [3, 0, 1], [4, 1, 0]], dtype=float)
    coplanar = a + [0.5, 0.5, 0]
    others = np.stack([piercing, above, corner_touch, beside, coplanar])
    hit = triangles_intersect(np.repeat(a[None], len(others), axis=0), others)
    assert hit.tolist() == [True, False, False, False, False]


def test_thin_sliver_through_large_triangle():
    """
    A sliver piercing a much larger triangle is found in either argument
    order, and random pairs give the same answer in both orders.
    """
    a = np.array([[-5, -5, 0], [5, -5, 0], [0, 5, 0]], dtype=float)
    for width in (1e-1, 1e-5, 1e-9):
        b = np.array([[0, 0, -1], [0, 0, 1], [width, 0, 1]], dtype=float)
        assert triangles_intersect(a[None], b[None])[0]
        assert triangles_intersect(b[None], a[None])[0]
    rng = np.random.default_rng(0)
    p = rng.normal(size=(2000, 3, 3))
    q = rng.normal(size=(2000, 3, 3)) * [1.0, 1.0, 1e-6]
    hit = triangles_intersect(p, q)
    assert 0 < hit.sum() < len(hit)
    np.testing.assert_array_equal(hit, triangles_intersect(q, p))


def test_overlapping_copies_intersect():
    """
    Two copies of a torus intersect each other where they overlap, never
    within one copy, and not at all when they are far apart.
    """
    v, t = _torus(30, 12, 3.0, 1.0)
    assert len(self_intersections(TriaMesh(v, t))) == 0
    both = np.vstack([t, t + len(v)])
    pairs = self_intersections(TriaMesh(np.vstack([v, v + [1.3, 0.4, 0.7]]), both))
    assert len(pairs) > 0
    assert np.all((pairs[:, 0] < len(t)) != (pairs[:, 1] < len(t)))
    assert len(self_intersections(TriaMesh(np.vstack([v, v + [20.0, 0, 0]]), both))) == 0


def test_subset_finds_the_same_pairs():
    """
    Restricting the search to the triangles around a moved vertex finds the
    same pairs as the full search.
    """
    _, moved, t, i = _pushed_through_torus()
    mesh = TriaMesh(moved, t)
    full = self_intersections(mesh)
    assert len(full) > 0
    around = np.flatnonzero(np.any(t == i, axis=1))
    np.testing.assert_array_equal(self_intersections(mesh, subset=around), full)
    far = np.setdiff1d(np.arange(len(t)), np.unique(full))[:10]
    assert len(self_intersections(mesh, subset=far)) == 0


def test_undo_intersections_moves_back_only_near_the_crossing():
    """
    After a vertex is pushed through the opposite tube, only the vertices
    near the crossing return to their old positions.
    """
    v, moved, t, i = _pushed_through_torus()
    # far from where the vertex started and from the tube it now pierces
    far = (np.linalg.norm(v - v[i], axis=1) > 3.0) & (np.linalg.norm(v - moved[i], axis=1) > 3.0)
    moved[far] += 0.01  # a small change elsewhere that keeps the surface clean
    mesh = TriaMesh(v, t)
    fixed, n_back = undo_intersections(mesh, moved)
    assert len(self_intersections(TriaMesh(fixed, t))) == 0
    np.testing.assert_allclose(fixed[i], v[i])
    np.testing.assert_allclose(fixed[far], moved[far])
    assert 0 < n_back < len(v) // 4
    # the mesh itself keeps its old positions
    np.testing.assert_allclose(mesh.v, v)
