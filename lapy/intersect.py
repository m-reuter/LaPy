"""Self-intersection tests for triangle meshes.

Candidate pairs of triangles come from a KD-tree on the triangle centroids,
and each candidate pair is tested exactly with the interval test of Moeller
(1997). Everything is vectorized over the candidate pairs.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import spatial

if TYPE_CHECKING:
    from .tria_mesh import TriaMesh

_EDGES = ((0, 1), (1, 2), (2, 0))


def _plane_distances(p: np.ndarray, n: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Signed distances of the corners p (k, 3, 3) to the planes through q (k, 3) with normals n."""
    return np.einsum("kij,kj->ki", p - q[:, None, :], n)


def _interval(x: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interval on the intersection line covered by a triangle.

    ``x`` holds the corner coordinates along the line and ``s`` the signed
    corner distances to the other triangle's plane. The interval spans corners
    on the plane and edge crossings.
    """
    lo = np.full(len(x), np.inf)
    hi = np.full(len(x), -np.inf)
    for i, j in _EDGES:
        cross = s[:, i] * s[:, j] < 0
        with np.errstate(divide="ignore", invalid="ignore"):
            val = x[:, i] + (x[:, j] - x[:, i]) * s[:, i] / (s[:, i] - s[:, j])
        lo = np.where(cross, np.minimum(lo, val), lo)
        hi = np.where(cross, np.maximum(hi, val), hi)
    for i in range(3):
        on = s[:, i] == 0
        lo = np.where(on, np.minimum(lo, x[:, i]), lo)
        hi = np.where(on, np.maximum(hi, x[:, i]), hi)
    return lo, hi


def triangles_intersect(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Test pairs of triangles for intersection.

    Uses the interval test of Moeller [1]_. Pairs that only touch in a
    single point and coplanar pairs are reported as not intersecting.

    Parameters
    ----------
    a : np.ndarray
        Corner coordinates of the first triangle of each pair, shape (k, 3, 3).
    b : np.ndarray
        Corner coordinates of the second triangle of each pair, shape (k, 3, 3).
    eps : float, default=1e-10
        Distances to the other triangle's plane below ``eps`` times the
        longest edge of the pair count as zero.

    Returns
    -------
    np.ndarray
        Boolean array of shape (k,), True where the two triangles intersect.

    References
    ----------
    .. [1] T. Moeller. A fast triangle-triangle intersection test. Journal of
       Graphics Tools, 2(2):25-30, 1997. doi:10.1080/10867651.1997.10487472
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    # unit normals make the plane values true distances; a degenerate
    # triangle gets a zero normal and is treated as coplanar, so never hit
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = np.nan_to_num(na / np.linalg.norm(na, axis=1, keepdims=True))
        ub = np.nan_to_num(nb / np.linalg.norm(nb, axis=1, keepdims=True))
    sb = _plane_distances(b, ua, a[:, 0])
    sa = _plane_distances(a, ub, b[:, 0])
    edges = np.concatenate([a - np.roll(a, 1, axis=1), b - np.roll(b, 1, axis=1)], axis=1)
    tol = eps * np.linalg.norm(edges, axis=2).max(axis=1)
    sa[np.abs(sa) <= tol[:, None]] = 0
    sb[np.abs(sb) <= tol[:, None]] = 0
    separated = (
        np.all(sb > 0, axis=1)
        | np.all(sb < 0, axis=1)
        | np.all(sa > 0, axis=1)
        | np.all(sa < 0, axis=1)
        | np.all(sa == 0, axis=1)
        | np.all(sb == 0, axis=1)
    )
    hit = ~separated
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return hit
    axis = np.argmax(np.abs(np.cross(na[idx], nb[idx])), axis=1)
    k = np.arange(len(idx))[:, None]
    lo_a, hi_a = _interval(a[idx][k, np.arange(3), axis[:, None]], sa[idx])
    lo_b, hi_b = _interval(b[idx][k, np.arange(3), axis[:, None]], sb[idx])
    hit[idx] = (lo_a < hi_b) & (lo_b < hi_a)
    return hit


def _test_pairs(v: np.ndarray, t: np.ndarray, pairs: np.ndarray, centre: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Keep the candidate pairs whose triangles intersect and share no vertex."""
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    dist = np.linalg.norm(centre[pairs[:, 0]] - centre[pairs[:, 1]], axis=1)
    pairs = pairs[dist <= radius[pairs[:, 0]] + radius[pairs[:, 1]]]
    shared = (t[pairs[:, 0], :, None] == t[pairs[:, 1], None, :]).any(axis=(1, 2))
    pairs = pairs[~shared]
    p = v[t]
    return pairs[triangles_intersect(p[pairs[:, 0]], p[pairs[:, 1]])]


def _intersections(v: np.ndarray, t: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """Intersecting triangle pairs of the mesh (v, t), optionally only those involving ``subset``."""
    p = v[t]
    centre = p.mean(axis=1)
    radius = np.linalg.norm(p - centre[:, None], axis=2).max(axis=1)
    if subset is None:
        # the few much larger triangles are searched one by one, so that they
        # do not inflate the search radius for all others
        big_r = np.percentile(radius, 99.9)
        tree = spatial.KDTree(centre)
        pairs = [tree.query_pairs(2 * big_r, output_type="ndarray")]
        for i in np.flatnonzero(radius > big_r):
            near = np.array(tree.query_ball_point(centre[i], radius[i] + radius.max()), dtype=np.int64)
            pairs.append(np.column_stack([np.minimum(i, near), np.maximum(i, near)]))
    else:
        subset = np.unique(np.asarray(subset, dtype=np.int64))
        if len(subset) == 0:
            return np.empty((0, 2), dtype=np.int64)
        # only triangles in the bounding box of the subset, widened by the
        # largest possible reach, can intersect it
        reach = radius[subset].max() + radius.max()
        lo = centre[subset].min(axis=0) - reach
        hi = centre[subset].max(axis=0) + reach
        box = np.flatnonzero(np.all((centre >= lo) & (centre <= hi), axis=1))
        near = spatial.KDTree(centre[box]).query_ball_point(centre[subset], radius[subset] + radius.max())
        a = np.repeat(subset, [len(n) for n in near])
        b = box[np.concatenate([np.asarray(n, dtype=np.int64) for n in near])]
        pairs = [np.column_stack([np.minimum(a, b), np.maximum(a, b)])]
    # every pair is ordered within its row
    pairs = np.unique(np.concatenate(pairs).reshape(-1, 2), axis=0)
    return _test_pairs(v, t, pairs, centre, radius)


def self_intersections(tria: "TriaMesh", subset: np.ndarray | None = None) -> np.ndarray:
    """Find pairs of triangles that intersect and do not share a vertex.

    Triangles that share a vertex or an edge are never reported, and
    neither are triangles that only touch in a point or lie in a common
    plane. For a 2D mesh, which is planar, nothing is reported.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh.
    subset : np.ndarray, default=None
        Triangle indices. If given, only pairs that involve at least one of
        these triangles are reported, which is much faster when only a few
        triangles have moved.

    Returns
    -------
    np.ndarray
        Array of shape (n_pairs, 2) with the indices of intersecting
        triangles, sorted within each row and lexicographically.
    """
    return _intersections(np.asarray(tria.v, dtype=float), np.asarray(tria.t, dtype=np.int64), subset)


def undo_intersections(
        tria: "TriaMesh", v_new: np.ndarray, rings: int = 2, max_rounds: int = 10
) -> tuple[np.ndarray, int]:
    """Move vertices back where new positions make the surface intersect itself.

    Useful after smoothing or another change of vertex positions on a
    surface that did not intersect itself before. Starting from the
    triangles that intersect with the new positions, their vertices and
    ``rings`` rings of neighbors return to their positions in ``tria``. This
    repeats, testing only the triangles that changed, until no triangle
    intersects or ``max_rounds`` is reached. The mesh itself is not
    modified.

    Parameters
    ----------
    tria : TriaMesh
        Triangle mesh with the old vertex positions.
    v_new : np.ndarray
        New vertex positions of shape (n_vertices, 3), or (n_vertices, 2)
        for a 2D mesh.
    rings : int, default=2
        Rings of neighbors that move back together with each intersecting
        triangle.
    max_rounds : int, default=10
        Maximum number of rounds.

    Returns
    -------
    v : np.ndarray
        Vertex positions, of the same shape as ``v_new``.
    n_reverted : int
        Number of vertices moved back to their old positions.

    Raises
    ------
    ValueError
        If ``v_new`` does not have the shape of the mesh vertices.
    """
    v_old = tria.get_vertices(original_dim=True)
    v = np.array(v_new, dtype=float)
    if v.shape != v_old.shape:
        raise ValueError(f"v_new has shape {v.shape}, the mesh has vertices of shape {v_old.shape}.")
    t = np.asarray(tria.t, dtype=np.int64)
    adj = (tria.adj_sym > 0).astype(float)

    def positions_3d(x: np.ndarray) -> np.ndarray:
        return x if x.shape[1] == 3 else np.column_stack([x, np.zeros(len(x))])

    reverted = np.zeros(len(v), dtype=bool)
    pairs = _intersections(positions_3d(v), t)
    for _ in range(max_rounds):
        if len(pairs) == 0:
            break
        mask = np.zeros(len(v), dtype=bool)
        mask[t[pairs].ravel()] = True
        for _ in range(rings):
            mask |= adj @ mask.astype(float) > 0
        v[mask] = v_old[mask]
        reverted |= mask
        pairs = _intersections(positions_3d(v), t, np.flatnonzero(np.any(mask[t], axis=1)))
    return v, int(reverted.sum())
