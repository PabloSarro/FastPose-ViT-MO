"""PnP on the dense outputs of the M18 heads: 2D-3D correspondences and four pose estimators.

A dense output (4, S, S) holds a mask logit and the normalised object coordinates of every pixel
of the object's crop. Each masked pixel is a correspondence between its position in the image
(K coordinates, see crop_warp_matrix) and a 3D point of the model (mm). Pose modes:
- "dr": the direct-regression pose, unchanged (the reference);
- "pnp": OpenCV PnP-RANSAC (EPnP) on the correspondences, then Levenberg-Marquardt on the inliers;
- "pnp_init": LM refinement started at the DR pose (inliers chosen by an adaptive threshold
  around its reprojection, shrinking as the pose converges);
- "dr_rot_pnp_t": the DR rotation, and the translation that best reprojects the correspondences
  (robust linear least squares).
A mode that cannot run (fewer than MIN_POINTS correspondences, or no inlier set) returns the DR
pose, flagged `fallback`, so a pose is always produced and the failures are counted.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.bop_dataset import crop_warp_matrix

POSE_MODES = ("dr", "pnp", "pnp_init", "dr_rot_pnp_t")
MIN_POINTS = 6
REPROJ_PX = 3.0  # RANSAC inlier threshold (pixels in the image)
INIT_MEDIAN_FACTOR = 2.5  # pnp_init: inlier threshold = this x the median error at the start

cv2.setNumThreads(1)  # one instance at a time; the DataLoader workers use the other cores


@dataclass
class Correspondences:
    """2D points (n, 2, K coordinates of the image) and the 3D model points (n, 3, mm)."""

    pts2d: np.ndarray
    pts3d: np.ndarray

    def __len__(self) -> int:
        return len(self.pts2d)


def dense_correspondences(
    dense: np.ndarray,
    bbox,
    centre: np.ndarray,
    half: np.ndarray,
    crop_scale: float = 1.0,
    mask_threshold: float = 0.0,
) -> Correspondences:
    """Correspondences of one instance from its dense output.

    Args:
        dense: (4, S, S): mask logit, normalised XYZ.
        bbox: [x1, y1, x2, y2] of the crop in K coordinates (the box the crop was made with).
        centre, half: The object's model-box centre and half-extent (mm), from object_extents.
        crop_scale: The crop scale the network saw.
        mask_threshold: Logit above which a pixel is used (0 = probability 0.5).
    """
    size = dense.shape[-1]
    M = crop_warp_matrix(bbox, size, crop_scale)
    ys, xs = np.nonzero(dense[0] > mask_threshold)
    u = (xs - M[0, 2]) / M[0, 0] + 0.5  # K coordinates of the crop pixels, as dense_targets
    v = (ys - M[1, 2]) / M[1, 1] + 0.5
    xyz = dense[1:4, ys, xs].T.astype(np.float64)
    return Correspondences(np.stack([u, v], axis=1), xyz * half + centre)


def reprojection_errors(corr: Correspondences, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """Pixel distance between each 2D point and its 3D point projected with (R, t); inf behind."""
    cam = corr.pts3d @ R.T + t
    z = cam[:, 2]
    proj = (cam[:, :2] / np.where(z > 1e-6, z, 1.0)[:, None]) * [K[0, 0], K[1, 1]]
    proj = proj + [K[0, 2], K[1, 2]]
    err = np.linalg.norm(proj - corr.pts2d, axis=1)
    return np.where(z > 1e-6, err, np.inf)


def _rodrigues(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.ascontiguousarray(R, dtype=np.float64))[0]


def _refine_lm(corr: Correspondences, K, rvec, tvec):
    """Levenberg-Marquardt on the reprojection error of the given correspondences."""
    rvec, tvec = cv2.solvePnPRefineLM(
        np.ascontiguousarray(corr.pts3d),
        np.ascontiguousarray(corr.pts2d),
        K,
        None,
        rvec.copy(),
        tvec.copy(),
    )
    return rvec, tvec


def _subset(corr: Correspondences, keep: np.ndarray) -> Correspondences:
    return Correspondences(corr.pts2d[keep], corr.pts3d[keep])


def pose_from_pnp(corr: Correspondences, K: np.ndarray, seed: int = 0):
    """PnP-RANSAC (EPnP) then LM on the inliers. Returns (R, t (3,), n_inliers) or None."""
    if len(corr) < MIN_POINTS:
        return None
    cv2.setRNGSeed(seed)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.ascontiguousarray(corr.pts3d),
        np.ascontiguousarray(corr.pts2d),
        K,
        None,
        iterationsCount=200,
        reprojectionError=REPROJ_PX,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < MIN_POINTS:
        return None
    inliers = inliers[:, 0]
    rvec, tvec = _refine_lm(_subset(corr, inliers), K, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec[:, 0], len(inliers)


def pose_from_init(corr: Correspondences, K: np.ndarray, R0: np.ndarray, t0: np.ndarray):
    """LM refinement from a starting pose (the DR pose), three rounds.

    A round keeps the correspondences that the current pose reprojects within
    max(REPROJ_PX, INIT_MEDIAN_FACTOR x the median error): the DR pose is only approximately
    right (a few degrees and centimetres = ~10 px), so a fixed small threshold would drop every
    point, while the median adapts to it and still excludes gross outliers (fewer than half of
    the points). The threshold shrinks to REPROJ_PX as the pose converges.
    Returns (R, t, n_inliers) or None.
    """
    R, t = R0, np.asarray(t0, dtype=np.float64)
    for _ in range(3):
        err = reprojection_errors(corr, K, R, t)
        finite = np.isfinite(err)
        if finite.sum() < MIN_POINTS:
            return None
        threshold = max(REPROJ_PX, INIT_MEDIAN_FACTOR * np.median(err[finite]))
        inliers = err < threshold
        if inliers.sum() < MIN_POINTS:
            return None
        rvec, tvec = _refine_lm(_subset(corr, inliers), K, _rodrigues(R), t.reshape(3, 1))
        R, t = cv2.Rodrigues(rvec)[0], tvec[:, 0]
    inliers = reprojection_errors(corr, K, R, t) < REPROJ_PX
    return R, t, int(inliers.sum())


def _translation_system(corr: Correspondences, K: np.ndarray, R: np.ndarray):
    """Rows of A t = b: the reprojection of R X + t onto (u, v), cleared of the division by z."""
    p = corr.pts3d @ R.T
    u, v = corr.pts2d[:, 0], corr.pts2d[:, 1]
    n = len(p)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[:n, 0], A[:n, 2] = K[0, 0], K[0, 2] - u
    b[:n] = (u - K[0, 2]) * p[:, 2] - K[0, 0] * p[:, 0]
    A[n:, 1], A[n:, 2] = K[1, 1], K[1, 2] - v
    b[n:] = (v - K[1, 2]) * p[:, 2] - K[1, 1] * p[:, 1]
    return A, b


def pose_from_rotation(
    corr: Correspondences, K: np.ndarray, R: np.ndarray, seed: int = 0, trials: int = 100
):
    """Translation for a given rotation: RANSAC over pairs of correspondences (4 equations for the
    3 unknowns), then weighted least squares on the inliers (weights 1/z turn the algebraic
    residual into a pixel one). Returns (R, t, n_inliers) or None."""
    if len(corr) < MIN_POINTS:
        return None
    A, b = _translation_system(corr, K, R)
    n = len(corr)
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, n, size=(trials, 2))
    best_t, best_count = None, -1
    for i, j in pairs:
        if i == j:
            continue
        rows = [i, j, n + i, n + j]
        t, *_ = np.linalg.lstsq(A[rows], b[rows], rcond=None)
        count = int((reprojection_errors(corr, K, R, t) < REPROJ_PX).sum())
        if count > best_count:
            best_t, best_count = t, count
    if best_t is None or best_count < MIN_POINTS:
        return None
    t = best_t
    for _ in range(3):
        inliers = reprojection_errors(corr, K, R, t) < REPROJ_PX
        if inliers.sum() < MIN_POINTS:
            return None
        z = np.maximum((corr.pts3d[inliers] @ R.T + t)[:, 2], 1e-6)
        w = np.concatenate([1.0 / z, 1.0 / z])
        rows = np.concatenate([np.flatnonzero(inliers), n + np.flatnonzero(inliers)])
        t, *_ = np.linalg.lstsq(A[rows] * w[:, None], b[rows] * w, rcond=None)
    return R, t, int((reprojection_errors(corr, K, R, t) < REPROJ_PX).sum())


def estimate_pose(
    mode: str,
    corr: Correspondences,
    K: np.ndarray,
    R_dr: np.ndarray,
    t_dr: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The pose of one instance under a pose mode: (R, t, info).

    info: {"fallback": True if the DR pose was returned because the mode failed, "inliers": n}.
    """
    if mode not in POSE_MODES:
        raise ValueError(f"pose mode must be one of {POSE_MODES}, got {mode!r}")
    result = None
    if mode == "pnp":
        result = pose_from_pnp(corr, K, seed)
    elif mode == "pnp_init":
        result = pose_from_init(corr, K, R_dr, t_dr)
    elif mode == "dr_rot_pnp_t":
        result = pose_from_rotation(corr, K, R_dr, seed)
    if result is None:
        return R_dr, np.asarray(t_dr, dtype=np.float64), {"fallback": mode != "dr", "inliers": 0}
    R, t, inliers = result
    return R, t, {"fallback": False, "inliers": inliers}
