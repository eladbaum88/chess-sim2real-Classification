"""
woelflein_crops.py — board localisation + warp + crop (inference subset).

Ported from the chesscog pipeline (Wölflein & Arandjelović 2021, J. Imaging;
https://github.com/georg-wolflein/chesscog, MIT). This is the trimmed,
self-contained inference subset of the project's `verify_woelflein_crops.py`:
only `find_corners`, `warp_chessboard_image`, `crop_square` and their helpers.
The diagnostic/visualisation code and all absolute project paths have been
removed so this module is portable (no dependency on the rest of the repo).

Pipeline:
  1. find_corners(img_bgr) — Canny edges -> Hough lines -> agglomerative cluster
     into horiz/vert -> DBSCAN dedup -> RANSAC homography of grid intersections
     -> Sobel border refinement -> 4 outer corner coords [TL, TR, BR, BL].
  2. warp_chessboard_image(img, corners) — perspective warp to IMG_SIZE x IMG_SIZE
     = 500x500 with the 8x8 board at the inner [50..450, 50..450] region.
  3. crop_square(warped, row, col) — 100x100 (2x2-square) patch centred on a square.
"""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


# --------------------------------------------------------------------------
# Config — flattened from chesscog's config/corner_detection.yaml.
# EDGE_DETECTION thresholds retuned (LOW=30/HIGH=90) for the project renders;
# everything else left at chesscog defaults.
# --------------------------------------------------------------------------
CFG = {
    "RESIZE_IMAGE_WIDTH": 1200,
    "EDGE_DETECTION": {
        "APERTURE": 3,
        "HIGH_THRESHOLD": 90,
        "LOW_THRESHOLD": 30,
    },
    "LINE_DETECTION": {
        "THRESHOLD": 150,
        "DIAGONAL_LINE_ELIMINATION": True,
        "DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES": 30,
    },
    "BORDER_REFINEMENT": {
        "LINE_WIDTH": 4,
        "WARPED_SQUARE_SIZE": (50, 50),
        "NUM_SURROUNDING_SQUARES_IN_WARPED_IMG": 5,
        "SOBEL_KERNEL_SIZE": 3,
        "EDGE_DETECTION_HORIZONTAL": {
            "APERTURE": 3,
            "HIGH_THRESHOLD": 300,
            "LOW_THRESHOLD": 120,
        },
        "EDGE_DETECTION_VERTICAL": {
            "APERTURE": 3,
            "HIGH_THRESHOLD": 200,
            "LOW_THRESHOLD": 100,
        },
    },
    "MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE": 0.7,
    "RANSAC_BEST_SOLUTION_TOLERANCE": 0.15,
    "RANSAC_OFFSET_TOLERANCE": 0.1,
}

# Warp / crop constants — from chesscog create_dataset.py
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE             # 400
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE  # 500 — board at inner [50..450]


# ==========================================================================
# Ported from chesscog/core/coordinates.py
# ==========================================================================
def to_homogenous_coordinates(coords):
    return np.concatenate(
        [coords, np.ones((*coords.shape[:-1], 1))], axis=-1)


def from_homogenous_coordinates(coords):
    return coords[..., :2] / coords[..., 2, np.newaxis]


# ==========================================================================
# Ported from chesscog/core/__init__.py
# ==========================================================================
def sort_corner_points(points: np.ndarray) -> np.ndarray:
    """Order corners as [TL, TR, BR, BL] by image position."""
    points = points[points[:, 1].argsort()]
    points[:2] = points[:2][points[:2, 0].argsort()]
    points[2:] = points[2:][points[2:, 0].argsort()[::-1]]
    return points


# ==========================================================================
# Ported from chesscog/corner_detection/detect_corners.py
# ==========================================================================
class ChessboardNotLocatedException(Exception):
    pass


def _resize_image(img):
    h, w = img.shape[:2]
    if w == CFG["RESIZE_IMAGE_WIDTH"]:
        return img, 1.0
    scale = CFG["RESIZE_IMAGE_WIDTH"] / w
    dims = (CFG["RESIZE_IMAGE_WIDTH"], int(h * scale))
    return cv2.resize(img, dims), scale


def _detect_edges(edge_cfg, gray):
    if gray.dtype != np.uint8:
        gray = (gray / gray.max() * 255).astype(np.uint8)
    return cv2.Canny(
        gray, edge_cfg["LOW_THRESHOLD"], edge_cfg["HIGH_THRESHOLD"],
        apertureSize=edge_cfg["APERTURE"])


def _fix_negative_rho(lines):
    lines = lines.copy()
    neg = lines[..., 0] < 0
    lines[neg, 0] = -lines[neg, 0]
    lines[neg, 1] = lines[neg, 1] - np.pi
    return lines


def _detect_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 360, CFG["LINE_DETECTION"]["THRESHOLD"])
    if lines is None:
        raise ChessboardNotLocatedException("no Hough lines")
    lines = lines.squeeze(axis=-2)
    lines = _fix_negative_rho(lines)
    if CFG["LINE_DETECTION"]["DIAGONAL_LINE_ELIMINATION"]:
        thr = np.deg2rad(CFG["LINE_DETECTION"]["DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES"])
        vmask = np.abs(lines[:, 1]) < thr
        hmask = np.abs(lines[:, 1] - np.pi / 2) < thr
        lines = lines[vmask | hmask]
    return lines


def _absolute_angle_difference(x, y):
    diff = np.mod(np.abs(x - y), 2 * np.pi)
    return np.min(np.stack([diff, np.pi - diff], axis=-1), axis=-1)


def _sort_lines(lines):
    if lines.ndim == 0 or lines.shape[-2] == 0:
        return lines
    return lines[np.argsort(lines[..., 0])]


def _cluster_horizontal_and_vertical_lines(lines):
    lines = _sort_lines(lines)
    thetas = lines[..., 1].reshape(-1, 1)
    # Vectorised inline of chesscog's pairwise angle distance (its Python-callable
    # metric broke in newer sklearn). Output identical.
    t = thetas.ravel()
    d = np.abs(t[:, None] - t[None, :]) % (2 * np.pi)
    distance_matrix = np.minimum(d, np.pi - d)
    # sklearn >=1.4 renamed `affinity` to `metric`.
    try:
        agg = AgglomerativeClustering(
            n_clusters=2, metric="precomputed", linkage="average")
    except TypeError:
        agg = AgglomerativeClustering(
            n_clusters=2, affinity="precomputed", linkage="average")
    clusters = agg.fit_predict(distance_matrix)
    angle_with_y_axis = _absolute_angle_difference(thetas, 0.)
    if angle_with_y_axis[clusters == 0].mean() > angle_with_y_axis[clusters == 1].mean():
        hcluster, vcluster = 0, 1
    else:
        hcluster, vcluster = 1, 0
    return lines[clusters == hcluster], lines[clusters == vcluster]


def _get_intersection_point(rho1, theta1, rho2, theta2):
    cos_t1 = np.cos(theta1)
    cos_t2 = np.cos(theta2)
    sin_t1 = np.sin(theta1)
    sin_t2 = np.sin(theta2)
    x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2)
    y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2)
    return x, y


def _eliminate_similar_lines(lines, perpendicular_lines):
    perp_rho, perp_theta = perpendicular_lines.mean(axis=0)
    rho, theta = np.moveaxis(lines, -1, 0)
    pts = np.stack(_get_intersection_point(rho, theta, perp_rho, perp_theta),
                   axis=-1)
    clustering = DBSCAN(eps=12, min_samples=1).fit(pts)
    out = []
    for c in range(clustering.labels_.max() + 1):
        in_cluster = lines[clustering.labels_ == c]
        rho_c = in_cluster[..., 0]
        median = np.argsort(rho_c)[len(rho_c) // 2]
        out.append(in_cluster[median])
    return np.stack(out)


def _choose_from_range(upper, n=2):
    return np.sort(np.random.choice(np.arange(upper), (n,), replace=False),
                   axis=-1)


def _get_intersection_points(h_lines, v_lines):
    rho1, theta1 = np.moveaxis(h_lines, -1, 0)
    rho2, theta2 = np.moveaxis(v_lines, -1, 0)
    rho1, rho2 = np.meshgrid(rho1, rho2, indexing="ij")
    theta1, theta2 = np.meshgrid(theta1, theta2, indexing="ij")
    return np.stack(_get_intersection_point(rho1, theta1, rho2, theta2), axis=-1)


def _compute_transformation_matrix(src, dst):
    M, _ = cv2.findHomography(src.reshape(-1, 2), dst.reshape(-1, 2))
    return M


def _compute_homography(intersection_points, row1, row2, col1, col2):
    p1 = intersection_points[row1, col1]
    p2 = intersection_points[row1, col2]
    p3 = intersection_points[row2, col2]
    p4 = intersection_points[row2, col1]
    src = np.stack([p1, p2, p3, p4])
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    return _compute_transformation_matrix(src, dst)


def _warp_points(M, points):
    p = to_homogenous_coordinates(points)
    return from_homogenous_coordinates(p @ M.T)


def _find_best_scale(values, scales=np.arange(1, 9)):
    scales = np.sort(scales)
    scaled = np.expand_dims(values, -1) * scales
    diff = np.abs(np.rint(scaled) - scaled)
    inlier_mask = diff < CFG["RANSAC_OFFSET_TOLERANCE"] / scales
    num = np.sum(inlier_mask, axis=tuple(range(inlier_mask.ndim - 1)))
    best = np.max(num)
    idx = np.argmax(num > (1 - CFG["RANSAC_BEST_SOLUTION_TOLERANCE"]) * best)
    return scales[idx], inlier_mask[..., idx]


def _discard_outliers(warped_points, intersection_points):
    h_scale, h_mask = _find_best_scale(warped_points[..., 0])
    v_scale, v_mask = _find_best_scale(warped_points[..., 1])
    mask = h_mask & v_mask
    n_rows = np.any(mask, axis=-1).sum()
    n_cols = np.any(mask, axis=-2).sum()
    keep_rows = mask.sum(-1) / n_rows > CFG["MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE"]
    keep_cols = mask.sum(-2) / n_cols > CFG["MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE"]
    return (warped_points[keep_rows][:, keep_cols],
            intersection_points[keep_rows][:, keep_cols],
            h_scale, v_scale)


def _quantize_points(warped_scaled, intersection_points):
    mean_col_xs = warped_scaled[..., 0].mean(axis=0)
    mean_row_ys = warped_scaled[..., 1].mean(axis=1)
    col_xs = np.rint(mean_col_xs).astype(np.int32)
    row_ys = np.rint(mean_row_ys).astype(np.int32)
    col_xs, col_idx = np.unique(col_xs, return_index=True)
    row_ys, row_idx = np.unique(row_ys, return_index=True)
    intersection_points = intersection_points[row_idx][:, col_idx]
    xmin, xmax = col_xs.min(), col_xs.max()
    ymin, ymax = row_ys.min(), row_ys.max()
    while xmax - xmin > 8:
        xmax -= 1
        xmin += 1
    while ymax - ymin > 8:
        ymax -= 1
        ymin += 1
    col_mask = (col_xs >= xmin) & (col_xs <= xmax)
    row_mask = (row_ys >= xmin) & (row_ys <= xmax)
    col_xs = col_xs[col_mask]
    row_ys = row_ys[row_mask]
    intersection_points = intersection_points[row_mask][:, col_mask]
    quantized_points = np.stack(np.meshgrid(col_xs, row_ys), axis=-1)
    translation = -np.array([xmin, ymin]) + \
        CFG["BORDER_REFINEMENT"]["NUM_SURROUNDING_SQUARES_IN_WARPED_IMG"]
    scale = np.array(CFG["BORDER_REFINEMENT"]["WARPED_SQUARE_SIZE"])
    scaled_quantized = (quantized_points + translation) * scale
    xmin_t, ymin_t = np.array((xmin, ymin)) + translation
    xmax_t, ymax_t = np.array((xmax, ymax)) + translation
    warped_img_size = (np.array((xmax_t, ymax_t)) +
                       CFG["BORDER_REFINEMENT"]["NUM_SURROUNDING_SQUARES_IN_WARPED_IMG"]) * scale
    return ((xmin_t, xmax_t, ymin_t, ymax_t), scale, scaled_quantized,
            intersection_points, warped_img_size)


def _compute_vertical_borders(warped, mask, scale, xmin, xmax):
    G_x = np.abs(cv2.Sobel(warped, cv2.CV_64F, 1, 0,
                           ksize=CFG["BORDER_REFINEMENT"]["SOBEL_KERNEL_SIZE"]))
    G_x[~mask] = 0
    G_x = _detect_edges(CFG["BORDER_REFINEMENT"]["EDGE_DETECTION_VERTICAL"], G_x)
    G_x[~mask] = 0
    lw = CFG["BORDER_REFINEMENT"]["LINE_WIDTH"]

    def nonmax(x):
        x = int(x * scale[0])
        return G_x[:, x - lw // 2: x + lw // 2 + 1].max(axis=1)

    while xmax - xmin < 8:
        top = nonmax(xmax + 1)
        bot = nonmax(xmin - 1)
        if top.sum() > bot.sum():
            xmax += 1
        else:
            xmin -= 1
    return xmin, xmax


def _compute_horizontal_borders(warped, mask, scale, ymin, ymax):
    G_y = np.abs(cv2.Sobel(warped, cv2.CV_64F, 0, 1,
                           ksize=CFG["BORDER_REFINEMENT"]["SOBEL_KERNEL_SIZE"]))
    G_y[~mask] = 0
    G_y = _detect_edges(CFG["BORDER_REFINEMENT"]["EDGE_DETECTION_HORIZONTAL"], G_y)
    G_y[~mask] = 0
    lw = CFG["BORDER_REFINEMENT"]["LINE_WIDTH"]

    def nonmax(y):
        y = int(y * scale[1])
        return G_y[y - lw // 2: y + lw // 2 + 1].max(axis=0)

    while ymax - ymin < 8:
        top = nonmax(ymax + 1)
        bot = nonmax(ymin - 1)
        if top.sum() > bot.sum():
            ymax += 1
        else:
            ymin -= 1
    return ymin, ymax


def find_corners(img_bgr):
    """chesscog's find_corners ported verbatim. Expects BGR input.
    Returns (4, 2) float32 corners ordered [TL, TR, BR, BL] in image pixels.
    Raises ChessboardNotLocatedException on failure."""
    img, img_scale = _resize_image(img_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = _detect_edges(CFG["EDGE_DETECTION"], gray)
    lines = _detect_lines(edges)
    if lines.shape[0] > 400:
        raise ChessboardNotLocatedException("too many lines in the image")
    all_h, all_v = _cluster_horizontal_and_vertical_lines(lines)
    h_lines = _eliminate_similar_lines(all_h, all_v)
    v_lines = _eliminate_similar_lines(all_v, all_h)
    all_pts = _get_intersection_points(h_lines, v_lines)

    best_inliers = 0
    best_cfg = None
    it = 0
    while it < 200 or best_inliers < 30:
        row1, row2 = _choose_from_range(len(h_lines))
        col1, col2 = _choose_from_range(len(v_lines))
        M = _compute_homography(all_pts, row1, row2, col1, col2)
        wp = _warp_points(M, all_pts)
        wp, ip, hs, vs = _discard_outliers(wp, all_pts)
        n = np.prod(wp.shape[:-1])
        if n > best_inliers:
            wp = wp * np.array((hs, vs))
            cfg_tuple = _quantize_points(wp, ip)
            (_xmm, _scale, qp, ip, _wsize) = cfg_tuple
            n = np.prod(qp.shape[:-1])
            if n > best_inliers:
                best_inliers = n
                best_cfg = cfg_tuple
        it += 1
        if it > 10000:
            raise ChessboardNotLocatedException("RANSAC produced no viable results")

    (xmin, xmax, ymin, ymax), scale, qp, ip, wsize = best_cfg
    M = _compute_transformation_matrix(ip, qp)
    inv_M = np.linalg.inv(M)
    dims = tuple(wsize.astype(np.int32))
    warped = cv2.warpPerspective(gray, M, dims)
    borders = np.zeros_like(gray)
    borders[3:-3, 3:-3] = 1
    warped_borders = cv2.warpPerspective(borders, M, dims)
    warped_mask = warped_borders == 1

    xmin, xmax = _compute_vertical_borders(warped, warped_mask, scale, xmin, xmax)
    sxmin, sxmax = int(xmin * scale[0]), int(xmax * scale[0])
    warped_mask[:, :sxmin] = warped_mask[:, sxmax:] = False
    ymin, ymax = _compute_horizontal_borders(warped, warped_mask, scale, ymin, ymax)

    corners = np.array([[xmin, ymin],
                        [xmax, ymin],
                        [xmax, ymax],
                        [xmin, ymax]], dtype=np.float32)
    corners = corners * scale
    img_corners = _warp_points(inv_M, corners)
    img_corners = img_corners / img_scale
    return sort_corner_points(img_corners)


# ==========================================================================
# Ported from chesscog/occupancy_classifier/create_dataset.py
# ==========================================================================
def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Verbatim from chesscog. img: H x W x 3 (any channel order). corners:
    (4, 2) in TL/TR/BR/BL order. Returns IMG_SIZE x IMG_SIZE warped board."""
    src = sort_corner_points(corners.astype(np.float32))
    dst = np.array(
        [[SQUARE_SIZE, SQUARE_SIZE],
         [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],
         [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE],
         [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]],
        dtype=np.float32,
    )
    M, _ = cv2.findHomography(src, dst)
    return cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE))


def crop_square(img: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop a 2x2-square (100x100 px) patch centred on board square (row, col).
    Verbatim slicing from chesscog/occupancy_classifier/create_dataset.py."""
    return img[int(SQUARE_SIZE * (row + 0.5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + 0.5)): int(SQUARE_SIZE * (col + 2.5))]
