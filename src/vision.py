from pathlib import Path

import cv2
import numpy as np


def fit_screen_quad(
    body_mask: np.ndarray,
    input_img: np.ndarray,
    tv_pts: np.ndarray,
    debug_dir: Path | None = None,
    image_stem: str = "",
    image_suffix: str = ".jpg",
) -> np.ndarray | None:
    """
    Find the screen inside SAM's TV body mask.

    Pipeline:
      1. Clean the SAM body mask (intersect with Florence-2 polygon, pick best CC).
      2. Crop to the body bbox.
      3. Compute a local-std map; the bezel-to-screen transition shows up as a
         rectangular ring of high-std pixels.
      4. Canny → Hough segments on that ring.
      5. Bucket segments into top/bottom/left/right by angle + position.
      6. Per side, cluster co-linear segments and pick the INNERMOST cluster
         (the screen is the innermost rectangle on the TV face).
      7. TLS-fit a line through each chosen cluster.
      8. Intersect adjacent sides to get four screen corners (in original-image
         coordinates).
    """

    def save_debug(name: str, img: np.ndarray):
        if debug_dir is None:
            return
        subdir = debug_dir / image_stem
        subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(subdir / f"{image_stem}_debug_{name}{image_suffix}"), img)

    h, w = body_mask.shape[:2]

    # ---- Clean SAM body mask ----
    _, binary = cv2.threshold(body_mask, 127, 255, cv2.THRESH_BINARY)

    # Intersect with (dilated) Florence-2 polygon to clip extraneous segmentations.
    # SAM sometimes bleeds into adjacent furniture; the Florence-2 polygon is a
    # reliable outer bound for the TV region.
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [tv_pts.astype(np.int32)], 255)
    poly_dilate_k = max(7, min(h, w) // 200)  # scale dilation with image size
    poly_mask = cv2.dilate(poly_mask, np.ones((poly_dilate_k, poly_dilate_k), np.uint8))
    binary = cv2.bitwise_and(binary, poly_mask)

    # Pick the connected component most likely to be the TV body:
    # large area + centroid close to the Florence-2 centroid
    tv_cx = float(tv_pts[:, 0].mean())
    tv_cy = float(tv_pts[:, 1].mean())
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None

    best_label = None
    best_score = -float("inf")
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < h * w * 0.0005:  # ignore tiny noise specks
            continue
        cx, cy = centroids[i]
        dist = np.hypot(cx - tv_cx, cy - tv_cy)
        # Penalise distance proportionally to area so large off-center blobs lose to
        # smaller on-center ones only when they're very far away
        score = area - dist * area / 100.0
        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return None

    tv_body = np.where(labels == best_label, 255, 0).astype(np.uint8)
    # Close small holes in the mask (gaps between bezel segments, etc.)
    tv_body = cv2.morphologyEx(tv_body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    save_debug("10_tv_body", tv_body)

    # ---- Crop to TV body bbox ----
    ys, xs = np.where(tv_body > 0)
    if len(xs) < 100:
        return None
    body_w = int(xs.max() - xs.min())
    body_h = int(ys.max() - ys.min())
    margin = max(5, int(min(body_w, body_h) * 0.03))
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(w - 1, int(xs.max()) + margin)
    y1 = min(h - 1, int(ys.max()) + margin)
    crop_h = y1 - y0 + 1
    crop_w = x1 - x0 + 1
    if crop_h < 30 or crop_w < 30:
        return None

    img_crop = input_img[y0:y1 + 1, x0:x1 + 1]
    mask_crop = tv_body[y0:y1 + 1, x0:x1 + 1]
    save_debug("11_tv_crop", img_crop)

    # ---- Local std map ----
    # The bezel-to-screen boundary is a sharp color transition that shows up as a
    # ring of high local standard deviation. We use boxFilter (fast integer box blur)
    # rather than Gaussian because we only need an approximate std map.
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(7, min(crop_h, crop_w) // 60)  # kernel scales with crop size
    if k % 2 == 0:
        k += 1  # boxFilter requires odd kernel for consistent centering
    mean_f = cv2.boxFilter(gray, ddepth=-1, ksize=(k, k))
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(k, k))
    # Var(X) = E[X²] - E[X]²; clip handles floating-point negatives near zero
    var = np.clip(mean_sq - mean_f * mean_f, 0, None)
    std = np.sqrt(var)
    std_vis = np.clip(std * 4, 0, 255).astype(np.uint8)

    if debug_dir is not None:
        std_vis_dbg = std_vis.copy()
        std_vis_dbg[mask_crop == 0] = 0  # show std only inside the TV body
        save_debug("12_std_map", std_vis_dbg)

    # ---- Canny edges ----
    edges = cv2.Canny(std_vis, 100, 200)
    save_debug("13_edges", edges)

    # ---- Hough lines ----
    min_dim = min(crop_h, crop_w)
    min_line_len = max(20, min_dim // 8)
    max_line_gap = max(5, min_dim // 40)

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if raw is None or len(raw) < 4:
        return None

    segs = raw.reshape(-1, 4).astype(np.float32)
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    lens = np.hypot(dx, dy)
    # Remap angle to (-90, 90] so 0° = horizontal and ±90° = vertical
    angles = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0

    if debug_dir is not None:
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for x1s, y1s, x2s, y2s in segs.astype(int):
            cv2.line(vis, (x1s, y1s), (x2s, y2s), (0, 255, 0), 1)
        save_debug("14_hough_raw", vis)

    # ---- Bucket into top/bottom/left/right ----
    ANGLE_HORIZ = 25.0
    ANGLE_VERT_MIN = 65.0
    # Segments between 25° and 65° are likely perspective distortion artifacts — drop them
    h_mask = np.abs(angles) < ANGLE_HORIZ
    v_mask = np.abs(angles) > ANGLE_VERT_MIN
    h_lines, h_lens = segs[h_mask], lens[h_mask]
    v_lines, v_lens = segs[v_mask], lens[v_mask]
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    def weighted_median(values, weights):
        # Length-weighted median is more robust than mean as a split point because
        # it isn't skewed by many short noisy segments on one side
        order = np.argsort(values)
        v, w_ = values[order], weights[order]
        cw = np.cumsum(w_)
        return v[np.searchsorted(cw, cw[-1] * 0.5)]

    h_mid_y = (h_lines[:, 1] + h_lines[:, 3]) * 0.5
    v_mid_x = (v_lines[:, 0] + v_lines[:, 2]) * 0.5
    split_y = weighted_median(h_mid_y, h_lens)
    split_x = weighted_median(v_mid_x, v_lens)

    sides = {
        "top":    (h_lines[h_mid_y <  split_y], h_lens[h_mid_y <  split_y]),
        "bottom": (h_lines[h_mid_y >= split_y], h_lens[h_mid_y >= split_y]),
        "left":   (v_lines[v_mid_x <  split_x], v_lens[v_mid_x <  split_x]),
        "right":  (v_lines[v_mid_x >= split_x], v_lens[v_mid_x >= split_x]),
    }
    if any(len(s[0]) == 0 for s in sides.values()):
        return None

    def fit_line(side_segs, side_lens):
        """Total-least-squares line fit. Returns (a, b, c) with a*x+b*y+c=0.

        TLS is used instead of ordinary least-squares because vertical lines have
        infinite slope and would cause OLS to fail or give poor results.
        """
        pts = np.vstack([side_segs[:, :2], side_segs[:, 2:]]).astype(np.float64)
        # Weight each point by its segment length so long segments pull the fit harder
        w_ = np.concatenate([side_lens, side_lens]).astype(np.float64) * 0.5
        W = w_.sum()
        cx, cy = (w_ * pts[:, 0]).sum() / W, (w_ * pts[:, 1]).sum() / W
        xs_, ys_ = pts[:, 0] - cx, pts[:, 1] - cy
        # Build the 2×2 weighted covariance matrix; its principal eigenvector
        # is the direction of the best-fit line
        M = np.array([
            [(w_ * xs_ * xs_).sum(), (w_ * xs_ * ys_).sum()],
            [(w_ * xs_ * ys_).sum(), (w_ * ys_ * ys_).sum()],
        ])
        _, eigvecs = np.linalg.eigh(M)
        vx, vy = eigvecs[:, -1]
        a, b = -vy, vx
        n = np.hypot(a, b)
        a, b = a / n, b / n
        c = -(a * cx + b * cy)
        return np.array([a, b, c])

    def fit_side_innermost(side_segs, side_lens, axis, center_xy, cluster_tol):
        """
        Cluster segments along the perpendicular axis, then pick the cluster
        with best (total_length / mean_distance_from_center). The screen is
        the innermost rectangle on the TV face, so the cluster closest to the
        center that still has substantial total length is the screen edge.
        """
        cx, cy = center_xy
        if axis == "h":
            offsets = (side_segs[:, 1] + side_segs[:, 3]) * 0.5 - cy
        else:
            offsets = (side_segs[:, 0] + side_segs[:, 2]) * 0.5 - cx

        order = np.argsort(offsets)
        sorted_off = offsets[order]
        if len(sorted_off) == 1:
            groups = [order]
        else:
            splits = np.where(np.diff(sorted_off) > cluster_tol)[0] + 1
            groups = np.split(order, splits)

        best = None
        best_score = -np.inf
        for g in groups:
            total_len = side_lens[g].sum()
            if total_len < 10:
                continue
            mean_abs_off = np.abs(offsets[g]).mean()
            score = total_len / (mean_abs_off + 1.0)
            if score > best_score:
                best_score = score
                best = g

        if best is None or len(best) == 0:
            return None, None
        return fit_line(side_segs[best], side_lens[best]), best

    center_xy = (crop_w * 0.5, crop_h * 0.5)
    # Tolerance for merging nearby parallel segments into one cluster; scales with crop size
    cluster_tol = max(4, min(crop_h, crop_w) // 50)

    fits = {}
    chosen_idx = {}
    for name, (segs_s, lens_s) in sides.items():
        axis = "h" if name in ("top", "bottom") else "v"
        L, idx = fit_side_innermost(segs_s, lens_s, axis, center_xy, cluster_tol)
        if L is None:
            return None
        fits[name] = L
        chosen_idx[name] = idx

    # ---- Intersect adjacent sides for corners ----
    def intersect(L1, L2):
        """Solve the 2×2 linear system a1*x+b1*y+c1=0, a2*x+b2*y+c2=0."""
        a1, b1, c1 = L1
        a2, b2, c2 = L2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-9:
            return None
        return np.array([
            (b1 * c2 - b2 * c1) / det,
            (a2 * c1 - a1 * c2) / det,
        ])

    tl = intersect(fits["top"],    fits["left"])
    tr = intersect(fits["top"],    fits["right"])
    br = intersect(fits["bottom"], fits["right"])
    bl = intersect(fits["bottom"], fits["left"])
    if any(c is None for c in (tl, tr, br, bl)):
        return None
    quad_crop = np.array([tl, tr, br, bl], dtype=np.float32)

    # ---- Sanity checks ----
    # Allow corners to fall slightly outside the crop (perspective distortion can push them)
    slack = max(crop_h, crop_w) * 0.10
    if (quad_crop[:, 0].min() < -slack or quad_crop[:, 0].max() > crop_w + slack or
        quad_crop[:, 1].min() < -slack or quad_crop[:, 1].max() > crop_h + slack):
        return None
    if not cv2.isContourConvex(quad_crop.astype(np.int32)):
        return None
    area = cv2.contourArea(quad_crop)
    if area < 0.10 * crop_h * crop_w or area > 1.05 * crop_h * crop_w:
        return None

    # ---- Debug viz ----
    if debug_dir is not None:
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        colors = {
            "top":    (0, 0, 255),
            "bottom": (255, 0, 0),
            "left":   (0, 255, 0),
            "right":  (0, 255, 255),
        }
        for name, (g_lines, _) in sides.items():
            for x1s, y1s, x2s, y2s in g_lines.astype(int):
                cv2.line(vis, (x1s, y1s), (x2s, y2s), colors[name], 1)
        for name, idx in chosen_idx.items():
            g_lines = sides[name][0][idx]
            for x1s, y1s, x2s, y2s in g_lines.astype(int):
                cv2.line(vis, (x1s, y1s), (x2s, y2s), (255, 255, 255), 2)
        cv2.polylines(vis, [quad_crop.astype(np.int32)], True, (255, 255, 255), 2)
        for p in quad_crop.astype(int):
            cv2.circle(vis, tuple(p), 4, (255, 255, 255), -1)
        save_debug("15_screen_quad", vis)

    # ---- Back to original image coordinates ----
    quad_full = quad_crop.copy()
    quad_full[:, 0] += x0
    quad_full[:, 1] += y0
    return quad_full


def warp_and_composite(
    input_img: np.ndarray,
    overlay_img: np.ndarray,
    corners: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Perspective-warp the overlay image onto the screen region and composite it."""
    h_in, w_in = input_img.shape[:2]
    h_ov, w_ov = overlay_img.shape[:2]
    src_pts = np.float32([
        [0, 0],
        [w_ov - 1, 0],
        [w_ov - 1, h_ov - 1],
        [0, h_ov - 1],
    ])
    M = cv2.getPerspectiveTransform(src_pts, corners)
    warped = cv2.warpPerspective(overlay_img, M, (w_in, h_in))
    _, screen_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([screen_mask, screen_mask, screen_mask])
    result = np.where(mask_3ch > 0, warped, input_img)
    return result.astype(np.uint8)
