import argparse
import json
import os
import sys
import urllib.request
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def check_env():
    if not os.environ.get("FAL_KEY"):
        print("ERROR: FAL_KEY environment variable is not set.")
        print("Get your key at https://fal.ai and run: export FAL_KEY=your_key_here")
        sys.exit(1)


def upload_to_fal(image_path: Path) -> str:
    # fal.ai models accept URLs, not raw bytes, so we upload the file first
    import fal_client
    return fal_client.upload_file(str(image_path))


def detect_tv(image_url: str, debug: bool = False) -> np.ndarray | None:
    """Florence-2 referring-expression segmentation to find the TV.

    Returns all polygon points merged into one array, or None if no TV found.
    We use Florence-2 here (not SAM) because it accepts a text query and gives
    us a coarse polygon cheaply — just enough to constrain the more expensive
    SAM3 call that follows.
    """
    import fal_client
    result = fal_client.run(
        "fal-ai/florence-2-large/referring-expression-segmentation",
        arguments={"image_url": image_url, "text_input": "tv"},
    )
    if debug:
        print(f"  [debug] TV SEG response:\n{json.dumps(result, indent=2, default=str)[:3000]}")
    polygons = (result.get("results") or {}).get("polygons") or []
    all_pts = []
    for poly in polygons:
        for p in poly.get("points", []):
            all_pts.append([p["x"], p["y"]])
    if len(all_pts) < 4:
        return None
    return np.array(all_pts, dtype=np.float32)


def _decode_mask_bytes(mask_bytes: bytes) -> np.ndarray | None:
    """Decode a downloaded mask image to a single-channel uint8 array."""
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        # Already grayscale
        return img
    if img.shape[2] == 4:
        # RGBA: SAM often encodes the mask in the alpha channel
        alpha = img[:, :, 3]
        if alpha.min() < alpha.max():
            return alpha
        # Uniform alpha means the mask is in RGB instead
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_tv_body_mask(
    image_url: str,
    tv_pts: np.ndarray,
    image_shape: tuple[int, int],
    debug: bool = False,
) -> np.ndarray | None:
    """
    Segment the TV BODY (screen + bezel + speaker) using SAM 3 with prompt 'tv'
    constrained by the Florence-2 TV bounding box. Returns a single-channel
    uint8 mask at the original image resolution, or None.
    """
    import fal_client

    h_img, w_img = image_shape
    # Add a small margin so SAM sees a bit of context beyond the tight bbox
    margin_x = (tv_pts[:, 0].max() - tv_pts[:, 0].min()) * 0.05
    margin_y = (tv_pts[:, 1].max() - tv_pts[:, 1].min()) * 0.05
    x_min = max(0, int(tv_pts[:, 0].min() - margin_x))
    y_min = max(0, int(tv_pts[:, 1].min() - margin_y))
    x_max = min(w_img - 1, int(tv_pts[:, 0].max() + margin_x))
    y_max = min(h_img - 1, int(tv_pts[:, 1].max() + margin_y))

    result = fal_client.run(
        "fal-ai/sam-3/image",
        arguments={
            "image_url": image_url,
            "prompt": "tv",
            "box_prompts": [{
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }],
            "apply_mask": False,       # return raw mask, not a masked image
            "return_multiple_masks": True,  # SAM may split body parts into separate masks
            "max_masks": 3,
            "include_scores": True,
        },
    )

    if debug:
        print(f"  [debug] SAM3 response (top-level keys): {list(result.keys())}")
        if "scores" in result:
            print(f"  [debug] Mask scores: {result.get('scores')}")
        n_masks = len(result.get("masks") or [])
        print(f"  [debug] Got {n_masks} masks from SAM3")

    masks_list = result.get("masks") or []
    scores_list = result.get("scores") or [None] * len(masks_list)
    # Pad scores list if shorter than masks list (API inconsistency guard)
    if len(scores_list) < len(masks_list):
        scores_list = scores_list + [None] * (len(masks_list) - len(scores_list))
    if not masks_list:
        return None

    tv_bbox_area = float((x_max - x_min) * (y_max - y_min))
    # Evaluate highest-confidence masks first
    sorted_pairs = sorted(
        zip(masks_list, scores_list),
        key=lambda p: -(p[1] if p[1] is not None else -1.0),
    )

    for mask_info, score in sorted_pairs:
        mask_url = mask_info.get("url") if isinstance(mask_info, dict) else None
        if not mask_url:
            continue
        try:
            with urllib.request.urlopen(mask_url, timeout=30) as r:
                mask_bytes = r.read()
        except Exception as e:
            if debug:
                print(f"  [debug] Failed to download mask: {e}")
            continue

        mask = _decode_mask_bytes(mask_bytes)
        if mask is None:
            continue

        # Reject masks that are implausibly tiny (noise) or larger than the bbox
        # (hallucination covering unrelated areas)
        mask_area = float((mask > 127).sum())
        ratio = mask_area / max(tv_bbox_area, 1.0)
        if debug:
            print(f"  [debug] Candidate mask: score={score}, area_ratio={ratio:.3f}")
        if ratio < 0.10 or ratio > 1.5:
            if debug:
                print(f"  [debug] → reject (ratio out of range)")
            continue
        return mask

    if debug:
        print("  [debug] No SAM3 mask passed sanity checks")
    return None


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
    min_line_len = max(20, min_dim // 8)   # minimum segment to register as a line
    max_line_gap = max(5, min_dim // 40)   # maximum gap to bridge within one segment

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
    ANGLE_HORIZ = 25.0       # ≤ this from horizontal → treat as horizontal
    ANGLE_VERT_MIN = 65.0    # ≥ this from horizontal → treat as vertical
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

    # Split horizontal lines into top/bottom and vertical lines into left/right
    # using the weighted median of their midpoints as the dividing line
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

    # ---- Per-side: cluster co-linear segments, pick innermost cluster, TLS-fit ----
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
        vx, vy = eigvecs[:, -1]   # largest eigenvector = line direction
        a, b = -vy, vx            # normal to the direction vector
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

        # Sort by offset and split into clusters wherever there's a gap > cluster_tol
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
            # Score = total length / distance from center: prefers long segments near center
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
    chosen_idx = {}  # for debug viz
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
        if abs(det) < 1e-9:  # lines are parallel
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
    # A non-convex quad means the lines crossed in the wrong order — discard
    if not cv2.isContourConvex(quad_crop.astype(np.int32)):
        return None
    # Screen area should be a substantial fraction of the crop, but not exceed it
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
        # Faint: all bucketed segments colored by side
        for name, (g_lines, _) in sides.items():
            for x1s, y1s, x2s, y2s in g_lines.astype(int):
                cv2.line(vis, (x1s, y1s), (x2s, y2s), colors[name], 1)
        # Bright white: the chosen cluster per side
        for name, idx in chosen_idx.items():
            g_lines = sides[name][0][idx]
            for x1s, y1s, x2s, y2s in g_lines.astype(int):
                cv2.line(vis, (x1s, y1s), (x2s, y2s), (255, 255, 255), 2)
        cv2.polylines(vis, [quad_crop.astype(np.int32)], True, (255, 255, 255), 2)
        for p in quad_crop.astype(int):
            cv2.circle(vis, tuple(p), 4, (255, 255, 255), -1)
        save_debug("15_screen_quad", vis)

    # ---- Back to original image coordinates ----
    # quad_crop is relative to the (x0, y0) crop origin
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
    # Map the four overlay corners to the four detected screen corners (TL/TR/BR/BL)
    src_pts = np.float32([
        [0, 0],
        [w_ov - 1, 0],
        [w_ov - 1, h_ov - 1],
        [0, h_ov - 1],
    ])
    M = cv2.getPerspectiveTransform(src_pts, corners)
    warped = cv2.warpPerspective(overlay_img, M, (w_in, h_in))
    # Use the filled screen quad mask to composite: warped inside, original outside
    _, screen_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.merge([screen_mask, screen_mask, screen_mask])
    result = np.where(mask_3ch > 0, warped, input_img)
    return result.astype(np.uint8)


def process_image(
    input_path: Path,
    output_dir: Path,
    overlay_path: Path,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> tuple[bool, list[str]]:
    logs = [f"\nProcessing: {input_path.name}"]
    input_img = cv2.imread(str(input_path))
    overlay_img = cv2.imread(str(overlay_path))
    if input_img is None:
        logs.append(f"  ERROR: could not read {input_path}")
        return False, logs
    if overlay_img is None:
        logs.append(f"  ERROR: could not read overlay {overlay_path}")
        return False, logs
    h_img, w_img = input_img.shape[:2]

    logs.append("  Uploading to fal.ai…")
    try:
        image_url = upload_to_fal(input_path)
    except Exception as e:
        logs.append(f"  ERROR uploading: {e}")
        return False, logs

    logs.append("  Detecting TV with Florence-2…")
    try:
        tv_pts = detect_tv(image_url, debug=debug)
    except Exception as e:
        logs.append(f"  ERROR running TV detection: {e}")
        return False, logs

    if tv_pts is None:
        logs.append("  No TV detected — skipping.")
        return False, logs

    if debug_dir is not None:
        subdir = debug_dir / input_path.stem
        subdir.mkdir(parents=True, exist_ok=True)
        prompt_vis = input_img.copy()
        x_min, y_min = int(tv_pts[:, 0].min()), int(tv_pts[:, 1].min())
        x_max, y_max = int(tv_pts[:, 0].max()), int(tv_pts[:, 1].max())
        cv2.rectangle(prompt_vis, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
        cv2.polylines(prompt_vis, [tv_pts.astype(np.int32)], True, (255, 0, 0), 2)
        cv2.imwrite(
            str(subdir / f"{input_path.stem}_debug_09_sam3_prompt{input_path.suffix}"),
            prompt_vis,
        )

    logs.append("  Segmenting TV body with SAM 3 (text prompt: 'tv')…")
    corners = None
    try:
        body_mask = detect_tv_body_mask(image_url, tv_pts, (h_img, w_img), debug=debug)
    except Exception as e:
        logs.append(f"  WARNING: SAM3 call failed: {e}")
        body_mask = None

    if body_mask is not None:
        corners = fit_screen_quad(
            body_mask,
            input_img,
            tv_pts,
            debug_dir=debug_dir,
            image_stem=input_path.stem,
            image_suffix=input_path.suffix,
        )
        if corners is None:
            logs.append("  WARNING: screen detection failed.")
    else:
        logs.append("  WARNING: no usable SAM3 mask; falling back to TV polygon bounds.")

    if corners is None:
        logs.append("  WARNING: No usable corners, saving original image")
        #out_path = output_dir / input_path.name
        #cv2.imwrite(str(out_path), input_img)
        #logs.append(f"  Saved → {out_path}")
        return True, logs

    logs.append(
        f"  Screen corners (TL/TR/BR/BL):\n"
        f"    {corners[0].astype(int).tolist()}\n"
        f"    {corners[1].astype(int).tolist()}\n"
        f"    {corners[2].astype(int).tolist()}\n"
        f"    {corners[3].astype(int).tolist()}"
    )

    # Rasterise the detected screen quad into a mask for compositing
    screen_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(screen_mask, [corners.astype(np.int32)], 255)

    if debug_dir is not None:
        subdir = debug_dir / input_path.stem
        subdir.mkdir(parents=True, exist_ok=True)
        vis = input_img.copy()
        pts_i = corners.astype(np.int32)
        cv2.polylines(vis, [pts_i], True, (0, 255, 0), 3)
        for pt in pts_i:
            cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)
        cv2.imwrite(
            str(subdir / f"{input_path.stem}_debug_15_final_quad{input_path.suffix}"),
            vis,
        )

    output_img = warp_and_composite(input_img, overlay_img, corners, screen_mask)
    out_path = output_dir / input_path.name
    cv2.imwrite(str(out_path), output_img)
    logs.append(f"  Saved → {out_path}")
    return True, logs


def main():
    parser = argparse.ArgumentParser(description="Replace TV screens with an overlay image.")
    parser.add_argument("--input_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    parser.add_argument("--overlay", default="image.jpg", help="Overlay image (default: image.jpg)")
    parser.add_argument("--debug", action="store_true", help="Print raw fal.ai API responses and save debug images")
    args = parser.parse_args()

    check_env()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    overlay_path = Path(args.overlay)

    if not input_dir.exists():
        print(f"ERROR: --input_dir does not exist: {input_dir}")
        sys.exit(1)
    if not overlay_path.exists():
        print(f"ERROR: overlay image not found: {overlay_path}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir: Path | None = None
    if args.debug:
        debug_dir = Path("debug_output")
        debug_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(f for f in input_dir.iterdir() if f.suffix.lower() in extensions)
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process")

    ok = 0
    # Use partial to bind fixed args so each worker only receives the image path.
    # Pool.imap_unordered yields results as workers finish (not in submission order),
    # which keeps the main thread from blocking on slow images.
    worker = partial(
        process_image,
        output_dir=output_dir,
        overlay_path=overlay_path,
        debug=args.debug,
        debug_dir=debug_dir,
    )
    with Pool(processes=4) as pool:
        try:
            for result, logs in pool.imap_unordered(worker, images):
                print("\n".join(logs))
                if result:
                    ok += 1
        except KeyboardInterrupt:
            print("\nInterrupted — terminating workers.")
            pool.terminate()
            sys.exit(1)

    print(f"\nDone: {ok}/{len(images)} succeeded.")


if __name__ == "__main__":
    main()
