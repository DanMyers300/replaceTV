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
                print(f"  [debug] → reject (area ratio out of range)")
            continue

        # Shape sanity checks — protects against Florence-2 + SAM returning a
        # bogus "tv" detection on a tall narrow appliance (microwave, dark
        # cabinet, fridge gap, etc.) when the actual TV is elsewhere in the
        # frame or absent. These checks catch the failure upstream of
        # fit_screen_quad so the pipeline saves the unmodified original
        # rather than warping the overlay onto the wrong object.
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            continue
        mask_w = float(xs.max() - xs.min() + 1)
        mask_h = float(ys.max() - ys.min() + 1)
        aspect = mask_w / max(mask_h, 1.0)
        rectangularity = mask_area / max(mask_w * mask_h, 1.0)

        # TVs are landscape — 16:9 (1.78) and 4:3 (1.33) are typical, and
        # heavy perspective foreshortening can compress this somewhat. A
        # mask that is taller than wide (aspect < ~0.9) is almost never a
        # real TV; the 1003 failure case (microwave next to fridge mis-
        # detected as a TV) produces a portrait L-shaped mask with aspect
        # near 0.3.
        if aspect < 0.9:
            if debug:
                print(f"  [debug] → reject (portrait mask, w/h={aspect:.2f}; "
                      f"TVs are landscape)")
            continue

        # Real TV masks fill their own bounding box — a clean SAM segmentation
        # of a TV is ≥ 0.90, and even imperfect ones rarely drop below 0.80.
        # The 1003 L-shaped mis-detection fills only ~0.55 of its bbox. A
        # 0.75 floor leaves room for imperfect segmentation (frayed edges,
        # missing corners) while catching the L-shape failure mode.
        if rectangularity < 0.75:
            if debug:
                print(f"  [debug] → reject (non-rectangular mask, "
                      f"fill={rectangularity:.2f}; TV masks fill their bbox)")
            continue

        return mask

    if debug:
        print("  [debug] No SAM3 mask passed sanity checks")
    return None


# ---------------------------------------------------------------------------
# Geometry helpers used by fit_screen_quad
# ---------------------------------------------------------------------------

def _order_quad_corners(quad: np.ndarray) -> np.ndarray:
    """Order 4 corners as [TL, TR, BR, BL] using x+y and x-y diagonals.

    Robust for any roughly axis-aligned quad; works for moderate tilts too.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],  # TL: smallest x+y
        pts[np.argmax(d)],  # TR: largest x-y
        pts[np.argmax(s)],  # BR: largest x+y
        pts[np.argmin(d)],  # BL: smallest x-y
    ], dtype=np.float32)


def _line_through_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray | None:
    """Return normalized line equation [a, b, c] for ax + by + c = 0."""
    a = float(p2[1] - p1[1])
    b = float(p1[0] - p2[0])
    c = -(a * float(p1[0]) + b * float(p1[1]))
    n = np.hypot(a, b)
    if n < 1e-9:
        return None
    return np.array([a / n, b / n, c / n], dtype=np.float64)


def _fit_line_tls(side_segs: np.ndarray, side_lens: np.ndarray) -> np.ndarray:
    """Total-least-squares line fit through segment endpoints.

    Returns [a, b, c] with a*x + b*y + c = 0. TLS handles vertical lines
    cleanly (OLS would blow up on infinite slope).
    """
    pts = np.vstack([side_segs[:, :2], side_segs[:, 2:]]).astype(np.float64)
    w = np.concatenate([side_lens, side_lens]).astype(np.float64) * 0.5
    W = w.sum()
    cx, cy = (w * pts[:, 0]).sum() / W, (w * pts[:, 1]).sum() / W
    xs, ys = pts[:, 0] - cx, pts[:, 1] - cy
    M = np.array([
        [(w * xs * xs).sum(), (w * xs * ys).sum()],
        [(w * xs * ys).sum(), (w * ys * ys).sum()],
    ])
    _, eigvecs = np.linalg.eigh(M)
    vx, vy = eigvecs[:, -1]  # largest eigenvector = line direction
    a, b = -vy, vx           # normal to direction
    n = np.hypot(a, b)
    a, b = a / n, b / n
    c = -(a * cx + b * cy)
    return np.array([a, b, c])


def _fit_side_outer_anchored(
    side_segs: np.ndarray,
    side_lens: np.ndarray,
    outer_line: np.ndarray,
    center_pt: tuple[float, float],
    tv_dim: float,
    cluster_tol: float,
    min_inset_frac: float = 0.035,
    max_inset_frac: float = 0.22,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Pick the DEEPEST cluster (among long-enough ones) within a plausible
    bezel-thickness inset range.

    For each segment we compute its perpendicular distance from ``outer_line``,
    signed so positive = toward ``center_pt`` (i.e. inside the TV). Segments
    are clustered by inset. Clusters outside [min_inset_frac, max_inset_frac]
    × tv_dim are rejected:

      * Below min_inset_frac: the TV outer casing / bezel-back edge. With the
        SAM-mask-derived outer_line, this is always a long, prominent cluster —
        biasing toward "outermost" or even pure length makes it always win,
        which is wrong.
      * Above max_inset_frac: screen content, reflections, branding strips
        rendered in the screen interior.

    Among the surviving clusters we apply a 0.5 × max-length floor to discard
    short content / reflection edges, then pick the cluster with the largest
    mean inset. The bezel-to-screen transition is the deepest *long* edge in
    the band by construction — it sits at the very inner boundary of the
    bezel zone. Pure length scoring (the previous heuristic) fails when an
    intermediate bezel feature (bezel-front lip, beveled outer edge) is
    slightly longer than the real screen edge; deepest-of-long picks the
    screen edge correctly while the length floor still rejects in-screen
    reflections (which are typically much shorter than the full screen edge).
    """
    if len(side_segs) == 0:
        return None, None

    a, b, c = outer_line
    pts_mid = np.column_stack([
        (side_segs[:, 0] + side_segs[:, 2]) * 0.5,
        (side_segs[:, 1] + side_segs[:, 3]) * 0.5,
    ])
    # Force sign convention: positive = toward TV center
    center_sign = a * center_pt[0] + b * center_pt[1] + c
    sgn = 1.0 if center_sign > 0 else -1.0
    dists = sgn * (a * pts_mid[:, 0] + b * pts_mid[:, 1] + c)

    # Cluster by inset distance
    order = np.argsort(dists)
    sorted_d = dists[order]
    if len(order) == 1:
        groups = [order]
    else:
        splits = np.where(np.diff(sorted_d) > cluster_tol)[0] + 1
        groups = np.split(order, splits)

    min_inset = min_inset_frac * tv_dim
    max_inset = max_inset_frac * tv_dim

    # First pass: collect every cluster that survives the length floor and
    # inset window. We need to see all of them before we can rank.
    qualifying = []  # list of (group, total_len, mean_inset)
    for g in groups:
        total_len = float(side_lens[g].sum())
        if total_len < 10:
            continue
        mean_inset = float(dists[g].mean())
        if mean_inset < min_inset or mean_inset > max_inset:
            continue
        qualifying.append((g, total_len, mean_inset))

    if not qualifying:
        return None, None

    # Rank by "deepest among long-enough" rather than strictly longest.
    #
    # Pure length scoring picks the wrong edge on TVs where a bezel-front lip
    # or other intermediate bezel feature happens to be slightly longer than
    # the real bezel-to-screen transition deeper inside. The bezel-to-screen
    # edge spans the full screen width or height by definition, so it's
    # already among the longer clusters; we use the length floor to discard
    # short content / reflection clusters and then pick the deepest of what
    # remains, which is the actual screen edge.
    #
    # The 0.5 × max-length floor was chosen to:
    #   * Keep the real screen edge in the running on TVs where the
    #     wall-to-bezel edge is a bit longer (1002 case).
    #   * Reject in-screen reflections, which are typically <50% the length
    #     of the real screen edge (the 1008 vertical-glare case).
    max_total_len = max(t for _, t, _ in qualifying)
    long_enough = [
        (g, total_len, mean_inset)
        for g, total_len, mean_inset in qualifying
        if total_len >= 0.5 * max_total_len
    ]
    # Pick the deepest cluster (largest mean_inset) among the long ones.
    best_tuple = max(long_enough, key=lambda x: x[2])
    best = best_tuple[0]
    return _fit_line_tls(side_segs[best], side_lens[best]), best


def _shrunken_quad(quad: np.ndarray, shrink_frac: float = 0.04) -> np.ndarray:
    """Uniformly shrink a quad toward its centroid by ``shrink_frac`` on each side.

    Used as a fallback when edge-based screen detection fails or produces an
    implausible result. For most modern thin-bezel TVs the SAM body mask
    closely traces the actual screen (with only a few pixels of bezel between),
    so a small inward shrink is a reasonable approximation of the screen
    region. Erring slightly inward is preferable to including bezel pixels:
    "warp slightly smaller than the screen" looks like a smaller picture
    deliberately placed inside the TV; "warp extends onto the bezel" looks
    like a glitch.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    return (center + (pts - center) * (1.0 - shrink_frac)).astype(np.float32)


def _perp_inset(p1: np.ndarray, p2: np.ndarray, outer_line: np.ndarray) -> float:
    """Perpendicular distance from the midpoint of (p1, p2) to ``outer_line``.

    Used to measure how far each side of the detected screen quad sits from
    the corresponding side of the SAM-derived TV outer quad — i.e. the
    effective bezel thickness on that side.
    """
    a, b, c = outer_line
    mx = 0.5 * (float(p1[0]) + float(p2[0]))
    my = 0.5 * (float(p1[1]) + float(p2[1]))
    return abs(a * mx + b * my + c)


class _BadEdgeQuad(Exception):
    """Raised inside ``fit_screen_quad`` when edge-based detection produces
    no usable quad — either no qualifying cluster was found on some side, or
    the four sides intersected to form a quad that failed a sanity check.

    Caught at the end of ``fit_screen_quad``, where the caller falls back to
    a shrunken outer quad. We use an exception rather than ``return None`` so
    that the many independent failure conditions in pass 2 can each abort the
    edge-based attempt cleanly without nested control flow.
    """


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
      3. Fit a TV outer quad from the cleaned SAM contour. We use this as a
         strong geometric prior for where the screen edges should lie.
      4. Compute a directional-edge map inside a thin annular bezel band.
      5. Canny → Hough segments on the directional channels.
      6. Bucket segments into top/bottom/left/right.
      7. Per side, cluster co-linear segments and pick the LONGEST cluster
         whose inset distance from the TV outer edge falls within a plausible
         bezel-thickness range (rejects the outer-casing transition that lives
         at near-zero inset and screen content beyond the bezel zone).
      8. TLS-fit a line through each chosen cluster.
      9. Intersect adjacent sides to get four screen corners.
     10. Sanity-check aspect ratio, side lengths, and L/R bezel symmetry.
     11. If any pass-2 step fails (no qualifying cluster, sanity check
         rejects), fall back to a shrunken SAM outer quad.
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
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [tv_pts.astype(np.int32)], 255)
    poly_dilate_k = max(7, min(h, w) // 200)
    poly_mask = cv2.dilate(poly_mask, np.ones((poly_dilate_k, poly_dilate_k), np.uint8))
    binary = cv2.bitwise_and(binary, poly_mask)

    # Pick the connected component most likely to be the TV body
    tv_cx = float(tv_pts[:, 0].mean())
    tv_cy = float(tv_pts[:, 1].mean())
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None

    best_label, best_score = None, -float("inf")
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < h * w * 0.0005:
            continue
        cx, cy = centroids[i]
        dist = np.hypot(cx - tv_cx, cy - tv_cy)
        score = area - dist * area / 100.0
        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return None

    tv_body = np.where(labels == best_label, 255, 0).astype(np.uint8)
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

    # ---- PASS 1: TV outer boundary from SAM mask contour ----
    _contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not _contours:
        return None
    _tv_contour = max(_contours, key=cv2.contourArea)
    _hull = cv2.convexHull(_tv_contour)
    tv_outer_quad = None
    for _eps in [0.02, 0.03, 0.05, 0.07, 0.10]:
        _approx = cv2.approxPolyDP(_hull, _eps * cv2.arcLength(_hull, True), True)
        if len(_approx) == 4:
            tv_outer_quad = _approx.reshape(4, 2).astype(np.float32)
            break
    if tv_outer_quad is None:
        tv_outer_quad = cv2.boxPoints(cv2.minAreaRect(_tv_contour)).astype(np.float32)

    # Order corners and build line equations for the four outer sides.
    # These become the geometric anchors for per-side screen edge fitting.
    outer_ordered = _order_quad_corners(tv_outer_quad)
    o_tl, o_tr, o_br, o_bl = outer_ordered
    outer_lines = {
        "top":    _line_through_points(o_tl, o_tr),
        "right":  _line_through_points(o_tr, o_br),
        "bottom": _line_through_points(o_br, o_bl),
        "left":   _line_through_points(o_bl, o_tl),
    }
    if any(L is None for L in outer_lines.values()):
        return None
    tv_w = float(np.linalg.norm(o_tr - o_tl))
    tv_h = float(np.linalg.norm(o_bl - o_tl))
    outer_center = ((o_tl[0] + o_br[0]) * 0.5, (o_tl[1] + o_br[1]) * 0.5)

    if debug_dir is not None:
        _vis1 = img_crop.copy()
        cv2.polylines(_vis1, [outer_ordered.astype(np.int32)], True, (0, 255, 255), 2)
        for pt in outer_ordered.astype(int):
            cv2.circle(_vis1, tuple(pt), 4, (0, 165, 255), -1)
        save_debug("12_tv_outer", _vis1)

    # ---- PASS 2: edge-based screen detection with graceful fallback ----
    #
    # Pass 2 attempts to find the bezel-to-screen transition by edge analysis
    # within the bezel band. It has two systematic failure modes worth
    # naming, both fixable by retrying with a tighter `min_inset_frac`:
    #
    #   (a) intra-screen reflections / on-screen text outscore the real
    #       screen edge on length; one side latches onto the false edge,
    #       caught by the L/R symmetry check.
    #   (b) the wall-to-bezel transition (high-contrast, light wall vs. dark
    #       bezel) outscores the bezel-to-screen transition (dark on dark)
    #       on every side, so the detected quad traces the TV's outer face
    #       instead of the screen — caught by the all-sides-at-outer check.
    #
    # We wrap pass 2 in a function so we can call it twice: once with the
    # default min_inset_frac, then once more with a larger value to push the
    # search past the wall-to-bezel edges if attempt one tripped the
    # "outer-boundary inset" sanity check. If both attempts fail we fall
    # through to a shrunken outer quad.

    def _try_pass2(min_inset_frac: float) -> np.ndarray:
        _min_dim = min(crop_h, crop_w)
        _outer_k = max(3, int(_min_dim * 0.02))
        _bezel_k = max(5, int(_min_dim * 0.18))

        _ek_outer = _outer_k * 2 + 1
        _ek_bezel = _bezel_k * 2 + 1
        _inner_from_outer = cv2.erode(mask_crop, np.ones((_ek_outer, _ek_outer), np.uint8))
        _inner_core = cv2.erode(mask_crop, np.ones((_ek_bezel, _ek_bezel), np.uint8))

        if _inner_core.any():
            bezel_band = cv2.bitwise_and(_inner_from_outer, cv2.bitwise_not(_inner_core))
        else:
            bezel_band = _inner_from_outer

        save_debug("13_bezel_band", bezel_band)

        # Directional edge detection.
        #
        # IMPORTANT: compute Sobel on the UN-MASKED gray and only restrict to
        # the bezel band AFTER Canny. Masking the gray (or the float h_map /
        # v_map) before Canny creates a sharp synthetic step at every
        # band-boundary pixel that Canny then picks up as a long, dense edge
        # ring tracing the band's perimeter.
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        _mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        h_map = (gy * gy / _mag).astype(np.float32)
        v_map = (gx * gx / _mag).astype(np.float32)

        def _to_uint8(m: np.ndarray) -> np.ndarray:
            m = np.clip(m, 0, None)
            valid = m[bezel_band > 0]
            hi = float(np.percentile(valid, 95)) if valid.size else 0.0
            hi = max(hi, 1.0)
            return np.clip(m * 255.0 / hi, 0, 255).astype(np.uint8)

        h_edge_full = cv2.Canny(_to_uint8(h_map), 50, 150)
        v_edge_full = cv2.Canny(_to_uint8(v_map), 50, 150)
        h_edge = cv2.bitwise_and(h_edge_full, bezel_band)
        v_edge = cv2.bitwise_and(v_edge_full, bezel_band)
        edges = cv2.bitwise_or(h_edge, v_edge)

        if debug_dir is not None:
            save_debug("14_h_edge", h_edge)
            save_debug("14_v_edge", v_edge)
        save_debug("14_edges", edges)

        # ---- Hough lines on directional channels ----
        min_dim = min(crop_h, crop_w)
        min_line_len = max(15, min_dim // 10)
        max_line_gap = max(5, min_dim // 40)

        _hough_kw = dict(rho=1, theta=np.pi / 180, threshold=30,
                         minLineLength=min_line_len, maxLineGap=max_line_gap)
        h_raw = cv2.HoughLinesP(h_edge, **_hough_kw)
        v_raw = cv2.HoughLinesP(v_edge, **_hough_kw)

        _pieces = []
        if h_raw is not None:
            _pieces.append(h_raw.reshape(-1, 4))
        if v_raw is not None:
            _pieces.append(v_raw.reshape(-1, 4))
        if not _pieces:
            raise _BadEdgeQuad("Hough returned no segments")

        segs = np.vstack(_pieces).astype(np.float32)
        if len(segs) < 4:
            raise _BadEdgeQuad("fewer than 4 Hough segments total")

        dx = segs[:, 2] - segs[:, 0]
        dy = segs[:, 3] - segs[:, 1]
        lens = np.hypot(dx, dy)
        angles = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0

        if debug_dir is not None:
            vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            for x1s, y1s, x2s, y2s in segs.astype(int):
                cv2.line(vis, (x1s, y1s), (x2s, y2s), (0, 255, 0), 1)
            save_debug("14_hough_raw", vis)

        # ---- Bucket into top/bottom/left/right ----
        ANGLE_HORIZ = 25.0
        ANGLE_VERT_MIN = 65.0
        h_mask = np.abs(angles) < ANGLE_HORIZ
        v_mask = np.abs(angles) > ANGLE_VERT_MIN
        h_lines, h_lens = segs[h_mask], lens[h_mask]
        v_lines, v_lens = segs[v_mask], lens[v_mask]
        if len(h_lines) < 2 or len(v_lines) < 2:
            raise _BadEdgeQuad("not enough horizontal or vertical segments")

        # Use the outer-quad center (a geometric constant) as the split rather
        # than a weighted median of segment midpoints; the median migrates
        # toward whichever side has more noise and mis-bins real edges.
        split_y = outer_center[1]
        split_x = outer_center[0]

        h_mid_y = (h_lines[:, 1] + h_lines[:, 3]) * 0.5
        v_mid_x = (v_lines[:, 0] + v_lines[:, 2]) * 0.5

        sides = {
            "top":    (h_lines[h_mid_y <  split_y], h_lens[h_mid_y <  split_y]),
            "bottom": (h_lines[h_mid_y >= split_y], h_lens[h_mid_y >= split_y]),
            "left":   (v_lines[v_mid_x <  split_x], v_lens[v_mid_x <  split_x]),
            "right":  (v_lines[v_mid_x >= split_x], v_lens[v_mid_x >= split_x]),
        }
        if any(len(s[0]) == 0 for s in sides.values()):
            raise _BadEdgeQuad("a side bucket has no candidate segments")

        # ---- Per-side: outer-anchored cluster scoring ----
        cluster_tol = max(4, min(crop_h, crop_w) // 50)
        side_to_dim = {"top": tv_h, "bottom": tv_h, "left": tv_w, "right": tv_w}

        fits = {}
        chosen_idx = {}
        for name, (segs_s, lens_s) in sides.items():
            L, idx = _fit_side_outer_anchored(
                segs_s, lens_s, outer_lines[name], outer_center,
                tv_dim=side_to_dim[name], cluster_tol=cluster_tol,
                min_inset_frac=min_inset_frac,
            )
            if L is None:
                raise _BadEdgeQuad(f"no qualifying cluster on {name}")
            fits[name] = L
            chosen_idx[name] = idx

        # ---- Intersect adjacent sides for corners ----
        def intersect(L1, L2):
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
            raise _BadEdgeQuad("adjacent sides are parallel")
        quad_crop = np.array([tl, tr, br, bl], dtype=np.float32)

        # ---- Sanity checks ----
        slack = max(crop_h, crop_w) * 0.10
        if (quad_crop[:, 0].min() < -slack or quad_crop[:, 0].max() > crop_w + slack or
            quad_crop[:, 1].min() < -slack or quad_crop[:, 1].max() > crop_h + slack):
            raise _BadEdgeQuad("quad falls outside the crop")
        if not cv2.isContourConvex(quad_crop.astype(np.int32)):
            raise _BadEdgeQuad("quad is non-convex")
        area = cv2.contourArea(quad_crop)
        if area < 0.10 * crop_h * crop_w or area > 1.05 * crop_h * crop_w:
            raise _BadEdgeQuad(f"area out of range: {area:.0f}")

        # Aspect ratio: real TVs are roughly 4:3 (1.33) to 21:9 (2.33).
        side_top    = np.linalg.norm(quad_crop[1] - quad_crop[0])
        side_bottom = np.linalg.norm(quad_crop[2] - quad_crop[3])
        side_left   = np.linalg.norm(quad_crop[3] - quad_crop[0])
        side_right  = np.linalg.norm(quad_crop[2] - quad_crop[1])
        width_avg  = 0.5 * (side_top + side_bottom)
        height_avg = 0.5 * (side_left + side_right)
        if height_avg < 1.0:
            raise _BadEdgeQuad("degenerate height")
        aspect = width_avg / height_avg
        if aspect < 1.20 or aspect > 2.60:
            raise _BadEdgeQuad(f"aspect={aspect:.2f} out of [1.20, 2.60]")

        # Opposite sides should be roughly equal length.
        if min(side_top, side_bottom) / max(side_top, side_bottom) < 0.70:
            raise _BadEdgeQuad(
                f"top/bottom length mismatch ({side_top:.0f} vs {side_bottom:.0f})"
            )
        if min(side_left, side_right) / max(side_left, side_right) < 0.70:
            raise _BadEdgeQuad(
                f"left/right length mismatch ({side_left:.0f} vs {side_right:.0f})"
            )

        # Bezel-inset symmetry. Real screens have nearly equal left/right
        # bezel widths, so a large L/R asymmetry is the smoking gun for one
        # side latching onto a reflection or content edge. The failure that
        # motivated this check: a window glare on a thin-bezel TV produced a
        # vertical edge cluster ~20% inset from the SAM left edge. That
        # cluster was long enough and inside the legal inset range, so it
        # beat the real screen-left edge (which sat below min_inset_frac).
        # The right side, with no reflection, picked the actual screen edge
        # at ~2% inset. The resulting L/R ratio was ~10×.
        inset_top    = _perp_inset(quad_crop[0], quad_crop[1], outer_lines["top"])
        inset_right  = _perp_inset(quad_crop[1], quad_crop[2], outer_lines["right"])
        inset_bottom = _perp_inset(quad_crop[2], quad_crop[3], outer_lines["bottom"])
        inset_left   = _perp_inset(quad_crop[3], quad_crop[0], outer_lines["left"])

        lr_ratio = max(inset_left, inset_right) / max(min(inset_left, inset_right), 1.0)
        if lr_ratio > 4.0:
            raise _BadEdgeQuad(
                f"L/R bezel inset asymmetric (left={inset_left:.1f}, "
                f"right={inset_right:.1f}, ratio={lr_ratio:.1f})"
            )
        # Top/bottom asymmetry is tolerated more generously (up to 6×) because
        # older TVs with bottom branding strips legitimately have a much
        # thicker bottom bezel than top.
        tb_ratio = max(inset_top, inset_bottom) / max(min(inset_top, inset_bottom), 1.0)
        if tb_ratio > 6.0:
            raise _BadEdgeQuad(
                f"T/B bezel inset asymmetric (top={inset_top:.1f}, "
                f"bottom={inset_bottom:.1f}, ratio={tb_ratio:.1f})"
            )

        # "All four sides at very low inset" — when every chosen edge sits
        # right at the TV's outer boundary, we've traced the wall-to-bezel
        # transition on every side instead of the bezel-to-screen transition.
        # This happens on thin-bezel TVs without distinguishing features: the
        # wall-to-bezel edge has high contrast (light wall, dark bezel) so it
        # produces longer, denser Hough segments than the dark-on-dark
        # bezel-to-screen edge and wins the length contest. The L/R and T/B
        # symmetry checks miss it because all four sides fail the *same way*.
        #
        # We catch it by checking the LARGEST of the four insets — if even
        # that is small, the detected quad is essentially the outer quad.
        # The shrunken-outer fallback is a better approximation in this case:
        # 5% inset is closer to a typical thin bezel than 1–3% is. Thick-
        # bezel TVs (e.g. AQUOS-style with a tall bottom branding strip) have
        # at least one large inset, so they pass this check and keep their
        # edge-based result.
        tv_min_dim = min(tv_w, tv_h)
        max_inset = max(inset_top, inset_right, inset_bottom, inset_left)
        if max_inset < 0.055 * tv_min_dim:
            raise _BadEdgeQuad(
                f"all four sides at outer-boundary inset (max={max_inset:.1f}, "
                f"threshold={0.055 * tv_min_dim:.1f}); traced wall-to-bezel "
                f"transition instead of screen edge"
            )

        # ---- Debug viz for the edge-based success path ----
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

        return quad_crop

    # Try the edge-based pipeline with increasing min_inset_frac. The first
    # value covers the common case (modern thin-bezel TVs whose screen edge
    # sits just past the SAM mask boundary). The second is a retry for the
    # specific "wall-to-bezel transition outscored the real screen edge on
    # every side" failure mode, where pushing the inset floor past the wall
    # transition lets the deeper bezel-to-screen edge win on its own merits.
    quad_crop = None
    last_exc: _BadEdgeQuad | None = None
    for attempt_min_inset in (0.035, 0.07):
        try:
            quad_crop = _try_pass2(attempt_min_inset)
            break  # success
        except _BadEdgeQuad as exc:
            last_exc = exc
            # Only the "outer-boundary inset" failure benefits from a deeper
            # search. Other failures (no Hough segments, parallel sides, L/R
            # asymmetry, etc.) won't be cured by retrying — go straight to
            # the fallback.
            if "outer-boundary inset" not in str(exc):
                break

    if quad_crop is None:
        # Fallback: SAM outer quad shrunken inward by ~5.5% per side. For
        # modern thin-bezel TVs this approximates the screen tightly. For
        # thick-bezel TVs the edge-based path normally succeeds and we don't
        # land here; if we do, the warp will spill onto a few pixels of
        # bezel but won't catastrophically miss.
        if debug_dir is not None:
            print(f"  [debug] edge-based screen detection failed ({last_exc}); "
                  f"falling back to shrunken outer quad")
        quad_crop = _shrunken_quad(outer_ordered, shrink_frac=0.055)
        if debug_dir is not None:
            vis = img_crop.copy()
            cv2.polylines(vis, [quad_crop.astype(np.int32)], True, (0, 165, 255), 3)
            for p in quad_crop.astype(int):
                cv2.circle(vis, tuple(p), 4, (0, 165, 255), -1)
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
        return True, logs

    logs.append(
        f"  Screen corners (TL/TR/BR/BL):\n"
        f"    {corners[0].astype(int).tolist()}\n"
        f"    {corners[1].astype(int).tolist()}\n"
        f"    {corners[2].astype(int).tolist()}\n"
        f"    {corners[3].astype(int).tolist()}"
    )

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
