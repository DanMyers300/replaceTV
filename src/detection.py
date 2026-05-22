import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def upload_to_fal(image_path: Path) -> str:
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
        return img
    if img.shape[2] == 4:
        # RGBA: SAM often encodes the mask in the alpha channel
        alpha = img[:, :, 3]
        if alpha.min() < alpha.max():
            return alpha
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
            "apply_mask": False,
            "return_multiple_masks": True,
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
