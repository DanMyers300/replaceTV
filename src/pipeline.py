from pathlib import Path

import cv2
import numpy as np

from .detection import detect_tv, detect_tv_body_mask, upload_to_fal
from .vision import fit_screen_quad, warp_and_composite


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
