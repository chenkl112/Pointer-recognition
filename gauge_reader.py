from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _imread_unicode(path: str) -> Optional[np.ndarray]:
    """Read image robustly for Unicode paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _auto_crop_black_frame(image: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Auto-crop the dark rectangular gauge frame when present."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, bin_dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_dark = cv2.morphologyEx(bin_dark, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bin_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None

    best_rect = None
    best_score = -1.0
    img_area = float(h * w)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = float(cw * ch)
        if area < img_area * 0.08:
            continue

        aspect = cw / max(ch, 1)
        if aspect < 1.2 or aspect > 4.5:
            continue

        score = area
        if score > best_score:
            best_score = score
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        return image, None

    x, y, cw, ch = best_rect
    pad = 2
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)
    cropped = image[y1:y2, x1:x2].copy()
    return cropped, (x1, y1, x2 - x1, y2 - y1)


def _estimate_center(shape: Tuple[int, int], edge_image: np.ndarray) -> Tuple[int, int]:
    """Estimate center from lower horizontal baseline, fallback to bottom midpoint."""
    h, w = shape
    fallback = (w // 2, h)

    lines = cv2.HoughLinesP(
        edge_image,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(30, int(w * 0.35)),
        maxLineGap=8,
    )
    if lines is None:
        return fallback

    best = None
    best_score = -1.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx < max(30, int(w * 0.25)):
            continue
        if dy > 3:
            continue

        y_avg = 0.5 * (y1 + y2)
        if y_avg < h * 0.6:
            continue

        score = dx + y_avg * 0.25
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)

    if best is None:
        return fallback

    x1, y1, x2, y2 = best
    cy = int(round((y1 + y2) * 0.5))
    cy = int(np.clip(cy, int(h * 0.7), h))
    return (w // 2, cy)


def _analyze_gauge(image: np.ndarray) -> Tuple[Optional[int], dict]:
    """Run full analysis and return value plus debug intermediates."""
    debug = {
        "work_image": None,
        "crop_rect": None,
        "center": None,
        "mask": None,
        "edges": None,
        "edges_roi": None,
        "lines": None,
        "tip": None,
        "value": None,
    }

    if image is None or image.size == 0:
        return None, debug

    if image.ndim != 3 or image.shape[2] != 3:
        return None, debug

    work_image, crop_rect = _auto_crop_black_frame(image)
    debug["work_image"] = work_image
    debug["crop_rect"] = crop_rect

    h, w = work_image.shape[:2]
    if h < 20 or w < 20:
        return None, debug

    gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    debug["edges"] = edges

    center = _estimate_center((h, w), edges)
    debug["center"] = center

    mask = _build_upper_semi_mask((h, w), center)
    edges_roi = cv2.bitwise_and(edges, mask)
    debug["mask"] = mask
    debug["edges_roi"] = edges_roi

    lines = cv2.HoughLinesP(
        edges_roi,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=max(20, int(min(h, w) * 0.25)),
        maxLineGap=8,
    )
    debug["lines"] = lines

    if lines is None:
        return None, debug

    tip = _line_pointer_tip(
        lines=lines,
        center=center,
        min_len=max(20, float(min(h, w) * 0.25)),
        center_tol=max(10.0, float(min(h, w) * 0.16)),
    )
    debug["tip"] = tip

    if tip is None:
        return None, debug

    value = _tip_to_value(tip, center)
    debug["value"] = value
    return value, debug


def _build_upper_semi_mask(shape: Tuple[int, int], center: Tuple[int, int]) -> np.ndarray:
    """Build an upper-semicircle mask to suppress lower-edge interference."""
    h, w = shape
    cx, cy = center

    radius = max(1, min(w // 2 - 1, max(1, cy - 1)))

    yy, xx = np.ogrid[:h, :w]
    in_circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2

    upper_region = yy <= cy
    mask = (in_circle & upper_region).astype(np.uint8) * 255
    return mask


def _line_pointer_tip(
    lines: np.ndarray,
    center: Tuple[int, int],
    min_len: float,
    center_tol: float,
) -> Optional[Tuple[int, int]]:
    """Select the most likely pointer line and return its tip coordinate."""
    cx, cy = center
    best_len = -1.0
    best_tip = None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = np.array([x1, y1], dtype=np.float32)
        p2 = np.array([x2, y2], dtype=np.float32)

        seg_len = float(np.linalg.norm(p1 - p2))
        if seg_len < min_len:
            continue

        d1 = float(np.linalg.norm(p1 - np.array([cx, cy], dtype=np.float32)))
        d2 = float(np.linalg.norm(p2 - np.array([cx, cy], dtype=np.float32)))

        if min(d1, d2) > center_tol:
            continue

        tip = (x2, y2) if d2 > d1 else (x1, y1)

        if seg_len > best_len:
            best_len = seg_len
            best_tip = tip

    return best_tip


def _tip_to_value(tip: Tuple[int, int], center: Tuple[int, int]) -> int:
    """Map pointer angle to gauge value in range 0-50."""
    cx, cy = center
    tx, ty = tip

    vx = tx - cx
    vy = cy - ty

    angle = np.degrees(np.arctan2(vy, vx))
    angle = float(np.clip(angle, 0.0, 180.0))

    value = (180.0 - angle) / 180.0 * 50.0
    return int(np.clip(round(value), 0, 50))


def read_gauge(image: np.ndarray) -> Optional[int]:
    """Read a single semicircular pointer gauge."""
    value, _ = _analyze_gauge(image)
    return value


def read_gauge_debug(image: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """Read gauge and return a debug visualization image."""
    if image is None or image.size == 0:
        return None, None

    value, debug = _analyze_gauge(image)
    base = debug.get("work_image")
    vis = base.copy() if base is not None else image.copy()
    h, w = vis.shape[:2]

    center = debug.get("center")
    if center is None:
        center = (w // 2, h)

    mask = debug.get("mask")
    if mask is not None:
        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = mask
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.15, 0)

    cv2.circle(vis, center, 5, (0, 255, 255), -1)

    lines = debug.get("lines")
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(vis, (x1, y1), (x2, y2), (255, 120, 0), 1, cv2.LINE_AA)

    tip = debug.get("tip")
    if tip is not None:
        cv2.line(vis, center, tip, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, tip, 5, (0, 0, 255), -1)

    text = f"value={value if value is not None else 'None'}"
    cv2.putText(vis, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2, cv2.LINE_AA)

    return value, vis


def read_gauge_stable(image_list: Sequence[np.ndarray]) -> Optional[int]:
    """Read multiple frames and return mode value for stability."""
    if not image_list:
        return None

    values: List[int] = []
    for img in image_list:
        v = read_gauge(img)
        if v is not None:
            values.append(v)

    if not values:
        return None

    counter = Counter(values)
    max_freq = max(counter.values())
    modes = [v for v, c in counter.items() if c == max_freq]

    return int(min(modes))


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    debug_mode = False
    output_path: Optional[str] = None

    if "--debug" in args:
        debug_mode = True
        args.remove("--debug")

    if "--out" in args:
        idx = args.index("--out")
        if idx + 1 >= len(args):
            print("[ERROR] --out requires a file path")
            raise SystemExit(1)
        output_path = args[idx + 1]
        del args[idx : idx + 2]

    if len(args) < 1:
        print("Usage: python gauge_reader.py [--debug] [--out result.txt] <image_path1> [image_path2 ...]")
        raise SystemExit(1)

    images = []
    valid_paths = []
    for p in args:
        img = _imread_unicode(p)
        if img is None:
            print(f"[WARN] Cannot read image: {p}")
            continue
        images.append(img)
        valid_paths.append(p)

    if not images:
        print("No valid images loaded.")
        raise SystemExit(1)

    if len(images) == 1 and debug_mode:
        value, vis = read_gauge_debug(images[0])
        print(value)

        if vis is not None:
            out_path = "debug_result.png"
            ok = cv2.imwrite(out_path, vis)
            if ok:
                print(f"[INFO] Debug image saved to: {out_path}")
            else:
                print("[WARN] Failed to save debug image.")
        else:
            print("[WARN] No debug visualization generated.")

        if output_path is not None:
            Path(output_path).write_text(f"{value}\n", encoding="utf-8")
            print(f"[INFO] Result written to: {output_path}")
    elif len(images) == 1:
        value = read_gauge(images[0])
        print(value)
        if output_path is not None:
            Path(output_path).write_text(f"{value}\n", encoding="utf-8")
            print(f"[INFO] Result written to: {output_path}")
    else:
        stable_value = read_gauge_stable(images)
        print(stable_value)

        if output_path is not None:
            Path(output_path).write_text(f"{stable_value}\n", encoding="utf-8")
            print(f"[INFO] Result written to: {output_path}")

        if debug_mode:
            for idx, (img, p) in enumerate(zip(images, valid_paths), start=1):
                v, vis = read_gauge_debug(img)
                out_path = f"debug_result_{idx}.png"
                if vis is not None and cv2.imwrite(out_path, vis):
                    print(f"[INFO] Frame {idx}: value={v}, path={p}, debug={out_path}")
                else:
                    print(f"[INFO] Frame {idx}: value={v}, path={p}, debug=<save_failed>")
