"""
Flood-fill ("paint bucket") mask computation for the paint editor.

Uses OpenCV when available (fast even on 4096 x 4096 textures) and falls back to a
numpy scanline fill otherwise.
"""

from typing import Optional, Tuple

import numpy as np

try:  # optional, but strongly recommended for speed
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on install
    cv2 = None


def flood_mask(
    rgba: np.ndarray,
    seed: Tuple[int, int],
    tolerance: int = 32,
    grow: int = 0,
    contiguous: bool = True,
) -> Optional[np.ndarray]:
    """
    Returns a bool mask (H, W) of the region connected to `seed` whose colour is within
    `tolerance` (0-255 per channel) of the seed colour. `grow` dilates the mask by N px so a fill
    slides under anti-aliased template lines instead of leaving a halo.
    """
    h, w = rgba.shape[:2]
    x, y = seed
    if not (0 <= x < w and 0 <= y < h):
        return None

    # Compare on straight RGB + alpha; transparent pixels compare as their alpha only
    img = np.ascontiguousarray(rgba[:, :, :4])
    ref = img[y, x].astype(np.int16)

    if not contiguous:
        diff = np.abs(img.astype(np.int16) - ref).max(axis=2)
        mask = diff <= tolerance
    elif cv2 is not None:
        src = np.ascontiguousarray(img[:, :, :3])
        # alpha as a 4th discriminator: encode into a separate channel check afterwards
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        tol = (tolerance, tolerance, tolerance)
        cv2.floodFill(src.copy(), ff_mask, (int(x), int(y)), (0, 0, 0), tol, tol, flags)
        mask = ff_mask[1:-1, 1:-1] > 0
        # respect alpha too (so transparent areas of a layer are their own region)
        region_alpha = img[:, :, 3][mask].astype(np.int16)
        if region_alpha.size and np.abs(region_alpha - ref[3]).max() > tolerance:
            alpha_ok = np.abs(img[:, :, 3].astype(np.int16) - ref[3]) <= tolerance
            mask &= alpha_ok
            # keep only component containing seed
            n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
            if n > 1:
                mask = labels == labels[y, x]
    else:
        mask = _scanline_fill(img, x, y, tolerance)

    if grow > 0 and mask is not None:
        bb = mask_bbox(mask)
        if bb is not None:
            bx, by, bw, bh = bb
            x0, y0 = max(0, bx - grow), max(0, by - grow)
            x1, y1 = min(w, bx + bw + grow), min(h, by + bh + grow)
            mask[y0:y1, x0:x1] = _dilate(mask[y0:y1, x0:x1], grow)
    return mask


def _scanline_fill(img: np.ndarray, x: int, y: int, tol: int) -> np.ndarray:
    h, w = img.shape[:2]
    ref = img[y, x].astype(np.int16)
    similar = np.abs(img.astype(np.int16) - ref).max(axis=2) <= tol
    mask = np.zeros((h, w), bool)
    stack = [(x, y)]
    while stack:
        sx, sy = stack.pop()
        if mask[sy, sx] or not similar[sy, sx]:
            continue
        row = similar[sy]
        lx = sx
        while lx > 0 and row[lx - 1] and not mask[sy, lx - 1]:
            lx -= 1
        rx = sx
        while rx < w - 1 and row[rx + 1] and not mask[sy, rx + 1]:
            rx += 1
        mask[sy, lx: rx + 1] = True
        for ny in (sy - 1, sy + 1):
            if 0 <= ny < h:
                seg = similar[ny, lx: rx + 1] & ~mask[ny, lx: rx + 1]
                if seg.any():
                    # push one seed per run
                    idx = np.flatnonzero(seg)
                    starts = idx[np.insert(np.diff(idx) > 1, 0, True)]
                    for s in starts:
                        stack.append((lx + int(s), ny))
    return mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(x, y, w, h) of True pixels, or None."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    return int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)
