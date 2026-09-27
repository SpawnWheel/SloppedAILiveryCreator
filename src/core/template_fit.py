"""
Keeping AI output locked to the UV template.

Image models redraw the whole picture, so their output drifts: a slightly different zoom / margin,
a shift of a few percent, paint spilling outside the UV islands. This module:

  1. prepare_for_ai()   - flattens transparent templates onto white (a black-lines-on-transparent PNG used to
                          reach the model as a solid black square) and pads non-square templates to an aspect
                          ratio the model supports, so nothing gets stretched.
  2. restore_geometry() - undoes that padding on the model's output.
  3. align()            - finds the shift / scale / skew between the output and the template (OpenCV ECC on
                          the panel silhouettes) and warps the output back onto the template.
  4. clip_to_islands()  - keeps paint only inside the template's UV islands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

# Aspect ratios accepted by Gemini image models
SUPPORTED_RATIOS = {
    "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3,
    "5:4": 5 / 4, "4:5": 4 / 5, "16:9": 16 / 9, "9:16": 9 / 16, "21:9": 21 / 9,
}


@dataclass
class PreparedTemplate:
    image: Image.Image                 # RGB, padded, ready to send
    aspect_ratio: str                  # e.g. "1:1"
    content_box: Tuple[float, float, float, float]  # original area inside padded image, as fractions (l, t, r, b)
    original_size: Tuple[int, int]


@dataclass
class FitReport:
    aligned: bool = False
    clipped: bool = False
    match_before: float = 0.0
    match_after: float = 0.0
    shift_px: Tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.aligned:
            parts.append(
                f"auto-aligned to template (moved {self.shift_px[0]:+.0f},{self.shift_px[1]:+.0f}px, "
                f"scale {self.scale:.3f})"
            )
        if self.match_after:
            parts.append(f"panel match {self.match_after * 100:.0f}%")
        if self.clipped:
            parts.append("paint clipped to UV islands")
        parts.extend(self.notes)
        return "; ".join(parts)


# ----------------------------------------------------------------------------------------------
# 1 + 2: sending the template in a shape the model can't misread
# ----------------------------------------------------------------------------------------------

def flatten_template(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Composites RGBA onto a solid background. Plain .convert('RGB') would just drop alpha,
    turning black-lines-on-transparent templates into an all-black image."""
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, background + (255,))
    bg.alpha_composite(rgba)
    return bg.convert("RGB")


def closest_ratio(w: int, h: int) -> str:
    r = w / h
    return min(SUPPORTED_RATIOS, key=lambda k: abs(np.log(SUPPORTED_RATIOS[k] / r)))


def prepare_for_ai(template: Image.Image, max_dim: int = 2048) -> PreparedTemplate:
    flat = flatten_template(template)
    w, h = flat.size
    ratio_name = closest_ratio(w, h)
    target_r = SUPPORTED_RATIOS[ratio_name]
    if abs(w / h - target_r) < 1e-3:
        pw, ph = w, h
    elif w / h > target_r:
        pw, ph = w, int(round(w / target_r))
    else:
        pw, ph = int(round(h * target_r)), h
    bg = _border_color(np.asarray(flat))
    padded = Image.new("RGB", (pw, ph), tuple(int(c) for c in bg))
    ox, oy = (pw - w) // 2, (ph - h) // 2
    padded.paste(flat, (ox, oy))
    if max(padded.size) > max_dim:
        padded.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    box = (ox / pw, oy / ph, (ox + w) / pw, (oy + h) / ph)
    return PreparedTemplate(padded, ratio_name, box, (w, h))


def restore_geometry(output: Image.Image, prep: PreparedTemplate, out_size: Tuple[int, int]) -> Image.Image:
    """Crops the padding back off the model output and scales it to `out_size`."""
    ow, oh = output.size
    l, t, r, b = prep.content_box
    if (l, t, r, b) != (0.0, 0.0, 1.0, 1.0):
        output = output.crop((round(l * ow), round(t * oh), round(r * ow), round(b * oh)))
    if output.size != out_size:
        output = output.resize(out_size, Image.Resampling.LANCZOS)
    return output


# ----------------------------------------------------------------------------------------------
# UV island masks
# ----------------------------------------------------------------------------------------------

def _border_color(arr: np.ndarray) -> np.ndarray:
    border = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]])
    return np.median(border, axis=0)


def island_mask(img: Image.Image, size: int = 1024, tolerance: int = 28) -> Optional[np.ndarray]:
    """
    Boolean mask (size x size-ish, keeping aspect) of everything that is NOT the background connected to the
    image border - i.e. the UV islands. Returns None when no sensible mask can be found.
    """
    flat = flatten_template(img)
    small = flat.copy()
    small.thumbnail((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(small).astype(np.int16)
    bg = _border_color(arr)
    similar = (np.abs(arr - bg).max(axis=2) <= tolerance).astype(np.uint8)

    if cv2 is not None:
        n, labels = cv2.connectedComponents(similar, connectivity=4)
        edge_labels = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
        edge_labels.discard(0)
        background = np.isin(labels, list(edge_labels)) & (similar > 0)
        mask = ~background
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    else:
        mask = ~(similar > 0)

    coverage = mask.mean()
    if coverage < 0.05 or coverage > 0.97:
        return None
    return mask


# ----------------------------------------------------------------------------------------------
# 3: alignment
# ----------------------------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _structure_map(img: Image.Image, size: int) -> np.ndarray:
    """Soft edge map used to align images that share panel outlines (e.g. two livery versions)."""
    g = flatten_template(img).convert("L")
    g.thumbnail((size, size), Image.Resampling.BILINEAR)
    g = g.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(2))
    a = np.asarray(g).astype(np.float32)
    return a / (a.max() or 1.0)


def estimate_affine(reference: Image.Image, moving: Image.Image, size: int = 768,
                    use_islands: bool = True) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Returns (warp_2x3 in `size`-space mapping reference coords -> moving coords, score_before, score_after).
    Score = IoU of island masks (use_islands) or normalised correlation of edge maps.
    """
    if cv2 is None:
        return None, 0.0, 0.0

    if use_islands:
        rm = island_mask(reference, size)
        mm = island_mask(moving, size)
        if rm is None or mm is None or rm.shape != mm.shape:
            return None, 0.0, 0.0
        ref = cv2.GaussianBlur(rm.astype(np.float32), (0, 0), 3)
        mov = cv2.GaussianBlur(mm.astype(np.float32), (0, 0), 3)
        before = _iou(rm, mm)
    else:
        ref = _structure_map(reference, size)
        mov = _structure_map(moving, size)
        if ref.shape != mov.shape:
            return None, 0.0, 0.0
        before = float(np.corrcoef(ref.ravel(), mov.ravel())[0, 1])

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
        # coarse (translation) then fine (affine)
        _, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
        _, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_AFFINE, criteria, None, 5)
    except cv2.error:
        return None, before, before

    h, w = ref.shape
    warped = cv2.warpAffine(mov, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    if use_islands:
        after = _iou(ref > 0.5, warped > 0.5)
    else:
        after = float(np.corrcoef(ref.ravel(), warped.ravel())[0, 1])
    return warp, before, after


def _sane(warp: np.ndarray, w: int, h: int) -> bool:
    a, b, tx = warp[0]
    c, d, ty = warp[1]
    sx, sy = np.hypot(a, c), np.hypot(b, d)
    skew = abs(a * b + c * d) / (sx * sy + 1e-9)
    return (0.8 < sx < 1.25 and 0.8 < sy < 1.25 and skew < 0.08
            and abs(tx) < 0.15 * w and abs(ty) < 0.15 * h)


def align(output: Image.Image, reference: Image.Image, use_islands: bool = True,
          report: Optional[FitReport] = None) -> Image.Image:
    """Warps `output` so its panels sit on top of `reference`'s panels. Output keeps its size."""
    report = report or FitReport()
    size = 768
    warp, before, after = estimate_affine(reference, output, size, use_islands)
    report.match_before = before
    report.match_after = before
    if warp is None:
        report.notes.append("alignment skipped (couldn't detect the template's panels)")
        return output

    # express the warp in the small image's pixel units, then scale to full size
    ref_small = flatten_template(reference)
    ref_small.thumbnail((size, size))
    sw, sh = ref_small.size
    if not _sane(warp, sw, sh) or after <= before + 0.003:
        return output

    W, H = output.size
    full = warp.copy()
    full[0, 2] *= W / sw
    full[1, 2] *= H / sh
    arr = np.asarray(output.convert("RGBA"))
    warped = cv2.warpAffine(arr, full, (W, H), flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
    report.aligned = True
    report.match_after = after
    report.shift_px = (float(-full[0, 2]), float(-full[1, 2]))
    report.scale = float((np.hypot(full[0, 0], full[1, 0]) + np.hypot(full[0, 1], full[1, 1])) / 2)
    return Image.fromarray(warped, "RGBA")


# ----------------------------------------------------------------------------------------------
# 4: clipping
# ----------------------------------------------------------------------------------------------

def clip_to_islands(output: Image.Image, template: Image.Image, background=(18, 20, 24),
                    grow_px: int = 6, report: Optional[FitReport] = None) -> Image.Image:
    mask = island_mask(template, 2048)
    if mask is None:
        if report:
            report.notes.append("clip skipped (template panels not closed)")
        return output
    m = Image.fromarray((mask * 255).astype(np.uint8), "L").resize(output.size, Image.Resampling.BILINEAR)
    if grow_px > 0:
        m = m.filter(ImageFilter.MaxFilter(grow_px * 2 + 1 if grow_px * 2 + 1 <= 15 else 15))
    base = Image.new("RGBA", output.size, background + (255,))
    result = Image.composite(output.convert("RGBA"), base, m)
    if report:
        report.clipped = True
    return result


# ----------------------------------------------------------------------------------------------
# One-call helper used by the app
# ----------------------------------------------------------------------------------------------

def lock_to_template(output: Image.Image, template: Optional[Image.Image],
                     previous: Optional[Image.Image] = None,
                     do_align: bool = True, do_clip: bool = True) -> Tuple[Image.Image, FitReport]:
    """
    output   - AI result (already cropped/resized to the template's geometry)
    template - the UV wireframe (may be None)
    previous - for refinements: the livery being edited (aligned against when there is no template)
    """
    report = FitReport()
    result = output
    if do_align:
        if template is not None:
            result = align(result, template, use_islands=True, report=report)
        elif previous is not None:
            result = align(result, previous, use_islands=False, report=report)
    if do_clip and template is not None:
        result = clip_to_islands(result, template, report=report)
    return result, report
