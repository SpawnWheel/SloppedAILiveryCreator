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

LOW_MATCH_WARNING = 0.6  # panel IoU below this after alignment = the model drew its own layout

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
    layout_text: str = ""              # measured panel list for the prompt ("" when panels couldn't be found)
    panel_count: int = 0


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


def prepare_for_ai(template: Image.Image, max_dim: int = 2048, as_guide: bool = True) -> PreparedTemplate:
    """
    Builds Image 1 for the model. With `as_guide` (default) and detectable panels, the template is redrawn as a
    high-contrast LAYOUT MAP: dark background, light-grey panels with thick black outlines, faint mesh, black
    number markers. Many AMS2 templates are white lines on light grey - after the model downsizes them the lines
    vanish and it sees a blank sheet, so it invents its own layout.
    """
    analysis = analyse_template(template) if as_guide else None
    flat = render_ai_guide(template, analysis) if analysis is not None else flatten_template(template)
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
    layout = describe_layout(analysis, box) if analysis is not None else ""
    return PreparedTemplate(padded, ratio_name, box, (w, h), layout_text=layout,
                            panel_count=len(analysis.panels) if analysis is not None else 0)


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
        # Low-contrast line templates (e.g. white mesh on light grey): detect panels from the lines instead.
        analysis = analyse_template(img)
        if analysis is None:
            return None
        return np.asarray(Image.fromarray(analysis.mask.astype(np.uint8) * 255)
                          .resize((mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST)) > 127
    return mask


# ----------------------------------------------------------------------------------------------
# Template analysis: panels + number markers, measured at full resolution
# ----------------------------------------------------------------------------------------------

GUIDE_BG = (18, 20, 24)        # same dark background the prompt asks the model to keep
GUIDE_PANEL = (222, 222, 222)
GUIDE_MESH = (178, 178, 178)
GUIDE_OUTLINE = (0, 0, 0)
_WORK_DIM = 4096               # thin gaps between panels only survive at (near) full resolution
_analysis_cache: dict = {}


@dataclass
class Marker:
    box: Tuple[float, float, float, float]   # (l, t, r, b) fractions of the template
    rotation: Optional[int] = None           # clockwise degrees of the placeholder text on the sheet (0/90/180/270)
    text: str = ""
    panel: int = -1                          # index into TemplateAnalysis.panels, -1 = none


@dataclass
class TemplateAnalysis:
    size: Tuple[int, int]            # work size (w, h) the arrays below refer to
    mask: np.ndarray                 # bool (h, w) - UV panels
    lines: np.ndarray                # bool (h, w) - wireframe line pixels
    markers_px: np.ndarray           # bool (h, w) - dark marker pixels (placeholder numbers/text)
    panels: list                     # [(l, t, r, b)] fractions of the work image, big panels only
    markers: list                    # [Marker]
    labels: Optional[np.ndarray] = None   # int32 (h, w) - component id per panel pixel (0 = background)
    panel_ids: list = field(default_factory=list)  # label id for each entry of `panels`

    def panel_mask(self, index: int) -> np.ndarray:
        return self.labels == self.panel_ids[index]

    def rotated_panels(self) -> list:
        """[(panel_index, rotation_cw, marker)] for panels whose placeholder text is known to be rotated."""
        out, seen = [], set()
        for m in self.markers:
            if m.panel >= 0 and m.rotation and m.panel not in seen:
                seen.add(m.panel)
                out.append((m.panel, m.rotation, m))
        return out


_marker_registry: dict = {}


def _cache_key(img: Image.Image) -> tuple:
    thumb = img.convert("RGBA").resize((64, 64), Image.Resampling.BILINEAR)
    return (img.size, hash(thumb.tobytes()))


def register_markers(img: Image.Image, markers: list) -> None:
    """
    Records placeholder text layers read from a layered template (PSD): [(text, (l, t, r, b) px, rotation_cw)].
    Their exact rotation tells the app which way is up for the panel they sit on.
    """
    w, h = img.size
    ms = []
    for text, (l, t, r, b), rot in markers:
        ms.append(Marker((l / w, t / h, r / w, b / h), int(round(rot / 90.0)) * 90 % 360, str(text).strip()))
    key = _cache_key(img)
    _marker_registry[key] = ms
    _analysis_cache.pop(key, None)


def _split_panels(unsealed_mask: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Panel identities come from the unsealed mask (touching panels such as nose + bonnet stay separate), the
    silhouette from the sealed one; sealed-only pixels join their nearest unsealed panel.
    """
    n, comp, stats, _ = cv2.connectedComponentsWithStats(unsealed_mask.astype(np.uint8), connectivity=8)
    # fragments too small to be a real panel are absorbed by their neighbour instead of drawing fake seams
    tiny = np.nonzero(stats[:, cv2.CC_STAT_AREA] < 0.002 * mask.size)[0]
    tiny = tiny[tiny > 0]
    if len(tiny):
        comp = np.where(np.isin(comp, tiny), 0, comp)
    labels = np.where(mask, comp, 0).astype(np.int32)
    missing = mask & (comp == 0)
    if missing.any() and n > 1:
        _, idx = cv2.distanceTransformWithLabels((comp == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(comp > 0)
        lut = np.zeros(idx.max() + 1, np.int32)
        lut[idx[ys, xs]] = comp[ys, xs]
        labels[missing] = lut[idx[missing]]
    return labels


def analyse_template(img: Image.Image) -> Optional[TemplateAnalysis]:
    """
    Finds the UV panels of a line-art template at full resolution: every non-line region connected to the image
    border is background; everything else (panel interiors + their lines) is a panel. Tries decreasing contrast
    thresholds so faint white-on-grey meshes work too. Returns None if no sensible panel layout is found.
    """
    if cv2 is None:
        return None
    key = _cache_key(img)
    if key in _analysis_cache:
        return _analysis_cache[key]

    flat = flatten_template(img)
    if max(flat.size) > _WORK_DIM:
        flat.thumbnail((_WORK_DIM, _WORK_DIM), Image.Resampling.LANCZOS)
    arr = np.asarray(flat).astype(np.int16)
    h, w = arr.shape[:2]
    bg = _border_color(arr)
    dev = np.abs(arr - bg).max(axis=2)
    lum = arr.mean(axis=2)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(7 * w / 4096)) | 1),) * 2)

    def panels_from(line_px: np.ndarray) -> np.ndarray:
        free = (~line_px).astype(np.uint8)
        _, lab = cv2.connectedComponents(free, connectivity=4)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        m = ~(np.isin(lab, edge) & (free > 0))
        return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k_open) > 0

    result = None
    for tol in (28, 12, 6):
        lines = dev > tol
        if lines.mean() < 0.003:
            continue
        # seal 1-2 px breaks in the mesh so panel cells don't leak into the background
        seal = max(3, int(round(3 * w / 4096)) | 1)
        sealed = cv2.dilate(lines.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (seal, seal))) > 0
        mask = panels_from(sealed)
        if 0.05 <= mask.mean() <= 0.97:
            result = (mask, lines & mask, panels_from(lines) & mask)
            break
    if result is None:
        _analysis_cache[key] = None
        return None
    mask, lines, unsealed = result
    labels = _split_panels(unsealed, mask)

    # Placeholder numbers/text: pixels clearly darker than the wireframe lines (only if the lines are light).
    line_lum = np.median(lum[lines]) if lines.any() else 0
    markers_px = (lum < 110) & mask if line_lum > 160 else np.zeros_like(mask)

    # Panels = label components above a minimum size
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    big = [(i, c) for i, c in zip(ids, counts) if c >= 0.002 * w * h]
    big.sort(key=lambda t: -t[1])
    big = big[:24]
    entries = []
    for pid, _ in big:
        ys, xs = np.nonzero(labels == pid)
        entries.append(((xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h), int(pid)))
    entries.sort(key=lambda e: (round(e[0][1], 2), e[0][0]))
    panels = [e[0] for e in entries]
    panel_ids = [e[1] for e in entries]

    # Markers: exact ones from the PSD text layers if known, otherwise dark-pixel clusters (rotation unknown)
    markers = [Marker(m.box, m.rotation, m.text) for m in _marker_registry.get(key, [])]
    if not markers and markers_px.any():
        mm = cv2.dilate(markers_px.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 60),) * 2))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
        found = [(x / w, y / h, (x + bw) / w, (y + bh) / h, a) for x, y, bw, bh, a in stats[1:]
                 if a >= 0.0004 * w * h]
        found.sort(key=lambda b: -b[4])
        markers = [Marker(b[:4]) for b in found[:12]]
    for m in markers:
        cx = min(w - 1, int((m.box[0] + m.box[2]) / 2 * w))
        cy = min(h - 1, int((m.box[1] + m.box[3]) / 2 * h))
        pid = labels[cy, cx]
        m.panel = panel_ids.index(pid) if pid in panel_ids else -1

    analysis = TemplateAnalysis((w, h), mask, lines, markers_px, panels, markers, labels, panel_ids)
    if len(_analysis_cache) > 8:
        _analysis_cache.clear()
    _analysis_cache[key] = analysis
    return analysis


def render_ai_guide(template: Image.Image, analysis: TemplateAnalysis,
                    only_panel: Optional[int] = None) -> Image.Image:
    """High-contrast layout map at the template's original size (see prepare_for_ai).
    `only_panel` draws just that panel (everything else = background)."""
    h, w = analysis.mask.shape
    mask = analysis.mask if only_panel is None else analysis.panel_mask(only_panel)
    out = np.empty((h, w, 3), np.uint8)
    out[:] = GUIDE_BG
    out[mask] = GUIDE_PANEL
    out[analysis.lines & mask] = GUIDE_MESH
    t = max(3, int(round(9 * w / 4096)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1,) * 2)
    eroded = cv2.erode(mask.astype(np.uint8), k)
    out[mask & (eroded == 0)] = GUIDE_OUTLINE
    if only_panel is None and analysis.labels is not None:
        # seams between touching panels (e.g. nose / bonnet), so the model sees them as separate pieces
        grad = cv2.morphologyEx(analysis.labels.astype(np.float32), cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (t, t)))
        out[(grad > 0) & mask] = GUIDE_OUTLINE
    out[analysis.markers_px & mask] = (0, 0, 0)
    img = Image.fromarray(out, "RGB")
    if img.size != template.size:
        img = img.resize(template.size, Image.Resampling.LANCZOS)
    return img


def _where(cx: float, cy: float) -> str:
    v = "top" if cy < 0.34 else "bottom" if cy > 0.66 else "middle"
    hz = "left" if cx < 0.34 else "right" if cx > 0.66 else "centre"
    return "centre" if (v, hz) == ("middle", "centre") else f"{v}-{hz}"


_ROTATION_WORDS = {
    90: ("stored SIDEWAYS (turned 90 degrees clockwise)",
         "the tops of all letters, numbers and figures point to the RIGHT of the sheet and text reads from top to "
         "bottom; the car's ground side is the panel's LEFT edge, so flames/graphics rise from left to right"),
    180: ("stored UPSIDE-DOWN (turned 180 degrees)",
          "every word, number and logo must look upside-down on the sheet; the car's ground side is the panel's TOP "
          "edge, so flames/graphics rise from the top edge DOWNWARD on the sheet"),
    270: ("stored SIDEWAYS (turned 90 degrees anticlockwise)",
          "the tops of all letters, numbers and figures point to the LEFT of the sheet and text reads from bottom to "
          "top; the car's ground side is the panel's RIGHT edge, so flames/graphics rise from right to left"),
}


def describe_layout(analysis: TemplateAnalysis, content_box=(0.0, 0.0, 1.0, 1.0)) -> str:
    """Text list of the measured panels / markers, in % of Image 1 as sent (padding included)."""
    l0, t0, r0, b0 = content_box
    sx, sy = r0 - l0, b0 - t0

    def pct(b):
        l, t, r, bb = b
        return (100 * (l0 + l * sx), 100 * (t0 + t * sy), 100 * (l0 + r * sx), 100 * (t0 + bb * sy))

    lines = [
        "TEMPLATE LAYOUT (measured from Image 1 by the app - trust these numbers):",
        f"- Image 1 is a layout map: {len(analysis.panels)} separate light-grey panels with thick black outlines "
        "on a dark charcoal background. Your output must have exactly these panels, in exactly these places.",
        "- Panels as % of the image (left-right across, top-bottom down), listed top to bottom:",
    ]
    for i, b in enumerate(analysis.panels, 1):
        l, t, r, bb = pct(b)
        shape = "long horizontal strip" if (r - l) > 2.5 * (bb - t) else \
            "tall vertical strip" if (bb - t) > 2.5 * (r - l) else "block"
        lines.append(f"  P{i}: x {l:.0f}-{r:.0f}%, y {t:.0f}-{bb:.0f}% ({shape}, {_where((l + r) / 200, (t + bb) / 200)})")

    if analysis.markers:
        lines.append(f"- {len(analysis.markers)} RACE-NUMBER MARKER(S) (black placeholder digits) - put the race "
                     "number at each of these places, about the same size:")
        for i, m in enumerate(analysis.markers, 1):
            l, t, r, bb = pct(m.box)
            on = f" on P{m.panel + 1}" if m.panel >= 0 else ""
            rot = f", placeholder rotated {m.rotation} deg clockwise" if m.rotation is not None else ""
            lines.append(f"  M{i}: x {l:.0f}-{r:.0f}%, y {t:.0f}-{bb:.0f}%{on}{rot}")

    known = [m for m in analysis.markers if m.rotation is not None and m.panel >= 0]
    if known:
        lines.append("- PANEL ORIENTATION (read from the template file - this is certain, follow it exactly):")
        done = set()
        for m in known:
            if m.panel in done:
                continue
            done.add(m.panel)
            p = f"P{m.panel + 1}"
            if m.rotation == 0:
                lines.append(f"  {p} is stored UPRIGHT: paint its text, numbers and graphics upright on the sheet "
                             "(ground side = bottom edge, graphics rise upward).")
            else:
                head, rule = _ROTATION_WORDS[m.rotation]
                lines.append(f"  {p} is {head}: {rule}. Paint the race number there with the same rotation as the "
                             "placeholder digits.")
        lines.append("  Do not 'fix' these panels to look upright on the sheet - upright on the sheet is WRONG for a "
                     "rotated panel, because the sheet gets wrapped onto the car.")
    elif analysis.markers:
        lines.append("  Each marker also shows which way is up for its panel: paint the race number with the SAME "
                     "rotation as the marker digits, and orient all other artwork on that panel the same way.")
    return "\n".join(lines)


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
        # coarse translation on a 1/3-size copy (cheap), then affine refinement at full working size
        f = 3
        h0, w0 = ref.shape
        ref_s = cv2.resize(ref, (w0 // f, h0 // f), interpolation=cv2.INTER_AREA)
        mov_s = cv2.resize(mov, (w0 // f, h0 // f), interpolation=cv2.INTER_AREA)
        coarse = np.eye(2, 3, dtype=np.float32)
        _, coarse = cv2.findTransformECC(ref_s, mov_s, coarse, cv2.MOTION_TRANSLATION,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 3)
        warp[:, 2] = coarse[:, 2] * f
        _, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
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
        k = min(grow_px * 2 + 1, 15)
        if cv2 is not None:
            m = Image.fromarray(cv2.dilate(np.asarray(m), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))), "L")
        else:
            m = m.filter(ImageFilter.MaxFilter(k))
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
            if 0 < report.match_after < LOW_MATCH_WARNING:
                report.notes.append(
                    f"WARNING: the AI ignored the template layout (panel match only {report.match_after * 100:.0f}%) - "
                    "Generate again, or switch to Gemini 3 Pro Image which follows layouts better")
        elif previous is not None:
            result = align(result, previous, use_islands=False, report=report)
    if do_clip and template is not None:
        result = clip_to_islands(result, template, report=report)
    return result, report
