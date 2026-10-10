"""
Mask helpers shared by selections, layer clips and the canvas.

A "mask" is a numpy uint8 array (H, W) in document pixels: 0 = outside, 255 = inside, values in between are
soft (feathered) edges.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QImage, QPainterPath, QPolygonF

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------------------------------------- geometry
def mask_bbox(mask: Optional[np.ndarray], threshold: int = 0) -> Optional[QRect]:
    """Bounding rectangle of pixels > threshold, or None when empty."""
    if mask is None:
        return None
    m = mask > threshold
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return QRect(int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))


def mask_to_path(mask: Optional[np.ndarray], threshold: int = 127) -> QPainterPath:
    """Outline of the mask as a QPainterPath (odd-even fill, so holes stay holes)."""
    path = QPainterPath()
    path.setFillRule(Qt.FillRule.OddEvenFill)
    if mask is None:
        return path
    binary = (mask > threshold).astype(np.uint8)
    if cv2 is not None:
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 2:
                pts = c.reshape(-1, 2)
                if len(pts):
                    x, y = pts[0]
                    path.addRect(float(x), float(y), 1.0, 1.0)
                continue
            pts = c.reshape(-1, 2).astype(np.float64) + 0.5
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            path.closeSubpath()
    else:  # pragma: no cover - coarse fallback: bounding box
        r = mask_bbox(binary)
        if r is not None:
            path.addRect(r.x(), r.y(), r.width(), r.height())
    return path


# ---------------------------------------------------------------------------------------------- combine
def combine(old: Optional[np.ndarray], new: np.ndarray, op: str) -> Optional[np.ndarray]:
    """op: replace | add | subtract | intersect.  Returns None when the result is empty."""
    if op == "replace":
        out = new
    elif old is None:  # adding to nothing = the new area; subtracting / intersecting nothing = nothing
        out = new if op == "add" else None
    elif op == "add":
        out = np.maximum(old, new)
    elif op == "subtract":
        out = (old.astype(np.int16) - new).clip(0, 255).astype(np.uint8)
    elif op == "intersect":
        out = np.minimum(old, new)
    else:
        out = new
    if out is None or not out.any():
        return None
    return out


def grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px == 0 or cv2 is None:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(px) * 2 + 1, abs(px) * 2 + 1))
    return cv2.dilate(mask, k) if px > 0 else cv2.erode(mask, k)


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or cv2 is None:
        return mask
    k = px * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), px / 2.0)


def polygon_mask(w: int, h: int, pts) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if cv2 is not None and len(pts) >= 3:
        arr = np.array([[int(round(p.x())), int(round(p.y()))] for p in pts], np.int32)
        cv2.fillPoly(m, [arr], 255, lineType=cv2.LINE_AA)
    return m


def rect_mask(w: int, h: int, rect: QRect, ellipse: bool = False) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    r = rect.normalized()
    if r.width() < 1 or r.height() < 1:
        return m
    if ellipse and cv2 is not None:
        c = (int(r.center().x()), int(r.center().y()))
        cv2.ellipse(m, c, (max(1, r.width() // 2), max(1, r.height() // 2)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    else:
        x0, y0 = max(0, r.x()), max(0, r.y())
        x1, y1 = min(w, r.x() + r.width()), min(h, r.y() + r.height())
        m[y0:y1, x0:x1] = 255
    return m


# ---------------------------------------------------------------------------------------------- Qt bridges
def mask_to_alpha_qimage(mask: np.ndarray, rgb=(255, 255, 255)) -> QImage:
    """Premultiplied ARGB image whose alpha is the mask - handy for DestinationIn compositing."""
    h, w = mask.shape
    arr = np.empty((h, w, 4), np.uint8)
    a = mask.astype(np.uint16)
    # Format_ARGB32_Premultiplied is BGRA in memory on little-endian machines
    arr[..., 0] = (a * rgb[2] // 255).astype(np.uint8)
    arr[..., 1] = (a * rgb[1] // 255).astype(np.uint8)
    arr[..., 2] = (a * rgb[0] // 255).astype(np.uint8)
    arr[..., 3] = mask
    return QImage(arr.data, w, h, w * 4, QImage.Format.Format_ARGB32_Premultiplied).copy()


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    h, w = mask.shape
    m = np.ascontiguousarray(mask, np.uint8)
    img = QImage(m.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def mask_from_png_bytes(data: bytes) -> Optional[np.ndarray]:
    img = QImage()
    if not img.loadFromData(data, "PNG"):
        return None
    img = img.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return buf[:, :w].copy()


def qimage_view(img: QImage) -> np.ndarray:
    """Writable (H, W, 4) numpy view of a 32-bit QImage (BGRA order for ARGB32 formats)."""
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.bits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return buf[:, : w * 4].reshape(h, w, 4)


def qimage_alpha(img: QImage) -> np.ndarray:
    """(H, W) uint8 alpha of a premultiplied ARGB32 image (copy)."""
    img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return buf[:, : w * 4].reshape(h, w, 4)[..., 3].copy()


def blend_masked(before: QImage, after: QImage, mask: np.ndarray) -> QImage:
    """
    Per-pixel mix of two same-size premultiplied images: after where mask=255, before where 0.
    Mixing premultiplied pixels linearly is exact, so soft edges stay clean.
    """
    a = after.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    b = before.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    out = a.copy()
    va, vb, vo = qimage_view(a), qimage_view(b), qimage_view(out)
    m = mask.astype(np.uint16)[..., None]
    vo[...] = ((va.astype(np.uint16) * m + vb.astype(np.uint16) * (255 - m) + 127) // 255).astype(np.uint8)
    return out


def apply_mask_to_image(img: QImage, mask: np.ndarray) -> QImage:
    """Returns a copy of img with its pixels multiplied by mask (outside becomes transparent)."""
    out = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
    v = qimage_view(out)
    m = mask.astype(np.uint16)[..., None]
    v[...] = ((v.astype(np.uint16) * m + 127) // 255).astype(np.uint8)
    return out


def clamp_rect(r: QRect, w: int, h: int) -> QRect:
    return r.intersected(QRect(0, 0, w, h))


def rect_slices(r: QRect) -> Tuple[slice, slice]:
    return slice(r.y(), r.y() + r.height()), slice(r.x(), r.x() + r.width())
