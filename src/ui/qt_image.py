"""Fast, copy-safe conversions between PIL images and Qt QImage/QPixmap."""

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtGui import QImage, QPixmap


def pil_to_qimage(img: Image.Image, premultiplied: bool = True) -> QImage:
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    q = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888).copy()
    if premultiplied:
        q = q.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    return q


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    return QPixmap.fromImage(pil_to_qimage(img, premultiplied=False))


def qimage_to_numpy_rgba(q: QImage) -> np.ndarray:
    """Returns an (H, W, 4) uint8 RGBA (straight alpha) copy of the QImage."""
    q = q.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h, bpl = q.width(), q.height(), q.bytesPerLine()
    buf = np.frombuffer(q.constBits(), dtype=np.uint8, count=bpl * h).reshape(h, bpl)
    return buf[:, : w * 4].reshape(h, w, 4).copy()


def numpy_rgba_to_qimage(arr: np.ndarray, premultiplied: bool = True) -> QImage:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    q = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    if premultiplied:
        q = q.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    return q


def qimage_to_pil(q: Optional[QImage]) -> Optional[Image.Image]:
    if q is None or q.isNull():
        return None
    return Image.fromarray(qimage_to_numpy_rgba(q), "RGBA")
