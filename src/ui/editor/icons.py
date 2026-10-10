"""
Monochrome vector icons for the paint editor, drawn with QPainter so they look the same on every
machine (emoji differ wildly between Windows / macOS / Linux and clash with a Photoshop-style UI).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Dict

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

S = 64  # drawing grid; icons are rendered at 2x for crisp HiDPI

FG = QColor(214, 217, 222)
FG_ACTIVE = QColor(255, 255, 255)
FG_DISABLED = QColor(96, 100, 108)


def _pen(c: QColor, w: float = 4.0, dash: bool = False) -> QPen:
    pen = QPen(c, w, Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine,
               Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    if dash:
        pen.setDashPattern([1.6, 1.4])
    return pen


# ------------------------------------------------------------------------------------------ drawings
def _move(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.drawLine(QPointF(32, 8), QPointF(32, 56))
    p.drawLine(QPointF(8, 32), QPointF(56, 32))
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    for pts in ([(32, 4), (24, 14), (40, 14)], [(32, 60), (24, 50), (40, 50)],
                [(4, 32), (14, 24), (14, 40)], [(60, 32), (50, 24), (50, 40)]):
        p.drawPolygon(QPolygonF([QPointF(*q) for q in pts]))


def _panel(p: QPainter, c: QColor):
    # two panel shapes, one highlighted, with a pointer
    p.setPen(_pen(c, 3))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(6, 14)
    path.lineTo(30, 8)
    path.lineTo(34, 30)
    path.lineTo(10, 36)
    path.closeSubpath()
    p.drawPath(path)
    fill = QColor(c)
    fill.setAlpha(110)
    p.setBrush(fill)
    path2 = QPainterPath()
    path2.moveTo(36, 12)
    path2.lineTo(58, 18)
    path2.lineTo(52, 40)
    path2.lineTo(38, 34)
    path2.closeSubpath()
    p.drawPath(path2)
    # cursor arrow
    p.setBrush(c)
    p.setPen(QPen(QColor(30, 32, 36), 2))
    arrow = QPolygonF([QPointF(30, 30), QPointF(30, 60), QPointF(37, 53), QPointF(42, 62),
                       QPointF(47, 60), QPointF(42, 51), QPointF(52, 50)])
    p.drawPolygon(arrow)


def _marquee(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4, dash=True))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(9, 13, 46, 38))


def _lasso(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(22, 46)
    path.cubicTo(2, 40, 6, 10, 32, 10)
    path.cubicTo(58, 10, 62, 36, 40, 42)
    path.cubicTo(32, 44, 26, 44, 22, 46)
    p.drawPath(path)
    tail = QPainterPath()
    tail.moveTo(22, 46)
    tail.cubicTo(16, 52, 24, 58, 18, 62)
    p.drawPath(tail)


def _brush(p: QPainter, c: QColor):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.save()
    p.translate(32, 32)
    p.rotate(45)
    p.drawRoundedRect(QRectF(-4, -30, 8, 30), 3, 3)          # handle
    p.drawRect(QRectF(-6, 0, 12, 7))                         # ferrule
    tip = QPainterPath()
    tip.moveTo(-6, 9)
    tip.lineTo(6, 9)
    tip.cubicTo(7, 20, 2, 26, 0, 30)
    tip.cubicTo(-2, 26, -7, 20, -6, 9)
    p.drawPath(tip)
    p.restore()


def _eraser(p: QPainter, c: QColor):
    p.save()
    p.translate(32, 34)
    p.rotate(-40)
    p.setPen(_pen(c, 3.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(-22, -11, 44, 22), 4, 4)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(-22, -11, 18, 22), 4, 4)
    p.restore()
    p.setPen(_pen(c, 3))
    p.drawLine(QPointF(30, 58), QPointF(58, 58))


def _gradient(p: QPainter, c: QColor):
    from PySide6.QtGui import QLinearGradient

    g = QLinearGradient(QPointF(8, 0), QPointF(56, 0))
    g.setColorAt(0, c)
    t = QColor(c)
    t.setAlpha(0)
    g.setColorAt(1, t)
    p.setPen(_pen(c, 3))
    p.setBrush(QBrush(g))
    p.drawRoundedRect(QRectF(7, 14, 50, 36), 4, 4)


def _bucket(p: QPainter, c: QColor):
    p.save()
    p.translate(28, 34)
    p.rotate(-35)
    p.setPen(_pen(c, 3.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(-14, -14, 28, 28))
    p.drawArc(QRectF(-10, -26, 20, 22), 0, 180 * 16)
    p.restore()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    drop = QPainterPath()
    drop.moveTo(52, 30)
    drop.cubicTo(46, 42, 46, 50, 52, 50)
    drop.cubicTo(58, 50, 58, 42, 52, 30)
    p.drawPath(drop)


def _line(p: QPainter, c: QColor):
    p.setPen(_pen(c, 5))
    p.drawLine(QPointF(10, 54), QPointF(54, 10))


def _rect(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(10, 14, 44, 36))


def _ellipse(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(8, 14, 48, 36))


def _polygon(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    pts = [QPointF(32 + 24 * math.cos(math.radians(-90 + i * 72)), 34 + 24 * math.sin(math.radians(-90 + i * 72)))
           for i in range(5)]
    p.drawPolygon(QPolygonF(pts))
    p.setBrush(c)
    for q in pts:
        p.drawEllipse(q, 3, 3)


def _eyedropper(p: QPainter, c: QColor):
    p.save()
    p.translate(32, 32)
    p.rotate(45)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawRoundedRect(QRectF(-7, -28, 14, 14), 5, 5)
    p.drawRect(QRectF(-10, -15, 20, 5))
    p.setPen(_pen(c, 3.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(-4, -10, 8, 30))
    p.drawLine(QPointF(0, 20), QPointF(0, 27))
    p.restore()


def _text(p: QPainter, c: QColor):
    f = QFont("Georgia")
    f.setPixelSize(50)
    f.setBold(True)
    p.setFont(f)
    p.setPen(c)
    p.drawText(QRectF(0, 0, S, S), Qt.AlignmentFlag.AlignCenter, "T")


def _hand(p: QPainter, c: QColor):
    p.setPen(_pen(c, 3.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(20, 58)
    path.cubicTo(12, 48, 6, 38, 10, 34)
    path.cubicTo(13, 31, 18, 36, 20, 40)
    path.lineTo(20, 14)
    path.cubicTo(20, 9, 27, 9, 27, 14)
    path.lineTo(27, 32)
    path.lineTo(27, 9)
    path.cubicTo(27, 4, 34, 4, 34, 9)
    path.lineTo(34, 32)
    path.lineTo(34, 12)
    path.cubicTo(34, 7, 41, 7, 41, 12)
    path.lineTo(41, 33)
    path.lineTo(41, 18)
    path.cubicTo(41, 13, 48, 13, 48, 18)
    path.lineTo(48, 42)
    path.cubicTo(48, 52, 44, 56, 42, 58)
    p.drawPath(path)


def _zoom(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QRectF(8, 8, 34, 34))
    p.setPen(_pen(c, 7))
    p.drawLine(QPointF(38, 38), QPointF(56, 56))


# small UI glyphs ---------------------------------------------------------------------------------
def _undo(p: QPainter, c: QColor):
    p.setPen(_pen(c, 5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(16, 26)
    path.cubicTo(28, 14, 56, 16, 54, 40)
    path.cubicTo(53, 50, 46, 54, 40, 54)
    p.drawPath(path)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF([QPointF(8, 16), QPointF(26, 14), QPointF(14, 32)]))


def _redo(p: QPainter, c: QColor):
    p.translate(S, 0)
    p.scale(-1, 1)
    _undo(p, c)


def _eye(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(4, 32)
    path.cubicTo(18, 12, 46, 12, 60, 32)
    path.cubicTo(46, 52, 18, 52, 4, 32)
    p.drawPath(path)
    p.setBrush(c)
    p.drawEllipse(QPointF(32, 32), 8, 8)


def _lock(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(20, 8, 24, 28), 0, 180 * 16)
    p.drawLine(QPointF(20, 22), QPointF(20, 30))
    p.drawLine(QPointF(44, 22), QPointF(44, 30))
    p.setBrush(c)
    p.drawRoundedRect(QRectF(14, 28, 36, 28), 4, 4)


def _plus(p: QPainter, c: QColor):
    p.setPen(_pen(c, 5))
    p.drawLine(QPointF(32, 12), QPointF(32, 52))
    p.drawLine(QPointF(12, 32), QPointF(52, 32))


def _trash(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(12, 16), QPointF(52, 16))
    p.drawLine(QPointF(26, 16), QPointF(28, 9))
    p.drawLine(QPointF(28, 9), QPointF(36, 9))
    p.drawLine(QPointF(36, 9), QPointF(38, 16))
    path = QPainterPath()
    path.moveTo(17, 20)
    path.lineTo(20, 56)
    path.lineTo(44, 56)
    path.lineTo(47, 20)
    p.drawPath(path)
    p.drawLine(QPointF(28, 26), QPointF(28, 48))
    p.drawLine(QPointF(36, 26), QPointF(36, 48))


def _duplicate(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(8, 8, 32, 32), 3, 3)
    p.setBrush(QColor(40, 42, 46))
    p.drawRoundedRect(QRectF(24, 24, 32, 32), 3, 3)


def _merge(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.drawLine(QPointF(32, 8), QPointF(32, 40))
    p.drawLine(QPointF(20, 28), QPointF(32, 40))
    p.drawLine(QPointF(44, 28), QPointF(32, 40))
    p.drawLine(QPointF(10, 52), QPointF(54, 52))


def _up(p: QPainter, c: QColor):
    p.setPen(_pen(c, 5))
    p.drawLine(QPointF(32, 52), QPointF(32, 14))
    p.drawLine(QPointF(18, 28), QPointF(32, 14))
    p.drawLine(QPointF(46, 28), QPointF(32, 14))


def _down(p: QPainter, c: QColor):
    p.translate(0, S)
    p.scale(1, -1)
    _up(p, c)


def _clip(p: QPainter, c: QColor):
    # dashed panel with an image inside
    p.setPen(_pen(c, 3.5, dash=True))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(6, 10, 52, 44), 4, 4)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawPolygon(QPolygonF([QPointF(14, 46), QPointF(28, 26), QPointF(38, 40), QPointF(44, 32), QPointF(52, 46)]))
    p.drawEllipse(QPointF(44, 20), 4, 4)


def _fit(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    for (x, y, dx, dy) in ((8, 8, 1, 1), (56, 8, -1, 1), (8, 56, 1, -1), (56, 56, -1, -1)):
        p.drawLine(QPointF(x, y), QPointF(x + 14 * dx, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + 14 * dy))


def _image(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(6, 10, 52, 44), 4, 4)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawPolygon(QPolygonF([QPointF(12, 48), QPointF(26, 28), QPointF(36, 40), QPointF(42, 32), QPointF(52, 48)]))
    p.drawEllipse(QPointF(44, 22), 4, 4)


def _swap(p: QPainter, c: QColor):
    p.setPen(_pen(c, 4))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(14, 46)
    path.cubicTo(14, 22, 22, 14, 46, 14)
    p.drawPath(path)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF([QPointF(44, 4), QPointF(58, 14), QPointF(44, 24)]))
    p.drawPolygon(QPolygonF([QPointF(4, 44), QPointF(14, 58), QPointF(24, 44)]))


_DRAW: Dict[str, Callable[[QPainter, QColor], None]] = {
    "move": _move, "panel": _panel, "marquee": _marquee, "lasso": _lasso, "brush": _brush, "eraser": _eraser,
    "gradient": _gradient, "fill": _bucket, "line": _line, "rect": _rect, "ellipse": _ellipse,
    "polygon": _polygon, "eyedropper": _eyedropper, "text": _text, "pan": _hand, "zoom": _zoom,
    "undo": _undo, "redo": _redo, "eye": _eye, "lock": _lock, "plus": _plus, "trash": _trash,
    "duplicate": _duplicate, "merge": _merge, "up": _up, "down": _down, "clip": _clip, "fit": _fit,
    "image": _image, "swap": _swap,
}


def _pixmap(name: str, color: QColor, size: int) -> QPixmap:
    pm = QPixmap(size * 2, size * 2)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.scale(size * 2 / S, size * 2 / S)
    fn = _DRAW.get(name)
    if fn is not None:
        fn(p, color)
    p.end()
    pm.setDevicePixelRatio(2.0)
    return pm


@lru_cache(maxsize=None)
def icon(name: str, size: int = 22) -> QIcon:
    ic = QIcon()
    ic.addPixmap(_pixmap(name, FG, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(_pixmap(name, FG_ACTIVE, size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(_pixmap(name, FG_ACTIVE, size), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(_pixmap(name, FG_DISABLED, size), QIcon.Mode.Disabled, QIcon.State.Off)
    return ic


def has_icon(name: str) -> bool:
    return name in _DRAW
