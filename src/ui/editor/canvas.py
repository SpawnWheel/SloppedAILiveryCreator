"""
EditorCanvas: the zoomable / pannable painting surface of the Livery Paint Editor.

Every tool that changes pixels respects the current selection: brushes and shapes are clipped live, and the
final edit is blended through the selection mask (EditorDocument.commit_raster), so paint can never leave the
selected panels.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QTextEdit,
    QWidget,
)

from src.core.fill import flood_mask, mask_bbox
from src.ui.editor.document import EditorDocument
from src.ui.editor.layers import BLEND_MODES, ItemLayer, RasterLayer, TextLayer, new_image
from src.ui.editor.masks import (
    apply_mask_to_image,
    combine,
    polygon_mask,
    rect_mask,
    rect_slices,
)
from src.ui.qt_image import numpy_rgba_to_qimage, qimage_to_numpy_rgba


class Tool:
    MOVE = "move"
    PANEL = "panel"
    MARQUEE = "marquee"
    LASSO = "lasso"
    BRUSH = "brush"
    ERASER = "eraser"
    GRADIENT = "gradient"
    LINE = "line"
    RECT = "rect"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    FILL = "fill"
    EYEDROPPER = "eyedropper"
    TEXT = "text"
    PAN = "pan"


RASTER_TOOLS = {Tool.BRUSH, Tool.ERASER, Tool.LINE, Tool.RECT, Tool.ELLIPSE, Tool.POLYGON, Tool.FILL, Tool.GRADIENT}
SELECT_TOOLS = {Tool.PANEL, Tool.MARQUEE, Tool.LASSO}
HANDLE_PX = 9
ROTATE_OFFSET_PX = 34
ACCENT = QColor(47, 143, 240)


def _checker_brush() -> QBrush:
    pm = QPixmap(16, 16)
    pm.fill(QColor(92, 94, 99))
    p = QPainter(pm)
    p.fillRect(0, 0, 8, 8, QColor(118, 120, 126))
    p.fillRect(8, 8, 8, 8, QColor(118, 120, 126))
    p.end()
    return QBrush(pm)


class EditorCanvas(QWidget):
    statusMessage = Signal(str)
    colorPicked = Signal(QColor)
    textRequested = Signal(QPointF)          # user clicked with the Text tool
    needPaintLayer = Signal()                # a raster tool was used on a non-raster layer
    zoomChanged = Signal(float)
    cursorMoved = Signal(int, int)
    filesDropped = Signal(list, QPointF)     # [paths], document position
    hoverPanelChanged = Signal(int)          # panel index under the cursor (-1 = none)
    colorUsed = Signal(QColor)               # a paint operation used this colour (recent colours)
    contextMenuAt = Signal(QPointF, object)  # document position, global QPoint

    def __init__(self, doc: EditorDocument, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.doc = doc
        self.doc.changed.connect(self.update)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 200)
        self.setAcceptDrops(True)

        # view
        self.scale = 0.2
        self.offset = QPointF(0, 0)
        self._auto_fit = True
        self._checker = _checker_brush()

        # tool settings
        self.tool = Tool.BRUSH
        self.primary = QColor(230, 30, 40)
        self.secondary = QColor(255, 255, 255)
        self.brush_size = 24
        self.brush_opacity = 1.0
        self.brush_hardness = 100          # 0..100
        self.shape_fill = True
        self.shape_outline = False
        self.fill_tolerance = 40
        self.fill_grow = 2
        self.fill_sample = "guide"  # "guide" | "layer" | "all"
        self.fill_contiguous = True
        self.select_op = "replace"  # replace | add | subtract | intersect
        self.marquee_shape = "rect"  # rect | ellipse
        self.panel_by = "panels"     # panels | colour
        self.wand_tolerance = 32
        self.wand_sample = "all"     # all | layer
        self.gradient_type = "linear"  # linear | radial
        self.gradient_to = "secondary"  # secondary | transparent
        self.show_selection_veil = True
        self.show_selection_edges = True
        self.show_panel_labels = False

        # interaction state
        self._space = False
        self._panning = False
        self._pan_start = QPointF()
        self._pan_offset_start = QPointF()
        self._mouse_doc = QPointF(-1, -1)
        self._mouse_widget = QPointF(-1, -1)
        self._drag_start: Optional[QPointF] = None
        self._drag_now: Optional[QPointF] = None
        self._stroking = False
        self._stroke_buf: Optional[QImage] = None
        self._stroke_rect = QRect()
        self._stroke_last: Optional[QPointF] = None
        self._dab_carry = 0.0
        self._dab_started = False
        self._last_brush_point: Optional[QPointF] = None
        self._eraser_snapshot: Optional[QImage] = None
        self._poly_points: List[QPointF] = []
        self._lasso_points: List[QPointF] = []
        self._sel_drag_op = "replace"
        self._panel_drag: Optional[dict] = None
        self._hover_panel = -1
        self._highlight_panel = -1  # from the Panels dock
        # item transform
        self._xform_mode: Optional[str] = None  # move|rotate|corner|edge-x|edge-y|raster-move
        self._xform_before: Optional[dict] = None
        self._xform_start_doc = QPointF()
        self._raster_move_offset = QPointF()
        self._float: Optional[dict] = None  # moving selected pixels: {img, hole, origin, rect}

        # marching ants
        self._ants_phase = 0
        self._ants = QTimer(self)
        self._ants.setInterval(160)
        self._ants.timeout.connect(self._tick_ants)
        self._ants.start()

    # ================================================================== view
    def view_transform(self) -> QTransform:
        t = QTransform()
        t.translate(self.offset.x(), self.offset.y())
        t.scale(self.scale, self.scale)
        return t

    def to_doc(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self.offset.x()) / self.scale, (p.y() - self.offset.y()) / self.scale)

    def to_widget(self, p: QPointF) -> QPointF:
        return QPointF(p.x() * self.scale + self.offset.x(), p.y() * self.scale + self.offset.y())

    def fit(self) -> None:
        w, h = self.width() - 48, self.height() - 48
        if w <= 0 or h <= 0:
            return
        self.scale = min(w / self.doc.width, h / self.doc.height)
        self.offset = QPointF((self.width() - self.doc.width * self.scale) / 2,
                              (self.height() - self.doc.height * self.scale) / 2)
        self._auto_fit = True
        self.zoomChanged.emit(self.scale)
        self.update()

    def set_zoom(self, new_scale: float, anchor: Optional[QPointF] = None) -> None:
        new_scale = max(0.02, min(new_scale, 16.0))
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        doc_pt = self.to_doc(anchor)
        self.scale = new_scale
        self.offset = QPointF(anchor.x() - doc_pt.x() * self.scale, anchor.y() - doc_pt.y() * self.scale)
        self._auto_fit = False
        self.zoomChanged.emit(self.scale)
        self._update_cursor()
        self.update()

    def zoom_to_rect(self, r: QRect) -> None:
        if r is None or r.isEmpty():
            return
        w, h = self.width() - 80, self.height() - 80
        s = max(0.02, min(16.0, min(w / r.width(), h / r.height())))
        self.scale = s
        c = QPointF(r.center())
        self.offset = QPointF(self.width() / 2 - c.x() * s, self.height() / 2 - c.y() * s)
        self._auto_fit = False
        self.zoomChanged.emit(self.scale)
        self.update()

    def resizeEvent(self, e) -> None:
        if self._auto_fit:
            self.fit()
        super().resizeEvent(e)

    def wheelEvent(self, e: QWheelEvent) -> None:
        mods = e.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier and not mods & Qt.KeyboardModifier.AltModifier:
            # Ctrl+wheel scrolls sideways (Photoshop); plain wheel zooms (the app's original behaviour)
            self.offset += QPointF(e.angleDelta().y() / 2.0, 0)
            self._auto_fit = False
            self.update()
            return
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        self.set_zoom(self.scale * (1.2 ** steps), e.position())

    def _tick_ants(self) -> None:
        if self.doc.selection.active and self.show_selection_edges and self.isVisible():
            self._ants_phase = (self._ants_phase + 1) % 8
            self.update()

    # ================================================================== tools
    def set_tool(self, tool: str) -> None:
        if self._poly_points and tool != Tool.POLYGON:
            self._poly_points = []
        self.tool = tool
        if tool != Tool.PANEL and self._hover_panel != -1:
            self._hover_panel = -1
            self.hoverPanelChanged.emit(-1)
        self._update_cursor()
        self.update()

    def _update_cursor(self) -> None:
        if self._space or self.tool == Tool.PAN:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.tool == Tool.MOVE:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        elif self.tool in (Tool.BRUSH, Tool.ERASER):
            self.setCursor(Qt.CursorShape.BlankCursor if self.brush_size * self.scale > 6 else Qt.CursorShape.CrossCursor)
        elif self.tool == Tool.TEXT:
            self.setCursor(Qt.CursorShape.IBeamCursor)
        elif self.tool in (Tool.FILL, Tool.PANEL):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _paint_target(self) -> Optional[RasterLayer]:
        layer = self.doc.active
        if isinstance(layer, RasterLayer):
            if layer.locked:
                self.statusMessage.emit(f"Layer '{layer.name}' is locked.")
                return None
            if not layer.visible:
                self.statusMessage.emit(f"Layer '{layer.name}' is hidden - show it to paint on it.")
                return None
            return layer
        self.needPaintLayer.emit()
        layer = self.doc.active
        return layer if isinstance(layer, RasterLayer) else None

    def _stroke_pen(self, color: QColor, width: float) -> QPen:
        return QPen(color, max(1.0, width), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

    def _sel_op(self, mods) -> str:
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        if shift and alt:
            return "intersect"
        if shift:
            return "add"
        if alt:
            return "subtract"
        return self.select_op

    def _clip_to_selection(self, p: QPainter) -> None:
        if self.doc.selection.active:
            p.setClipPath(self.doc.selection.path(), Qt.ClipOperation.IntersectClip)

    def set_highlight_panel(self, idx: int) -> None:
        if idx != self._highlight_panel:
            self._highlight_panel = idx
            self.update()

    # ================================================================== painting
    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(28, 29, 32))
        doc_rect_w = QRectF(self.to_widget(QPointF(0, 0)), self.to_widget(QPointF(self.doc.width, self.doc.height)))
        # soft drop shadow under the document
        for i, a in ((6, 18), (3, 30), (1, 50)):
            p.fillRect(doc_rect_w.adjusted(-i, -i + 2, i, i + 2), QColor(0, 0, 0, a))
        p.fillRect(doc_rect_w, self._checker)

        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale < 2.0)
        p.setTransform(self.view_transform())
        p.setClipRect(QRectF(0, 0, self.doc.width, self.doc.height))

        below, above, above_layers = self.doc.caches()
        if below is not None:
            self._draw_scaled_cached(p, below, "below")

        active = self.doc.active
        if active is not None and active.visible:
            if isinstance(active, RasterLayer) and self._xform_mode == "raster-move":
                self._paint_raster_move(p, active)
            else:
                active.paint(p)
            if self._stroking and self._stroke_buf is not None and self.tool == Tool.BRUSH:
                p.save()
                p.setOpacity(self.brush_opacity * active.opacity)
                p.drawImage(0, 0, self._stroke_buf)
                p.restore()

        p.save()
        self._clip_to_selection(p)
        self._paint_shape_preview(p)
        p.restore()

        if above is not None:
            self._draw_scaled_cached(p, above, "above")
        else:
            for layer in above_layers:
                if layer.visible:
                    layer.paint(p)

        g = self.doc.guide
        if g is not None and self.doc.guide_visible and self.doc.guide_opacity > 0:
            p.save()
            p.setOpacity(self.doc.guide_opacity)
            p.setCompositionMode(BLEND_MODES.get(self.doc.guide_blend, BLEND_MODES["Normal"]))
            self._draw_scaled_cached(p, g, "guide")
            p.restore()

        sel = self.doc.selection
        if sel.active and self.show_selection_veil:
            ov = sel.overlay()
            if ov is not None:
                self._draw_scaled_cached(p, ov, "veil")

        self._paint_panel_highlight(p)

        p.setClipping(False)

        # marching ants (cosmetic pens: constant on-screen width at any zoom)
        if sel.active and self.show_selection_edges:
            path = sel.path()
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setBrush(Qt.BrushStyle.NoBrush)
            white = QPen(QColor(255, 255, 255), 1)
            white.setCosmetic(True)
            p.setPen(white)
            p.drawPath(path)
            dash = QPen(QColor(0, 0, 0), 1, Qt.PenStyle.CustomDashLine)
            dash.setCosmetic(True)
            dash.setDashPattern([4, 4])
            dash.setDashOffset(self._ants_phase)
            p.setPen(dash)
            p.drawPath(path)

        p.resetTransform()

        # document border
        p.setPen(QPen(QColor(12, 12, 14), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(doc_rect_w)

        self._paint_panel_labels(p)
        self._paint_item_handles(p)
        self._paint_polygon_overlay(p)
        self._paint_selection_drag(p)
        self._paint_brush_cursor(p)
        p.end()

    def _draw_scaled_cached(self, p: QPainter, img: QImage, key: str) -> None:
        """
        When zoomed out, drawing a 4096px image scaled on every repaint is the main cost.
        Keep a pre-scaled copy per (image, zoom) and draw that 1:1 instead.
        """
        if self.scale >= 0.999:
            p.drawImage(0, 0, img)
            return
        cache = getattr(self, "_scaled_cache", None)
        if cache is None:
            cache = self._scaled_cache = {}
        ck = (img.cacheKey(), round(self.scale, 5))
        entry = cache.get(key)
        if entry is None or entry[0] != ck:
            sw = max(1, int(round(img.width() * self.scale)))
            sh = max(1, int(round(img.height() * self.scale)))
            scaled = img.scaled(sw, sh, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            entry = (ck, scaled)
            cache[key] = entry
        scaled = entry[1]
        p.save()
        p.scale(img.width() / scaled.width(), img.height() / scaled.height())
        p.drawImage(0, 0, scaled)
        p.restore()

    def _paint_raster_move(self, p: QPainter, active: RasterLayer) -> None:
        p.save()
        active._apply_common(p)
        if self._float is not None:
            p.drawImage(0, 0, self._float["hole"])
            self._clip_to_selection(p)
            o = self._float["origin"]
            p.drawImage(QPointF(o.x() + self._raster_move_offset.x(), o.y() + self._raster_move_offset.y()),
                        self._float["img"])
        else:
            p.translate(self._raster_move_offset)
            active.paint_content(p)
        p.restore()

    def _paint_shape_preview(self, p: QPainter) -> None:
        if not (self._drag_start and self._drag_now):
            return
        if self.tool == Tool.GRADIENT:
            p.save()
            p.setOpacity(self.brush_opacity)
            r = self.doc.selection.bbox() or self.doc.rect()
            p.fillRect(r, self._gradient_brush(self._drag_start, self._drag_now))
            p.restore()
            return
        if self.tool not in (Tool.LINE, Tool.RECT, Tool.ELLIPSE):
            return
        p.save()
        p.setOpacity(self.brush_opacity)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_shape(p, self.tool, self._drag_start, self._constrained(self._drag_start, self._drag_now))
        p.restore()

    def _gradient_brush(self, a: QPointF, b: QPointF) -> QBrush:
        c0 = QColor(self.primary)
        c0.setAlpha(255)
        if self.gradient_to == "transparent":
            c1 = QColor(self.primary)
            c1.setAlpha(0)
        else:
            c1 = QColor(self.secondary)
            c1.setAlpha(255)
        if self.gradient_type == "radial":
            g = QRadialGradient(a, max(1.0, math.hypot(b.x() - a.x(), b.y() - a.y())))
        else:
            g = QLinearGradient(a, b)
        g.setColorAt(0.0, c0)
        g.setColorAt(1.0, c1)
        return QBrush(g)

    def _draw_shape(self, p: QPainter, tool: str, a: QPointF, b: QPointF) -> None:
        if tool == Tool.LINE:
            p.setPen(self._stroke_pen(self.primary, self.brush_size))
            p.drawLine(a, b)
            return
        r = QRectF(a, b).normalized()
        p.setPen(self._stroke_pen(self.secondary, self.brush_size) if self.shape_outline else Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self.primary) if self.shape_fill or not self.shape_outline else Qt.BrushStyle.NoBrush)
        if tool == Tool.RECT:
            p.drawRect(r)
        else:
            p.drawEllipse(r)

    def _constrained(self, a: QPointF, b: QPointF) -> QPointF:
        if not (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
            return b
        dx, dy = b.x() - a.x(), b.y() - a.y()
        if self.tool in (Tool.LINE, Tool.GRADIENT):
            ang = math.atan2(dy, dx)
            snap = round(ang / (math.pi / 12)) * (math.pi / 12)
            length = math.hypot(dx, dy)
            return QPointF(a.x() + length * math.cos(snap), a.y() + length * math.sin(snap))
        side = max(abs(dx), abs(dy))
        return QPointF(a.x() + math.copysign(side, dx or 1), a.y() + math.copysign(side, dy or 1))

    def _paint_polygon_overlay(self, p: QPainter) -> None:
        if self.tool != Tool.POLYGON or not self._poly_points:
            return
        pts = [self.to_widget(pt) for pt in self._poly_points]
        pts_live = pts + [self._mouse_widget] if self._mouse_widget.x() >= 0 else pts
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(self.primary)
        c.setAlpha(110)
        p.setBrush(c)
        p.setPen(QPen(QColor(255, 255, 255), 1.5, Qt.PenStyle.DashLine))
        p.drawPolygon(QPolygonF(pts_live))
        p.setBrush(QColor(255, 255, 255))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        for pt in pts:
            p.drawEllipse(pt, 4, 4)

    def _paint_selection_drag(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.tool == Tool.MARQUEE and self._drag_start is not None and self._drag_now is not None:
            r = QRectF(self.to_widget(self._drag_start), self.to_widget(self._drag_now)).normalized()
            for pen in (QPen(QColor(0, 0, 0, 200), 1), QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)):
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(r) if self.marquee_shape == "ellipse" else p.drawRect(r)
        if self.tool == Tool.LASSO and len(self._lasso_points) > 1:
            poly = QPolygonF([self.to_widget(pt) for pt in self._lasso_points])
            for pen in (QPen(QColor(0, 0, 0, 200), 1), QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)):
                p.setPen(pen)
                p.drawPolyline(poly)
        if self.tool == Tool.GRADIENT and self._drag_start is not None and self._drag_now is not None:
            a, b = self.to_widget(self._drag_start), self.to_widget(self._drag_now)
            p.setPen(QPen(QColor(0, 0, 0, 200), 3))
            p.drawLine(a, b)
            p.setPen(QPen(QColor(255, 255, 255), 1.4))
            p.drawLine(a, b)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(a, 3.5, 3.5)
            p.drawEllipse(b, 3.5, 3.5)

    def _paint_panel_highlight(self, p: QPainter) -> None:
        pm = self.doc.panels
        if pm is None:
            return
        for idx, strong in ((self._highlight_panel, False), (self._hover_panel, True)):
            if not (0 <= idx < len(pm)):
                continue
            path = pm.path(idx)
            fill = QColor(ACCENT)
            fill.setAlpha(55 if strong else 35)
            p.save()
            p.setBrush(fill)
            pen = QPen(QColor(120, 190, 255) if strong else ACCENT, 2 if strong else 1.5)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawPath(path)
            p.restore()

    def _paint_panel_labels(self, p: QPainter) -> None:
        pm = self.doc.panels
        if pm is None or not (self.show_panel_labels or self.tool == Tool.PANEL):
            return
        f = QFont()
        f.setPixelSize(11)
        f.setBold(True)
        p.setFont(f)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cov = None
        if self.doc.selection.active:
            cov = pm.coverage(self.doc.selection.mask)
        for info in pm.infos:
            if info.bbox.width() * self.scale < 22 or info.bbox.height() * self.scale < 14:
                continue
            c = self.to_widget(info.centroid)
            text = info.title if info.name else info.code
            w = p.fontMetrics().horizontalAdvance(text) + 10
            r = QRectF(c.x() - w / 2, c.y() - 9, w, 18)
            selected = cov is not None and cov[info.index] >= 0.9
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(47, 143, 240, 230) if selected else QColor(20, 21, 24, 200))
            p.drawRoundedRect(r, 4, 4)
            p.setPen(QColor(255, 255, 255))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_brush_cursor(self, p: QPainter) -> None:
        if self.tool not in (Tool.BRUSH, Tool.ERASER) or self._mouse_widget.x() < 0 or self._space:
            return
        r = self.brush_size * self.scale / 2
        if r <= 3:
            return
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 200), 2.5))
        p.drawEllipse(self._mouse_widget, r, r)
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.2))
        p.drawEllipse(self._mouse_widget, r, r)
        if self.brush_hardness < 98:
            inner = r * max(0.05, self.brush_hardness / 100.0)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.DotLine))
            p.drawEllipse(self._mouse_widget, inner, inner)

    # ---------------------------------------------------------- item handles
    def _handle_points(self, layer: ItemLayer):
        c = [self.to_widget(pt) for pt in layer.corners()]
        mids = [(c[i] + c[(i + 1) % 4]) / 2 for i in range(4)]
        center = (c[0] + c[2]) / 2
        top_mid = mids[0]
        v = top_mid - center
        length = math.hypot(v.x(), v.y()) or 1.0
        rot = top_mid + QPointF(v.x() / length, v.y() / length) * ROTATE_OFFSET_PX
        return c, mids, rot, center

    def _paint_item_handles(self, p: QPainter) -> None:
        layer = self.doc.active
        if not isinstance(layer, ItemLayer) or self.tool != Tool.MOVE or not layer.visible:
            return
        if layer.clip is not None:  # show the panel area the item is confined to
            p.save()
            p.setTransform(self.view_transform())
            pen = QPen(QColor(255, 196, 0, 200), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(layer.clip.path())
            p.restore()
        corners, mids, rot, _ = self._handle_points(layer)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 180), 3))
        p.drawPolygon(QPolygonF(corners))
        p.setPen(QPen(QColor(0, 153, 255), 1.5, Qt.PenStyle.DashLine))
        p.drawPolygon(QPolygonF(corners))
        p.setPen(QPen(QColor(0, 153, 255), 1.5))
        p.drawLine(mids[0], rot)
        p.setBrush(QColor(255, 255, 255))
        p.setPen(QPen(QColor(0, 120, 212), 1.5))
        hs = HANDLE_PX / 2
        for pt in corners:
            p.drawRect(QRectF(pt.x() - hs, pt.y() - hs, HANDLE_PX, HANDLE_PX))
        for pt in mids:
            p.drawEllipse(pt, hs - 1, hs - 1)
        p.setBrush(QColor(0, 153, 255))
        p.drawEllipse(rot, hs, hs)

    def _hit_handle(self, layer: ItemLayer, wpos: QPointF) -> Optional[str]:
        corners, mids, rot, _ = self._handle_points(layer)
        tol = HANDLE_PX + 3

        def near(a: QPointF):
            return abs(a.x() - wpos.x()) <= tol and abs(a.y() - wpos.y()) <= tol

        if near(rot):
            return "rotate"
        for pt in corners:
            if near(pt):
                return "corner"
        for i, pt in enumerate(mids):
            if near(pt):
                return "edge-y" if i in (0, 2) else "edge-x"
        return None

    # ================================================================== mouse
    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus()
        wpos = e.position()
        dpos = self.to_doc(wpos)

        if e.button() == Qt.MouseButton.MiddleButton or (
            e.button() == Qt.MouseButton.LeftButton and (self._space or self.tool == Tool.PAN)
        ):
            self._panning = True
            self._pan_start = wpos
            self._pan_offset_start = QPointF(self.offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if e.button() == Qt.MouseButton.RightButton:
            if self.tool == Tool.POLYGON and self._poly_points:
                self._finish_polygon()
            else:
                self.contextMenuAt.emit(dpos, e.globalPosition().toPoint())
            return

        if e.button() != Qt.MouseButton.LeftButton:
            return

        mods = e.modifiers()
        alt = bool(mods & Qt.KeyboardModifier.AltModifier)
        if self.tool == Tool.EYEDROPPER or (alt and self.tool in (Tool.BRUSH, Tool.FILL, Tool.LINE, Tool.RECT,
                                                                   Tool.ELLIPSE, Tool.POLYGON, Tool.GRADIENT)):
            self._pick_color(dpos)
            return

        if self.tool == Tool.TEXT:
            self.textRequested.emit(dpos)
            return

        if self.tool == Tool.MOVE:
            self._begin_move(wpos, dpos)
            return

        if self.tool == Tool.PANEL:
            self._panel_press(dpos, mods)
            return

        if self.tool in (Tool.MARQUEE, Tool.LASSO):
            self._sel_drag_op = self._sel_op(mods)
            self._drag_start = dpos
            self._drag_now = dpos
            self._lasso_points = [dpos]
            return

        if self.tool == Tool.POLYGON:
            if len(self._poly_points) >= 3:
                first_w = self.to_widget(self._poly_points[0])
                if abs(first_w.x() - wpos.x()) < 10 and abs(first_w.y() - wpos.y()) < 10:
                    self._finish_polygon()
                    return
            self._poly_points.append(dpos)
            self.update()
            return

        target = self._paint_target()
        if target is None:
            return

        if self.tool == Tool.FILL:
            self._bucket_fill(target, dpos)
            return

        if self.tool in (Tool.BRUSH, Tool.ERASER):
            self._begin_stroke(target, dpos, bool(mods & Qt.KeyboardModifier.ShiftModifier))
            return

        if self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE, Tool.GRADIENT):
            self._drag_start = dpos
            self._drag_now = dpos

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        wpos = e.position()
        dpos = self.to_doc(wpos)
        old_w = QPointF(self._mouse_widget)
        self._mouse_widget = wpos
        self._mouse_doc = dpos
        self.cursorMoved.emit(int(dpos.x()), int(dpos.y()))

        if self._panning:
            self.offset = self._pan_offset_start + (wpos - self._pan_start)
            self._auto_fit = False
            self.update()
            return

        if self._stroking:
            self._continue_stroke(dpos)
            return

        if self._xform_mode:
            self._continue_move(wpos, dpos, e.modifiers())
            return

        if self._panel_drag is not None:
            self._panel_drag_move(dpos)
            return

        if self._drag_start is not None:
            self._drag_now = dpos
            if self.tool == Tool.LASSO:
                last = self.to_widget(self._lasso_points[-1]) if self._lasso_points else wpos
                if abs(last.x() - wpos.x()) + abs(last.y() - wpos.y()) >= 2:
                    self._lasso_points.append(dpos)
            self.update()
            return

        if self.tool == Tool.PANEL:
            self._update_hover(dpos)

        if self.tool == Tool.MOVE and isinstance(self.doc.active, ItemLayer):
            h = self._hit_handle(self.doc.active, wpos)
            cursors = {"rotate": Qt.CursorShape.CrossCursor, "corner": Qt.CursorShape.SizeFDiagCursor,
                       "edge-x": Qt.CursorShape.SizeHorCursor, "edge-y": Qt.CursorShape.SizeVerCursor}
            self.setCursor(cursors.get(h, Qt.CursorShape.SizeAllCursor))

        if self.tool in (Tool.BRUSH, Tool.ERASER, Tool.POLYGON):
            r = int(self.brush_size * self.scale / 2) + 6
            if self.tool == Tool.POLYGON:
                self.update()
            else:
                self.update(QRect(int(old_w.x()) - r, int(old_w.y()) - r, r * 2, r * 2))
                self.update(QRect(int(wpos.x()) - r, int(wpos.y()) - r, r * 2, r * 2))

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._panning and e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
            self._update_cursor()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self._stroking:
            self._end_stroke()
            return
        if self._xform_mode:
            self._end_move()
            return
        if self._panel_drag is not None:
            self._panel_drag_end()
            return
        if self._drag_start is not None and self._drag_now is not None:
            a, b = self._drag_start, self._constrained(self._drag_start, self._drag_now)
            self._drag_start = self._drag_now = None
            if self.tool == Tool.MARQUEE:
                self._finish_marquee(a, b)
            elif self.tool == Tool.LASSO:
                self._finish_lasso()
            else:
                target = self._paint_target()
                if target is not None:
                    if self.tool == Tool.GRADIENT:
                        self._commit_gradient(target, a, b)
                    else:
                        self._commit_shape(target, a, b)
            self.update()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if self.tool == Tool.POLYGON and len(self._poly_points) >= 3:
            self._finish_polygon()
        elif self.tool == Tool.MOVE and isinstance(self.doc.active, TextLayer):
            self.textRequested.emit(QPointF(-1, -1))  # edit existing text

    def enterEvent(self, e) -> None:
        # take keyboard focus on hover (so tool shortcuts work) unless the user is typing somewhere
        if self.window().isActiveWindow() and not isinstance(
                QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox, QPlainTextEdit, QTextEdit)):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:
        self._mouse_widget = QPointF(-1, -1)
        if self._hover_panel != -1:
            self._hover_panel = -1
            self.hoverPanelChanged.emit(-1)
        self.update()

    # ---------------------------------------------------------- keyboard
    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        if k == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor()
            return
        if self.tool == Tool.POLYGON and self._poly_points:
            if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._finish_polygon()
                return
            if k == Qt.Key.Key_Escape:
                self._poly_points = []
                self.update()
                return
            if k == Qt.Key.Key_Backspace:
                self._poly_points.pop()
                self.update()
                return
        if k == Qt.Key.Key_Escape and (self._drag_start is not None or self._lasso_points):
            self._drag_start = self._drag_now = None
            self._lasso_points = []
            self.update()
            return
        layer = self.doc.active
        if isinstance(layer, ItemLayer) and k in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            step = 10 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            before = layer.transform_state()
            dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(k, 0)
            dy = {Qt.Key.Key_Up: -step, Qt.Key.Key_Down: step}.get(k, 0)
            layer.x, layer.y = before["x"] + dx, before["y"] + dy
            layer.clamp_to_clip()
            after = layer.transform_state()
            layer.set_transform_state(before)
            self.doc.commit_transform(layer, before, after)
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._update_cursor()
            return
        super().keyReleaseEvent(e)

    # ---------------------------------------------------------- drag & drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths, self.to_doc(e.position()))
            e.acceptProposedAction()

    # ================================================================== selection tools
    def _update_hover(self, dpos: QPointF) -> None:
        pm = self.doc.panels
        idx = pm.index_at(int(dpos.x()), int(dpos.y())) if pm is not None else -1
        if idx != self._hover_panel:
            self._hover_panel = idx
            self.hoverPanelChanged.emit(idx)
            if idx >= 0:
                info = pm.infos[idx]
                self.statusMessage.emit(f"{info.title}  ({info.where}, {info.area * 100:.1f}% of the texture)  -  "
                                        "click = select · Shift+click = add · Alt+click = remove · drag across "
                                        "panels to pick several")
            self.update()

    def _panel_press(self, dpos: QPointF, mods) -> None:
        op = self._sel_op(mods)
        x, y = int(dpos.x()), int(dpos.y())
        if not (0 <= x < self.doc.width and 0 <= y < self.doc.height):
            if op == "replace":
                self.doc.set_selection(None, "Deselect")
            return
        if self.panel_by == "panels":
            pm = self.doc.panels
            if pm is None and self.doc.guide is not None and self.doc.panels_state in ("analysing", "none"):
                self.setCursor(Qt.CursorShape.WaitCursor)
                try:
                    pm = self.doc.ensure_panels()
                finally:
                    self._update_cursor()
            if pm is not None:
                idx = pm.index_at(x, y)
                if idx < 0:
                    if op == "replace":
                        self.doc.set_selection(None, "Deselect")
                        self.statusMessage.emit("Clicked the background - selection cleared.")
                    return
                before = self.doc.selection.copy_mask()
                before_label = self.doc.selection.label
                area = pm.mask([idx])
                new = area if op == "replace" else combine(self.doc.selection.mask, area, op)
                drag_op = "subtract" if op == "subtract" else ("intersect" if op == "intersect" else "add")
                self._panel_drag = {"op": drag_op, "before": before, "before_label": before_label,
                                    "seen": {idx}, "label": pm.infos[idx].title}
                self.doc.set_selection(new, pm.infos[idx].title, record=False)
                return
            if self.doc.guide is None:
                self.statusMessage.emit("No template loaded, so there are no panels - selected by colour instead "
                                        "(load the template in the Template dock for panel selection).")
            else:
                self.statusMessage.emit("Couldn't detect panels in this template - selected the area enclosed by "
                                        "the template's lines instead.")
        self._wand_select(x, y, op)

    def _panel_drag_move(self, dpos: QPointF) -> None:
        pm = self.doc.panels
        d = self._panel_drag
        if pm is None or d is None:
            return
        idx = pm.index_at(int(dpos.x()), int(dpos.y()))
        self._update_hover(dpos)
        if idx < 0 or idx in d["seen"]:
            return
        d["seen"].add(idx)
        new = combine(self.doc.selection.mask, pm.mask([idx]), d["op"])
        self.doc.set_selection(new, f"{len(d['seen'])} panels", record=False)

    def _panel_drag_end(self) -> None:
        d = self._panel_drag
        self._panel_drag = None
        if d is None:
            return
        label = d["label"] if len(d["seen"]) == 1 else f"{len(d['seen'])} panels"
        self.doc.set_selection(self.doc.selection.copy_mask(), label, record=True,
                               before=d["before"], before_label=d["before_label"])

    def _wand_select(self, x: int, y: int, op: str) -> None:
        use_guide = self.panel_by == "panels" and self.doc.guide is not None
        if use_guide:
            sample, tol, grow = self.doc.guide, 40, 2
        else:
            layer = self.doc.active
            if self.wand_sample == "layer" and isinstance(layer, RasterLayer):
                sample = layer.image
            else:
                sample = self.doc.flatten()
            tol, grow = self.wand_tolerance, 0
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            mask = flood_mask(qimage_to_numpy_rgba(sample), (x, y), tol, grow, True)
        finally:
            self._update_cursor()
        if mask is None:
            return
        if use_guide and mask.mean() > 0.97:
            self.statusMessage.emit("The template has no closed outline there - nothing selected.")
            return
        self.doc.apply_selection(mask.astype(np.uint8) * 255, op, "Magic wand")

    def _finish_marquee(self, a: QPointF, b: QPointF) -> None:
        r = QRectF(a, b).normalized().toAlignedRect()
        if r.width() * self.scale < 3 and r.height() * self.scale < 3:
            if self._sel_drag_op == "replace":
                self.doc.set_selection(None, "Deselect")
            return
        m = rect_mask(self.doc.width, self.doc.height, r, self.marquee_shape == "ellipse")
        self.doc.apply_selection(m, self._sel_drag_op,
                                 "Elliptical marquee" if self.marquee_shape == "ellipse" else "Rectangular marquee")

    def _finish_lasso(self) -> None:
        pts = self._lasso_points
        self._lasso_points = []
        if len(pts) < 3:
            if self._sel_drag_op == "replace":
                self.doc.set_selection(None, "Deselect")
            return
        m = polygon_mask(self.doc.width, self.doc.height, pts)
        self.doc.apply_selection(m, self._sel_drag_op, "Lasso")

    # ================================================================== operations
    def _pick_color(self, dpos: QPointF) -> None:
        x, y = int(dpos.x()), int(dpos.y())
        if 0 <= x < self.doc.width and 0 <= y < self.doc.height:
            c = self.doc.sample_color(x, y)
            c.setAlpha(255)
            self.primary = c
            self.colorPicked.emit(c)

    # ---- brush / eraser
    def _begin_stroke(self, target: RasterLayer, dpos: QPointF, shift: bool) -> None:
        self._stroking = True
        self._stroke_rect = QRect()
        self._dab_carry = 0.0
        self._dab_started = False
        if self._stroke_buf is None or self._stroke_buf.size() != target.image.size():
            self._stroke_buf = new_image(self.doc.width, self.doc.height)
        else:
            self._stroke_buf.fill(Qt.GlobalColor.transparent)
        if self.tool == Tool.ERASER:
            self._eraser_snapshot = target.image.copy()
        start = self._last_brush_point if (shift and self._last_brush_point is not None) else dpos
        self._stroke_last = start
        self._stroke_segment(target, start, dpos)

    def _continue_stroke(self, dpos: QPointF) -> None:
        target = self.doc.active
        if not isinstance(target, RasterLayer) or self._stroke_last is None:
            return
        self._stroke_segment(target, self._stroke_last, dpos)
        self._stroke_last = dpos

    def _dabs(self, p: QPainter, a: QPointF, b: QPointF, color: QColor) -> None:
        """Soft brush: stamp radial-gradient dabs along the segment (max-alpha, so dabs don't pile up)."""
        r = self.brush_size / 2.0
        spacing = max(0.75, self.brush_size * 0.12)
        hard = max(0.0, min(0.99, self.brush_hardness / 100.0))
        transparent = QColor(color)
        transparent.setAlpha(0)
        p.setPen(Qt.PenStyle.NoPen)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Lighten)

        def dab(pt: QPointF):
            g = QRadialGradient(pt, r)
            g.setColorAt(0.0, color)
            g.setColorAt(hard, color)
            g.setColorAt(1.0, transparent)
            p.setBrush(QBrush(g))
            p.drawEllipse(pt, r, r)

        if not self._dab_started:
            dab(a)
            self._dab_started = True
            self._dab_carry = 0.0
        length = math.hypot(b.x() - a.x(), b.y() - a.y())
        if length < 1e-6:
            return
        pos = spacing - self._dab_carry  # distance along this segment to the next dab
        while pos <= length:
            t = pos / length
            dab(QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t))
            pos += spacing
        self._dab_carry = length - (pos - spacing)

    def _stroke_segment(self, target: RasterLayer, a: QPointF, b: QPointF) -> None:
        img = self._stroke_buf
        color = QColor(self.primary) if self.tool == Tool.BRUSH else QColor(0, 0, 0)
        color.setAlpha(255)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._clip_to_selection(p)
        if self.brush_hardness >= 98:
            if a == b:
                p.setBrush(color)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(a, self.brush_size / 2, self.brush_size / 2)
            else:
                p.setPen(self._stroke_pen(color, self.brush_size))
                p.drawLine(a, b)
        else:
            self._dabs(p, a, b, color)
        p.end()
        pad = int(self.brush_size / 2) + 3
        seg = QRectF(a, b).normalized().toAlignedRect().adjusted(-pad, -pad, pad, pad)
        seg = seg.intersected(self.doc.rect())
        self._stroke_rect = self._stroke_rect.united(seg)
        if self.tool == Tool.ERASER and self._eraser_snapshot is not None and not seg.isEmpty():
            # live eraser: restore the segment area from the snapshot, then cut the whole stroke out of it
            p = QPainter(target.image)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            p.drawImage(seg.topLeft(), self._eraser_snapshot, seg)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.setOpacity(self.brush_opacity)
            p.drawImage(seg.topLeft(), self._stroke_buf, seg)
            p.end()
        wr = QRectF(self.to_widget(QPointF(seg.topLeft())), self.to_widget(QPointF(seg.bottomRight()))).toAlignedRect()
        r = int(self.brush_size * self.scale) + 8
        self.update(wr.adjusted(-r, -r, r, r))

    def _end_stroke(self) -> None:
        self._stroking = False
        target = self.doc.active
        rect = self._stroke_rect.intersected(QRect(0, 0, self.doc.width, self.doc.height))
        self._last_brush_point = self._stroke_last
        if not isinstance(target, RasterLayer) or rect.isEmpty():
            self._eraser_snapshot = None
            return
        if self.tool == Tool.BRUSH:
            before = target.image.copy(rect)
            p = QPainter(target.image)
            p.setOpacity(self.brush_opacity)
            p.drawImage(rect.topLeft(), self._stroke_buf, rect)
            p.end()
            self._stroke_buf.fill(Qt.GlobalColor.transparent)
            self.doc.commit_raster(target, rect, before, "Brush")
            self.colorUsed.emit(QColor(self.primary))
        else:
            before = self._eraser_snapshot.copy(rect) if self._eraser_snapshot is not None else target.image.copy(rect)
            self._eraser_snapshot = None
            self._stroke_buf.fill(Qt.GlobalColor.transparent)
            self.doc.commit_raster(target, rect, before, "Eraser")

    # ---- shapes
    def _commit_shape(self, target: RasterLayer, a: QPointF, b: QPointF) -> None:
        if (a - b).manhattanLength() < 2:
            return
        pad = int(self.brush_size) + 4
        rect = QRectF(a, b).normalized().toAlignedRect().adjusted(-pad, -pad, pad, pad)
        rect = rect.intersected(QRect(0, 0, self.doc.width, self.doc.height))
        if rect.isEmpty():
            return
        before = target.image.copy(rect)
        p = QPainter(target.image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self.brush_opacity)
        self._draw_shape(p, self.tool, a, b)
        p.end()
        self.doc.commit_raster(target, rect, before, self.tool.capitalize())
        self.colorUsed.emit(QColor(self.primary))

    def _commit_gradient(self, target: RasterLayer, a: QPointF, b: QPointF) -> None:
        if (a - b).manhattanLength() < 2:
            return
        rect = (self.doc.selection.bbox() or self.doc.rect()).intersected(self.doc.rect())
        before = target.image.copy(rect)
        p = QPainter(target.image)
        p.setOpacity(self.brush_opacity)
        p.fillRect(rect, self._gradient_brush(a, b))
        p.end()
        self.doc.commit_raster(target, rect, before, "Gradient")
        self.colorUsed.emit(QColor(self.primary))

    def _finish_polygon(self) -> None:
        pts = self._poly_points
        self._poly_points = []
        if len(pts) < 3:
            self.update()
            return
        target = self._paint_target()
        if target is None:
            self.update()
            return
        poly = QPolygonF(pts)
        pad = int(self.brush_size) + 4
        rect = poly.boundingRect().toAlignedRect().adjusted(-pad, -pad, pad, pad)
        rect = rect.intersected(QRect(0, 0, self.doc.width, self.doc.height))
        before = target.image.copy(rect)
        p = QPainter(target.image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self.brush_opacity)
        p.setPen(self._stroke_pen(self.secondary, self.brush_size) if self.shape_outline else Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self.primary) if self.shape_fill or not self.shape_outline else Qt.BrushStyle.NoBrush)
        p.drawPolygon(poly)
        p.end()
        self.doc.commit_raster(target, rect, before, "Polygon")
        self.colorUsed.emit(QColor(self.primary))

    # ---- bucket fill
    def _bucket_fill(self, target: RasterLayer, dpos: QPointF) -> None:
        x, y = int(dpos.x()), int(dpos.y())
        if not (0 <= x < self.doc.width and 0 <= y < self.doc.height):
            return
        sel = self.doc.selection
        if sel.active and not sel.contains(x, y):
            self.statusMessage.emit("That spot is outside the selection - click inside the selected panels "
                                    "(or Ctrl+D to deselect).")
            return
        if self.fill_sample == "guide" and self.doc.guide is not None:
            sample = self.doc.guide
        elif self.fill_sample == "all":
            sample = self.doc.flatten()
        else:
            sample = target.image
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            arr = qimage_to_numpy_rgba(sample)
            mask = flood_mask(arr, (x, y), self.fill_tolerance, self.fill_grow, self.fill_contiguous)
        finally:
            self._update_cursor()
        if mask is None:
            return
        if sel.active:
            mask &= sel.mask > 0
        bb = mask_bbox(mask)
        if bb is None:
            return
        bx, by, bw, bh = bb
        if mask.mean() > 0.97 and self.fill_sample == "guide":
            self.statusMessage.emit("Fill covered the whole texture - the template has no closed outline there. "
                                    "Try raising 'Grow', lowering tolerance, or select the panel first (W).")
        rect = QRect(bx, by, bw, bh)
        sub = mask[by:by + bh, bx:bx + bw]
        rgba = np.zeros((bh, bw, 4), np.uint8)
        c = self.primary
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c.red(), c.green(), c.blue()
        rgba[..., 3] = (sub * int(255 * self.brush_opacity)).astype(np.uint8)
        fill_img = numpy_rgba_to_qimage(rgba)
        before = target.image.copy(rect)
        p = QPainter(target.image)
        p.drawImage(rect.topLeft(), fill_img)
        p.end()
        self.doc.commit_raster(target, rect, before, "Fill")
        self.colorUsed.emit(QColor(self.primary))

    # ---- move / transform
    def _begin_move(self, wpos: QPointF, dpos: QPointF) -> None:
        layer = self.doc.active
        if isinstance(layer, ItemLayer) and layer.visible:
            h = self._hit_handle(layer, wpos)
            if h:
                self._start_xform(layer, h, dpos)
                return
            if layer.contains(dpos):
                self._start_xform(layer, "move", dpos)
                return
        # click-select the top-most item under the cursor
        for i in range(len(self.doc.layers) - 1, -1, -1):
            l = self.doc.layers[i]
            if isinstance(l, ItemLayer) and l.visible and not l.locked and l.contains(dpos):
                if l.clip is not None and not self._clip_contains(l, dpos):
                    continue  # the part under the cursor is clipped away
                self.doc.set_active(i)
                self._start_xform(l, "move", dpos)
                return
        if isinstance(layer, RasterLayer) and not layer.locked and layer.visible:
            self._xform_mode = "raster-move"
            self._xform_start_doc = dpos
            self._raster_move_offset = QPointF(0, 0)
            self._float = self._make_float(layer) if self.doc.selection.active else None

    @staticmethod
    def _clip_contains(layer, dpos: QPointF) -> bool:
        x, y = int(dpos.x()), int(dpos.y())
        m = layer.clip.mask
        return 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0

    def _make_float(self, layer: RasterLayer) -> Optional[dict]:
        sel = self.doc.selection
        r = sel.bbox()
        if r is None:
            return None
        m = sel.mask[rect_slices(r)]
        crop = layer.image.copy(r)
        floating = apply_mask_to_image(crop, m)
        hole = layer.image.copy()
        p = QPainter(hole)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(r.topLeft(), apply_mask_to_image(crop, 255 - m))
        p.end()
        return {"img": floating, "hole": hole, "origin": r.topLeft(), "rect": r}

    def _start_xform(self, layer: ItemLayer, mode: str, dpos: QPointF) -> None:
        if layer.locked:
            self.statusMessage.emit(f"Layer '{layer.name}' is locked.")
            return
        self._xform_mode = mode
        self._xform_before = layer.transform_state()
        self._xform_start_doc = dpos

    def _continue_move(self, wpos: QPointF, dpos: QPointF, mods) -> None:
        if self._xform_mode == "raster-move":
            self._raster_move_offset = QPointF(round(dpos.x() - self._xform_start_doc.x()),
                                               round(dpos.y() - self._xform_start_doc.y()))
            self.update()
            return
        layer = self.doc.active
        if not isinstance(layer, ItemLayer) or self._xform_before is None:
            return
        b = self._xform_before
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        cx, cy = b["x"], b["y"]
        if self._xform_mode == "move":
            layer.x = b["x"] + dpos.x() - self._xform_start_doc.x()
            layer.y = b["y"] + dpos.y() - self._xform_start_doc.y()
            layer.clamp_to_clip()
        elif self._xform_mode == "rotate":
            a0 = math.degrees(math.atan2(self._xform_start_doc.y() - cy, self._xform_start_doc.x() - cx))
            a1 = math.degrees(math.atan2(dpos.y() - cy, dpos.x() - cx))
            rot = b["rotation"] + (a1 - a0)
            if shift:
                rot = round(rot / 15.0) * 15.0
            layer.rotation = rot
        else:
            # work in the item's un-rotated local frame
            rad = math.radians(-b["rotation"])

            def local(pt: QPointF):
                dx, dy = pt.x() - cx, pt.y() - cy
                return dx * math.cos(rad) - dy * math.sin(rad), dx * math.sin(rad) + dy * math.cos(rad)

            w, h = layer.source_size()
            lx, ly = local(dpos)
            sx0, sy0 = abs(b["sx"]) or 1e-3, abs(b["sy"]) or 1e-3
            if self._xform_mode == "corner":
                if shift:  # free stretch
                    layer.sx = max(0.01, 2 * abs(lx) / w)
                    layer.sy = max(0.01, 2 * abs(ly) / h)
                else:
                    slx, sly = local(self._xform_start_doc)
                    d0 = math.hypot(slx, sly) or 1.0
                    f = math.hypot(lx, ly) / d0
                    layer.sx = max(0.01, sx0 * f)
                    layer.sy = max(0.01, sy0 * f)
            elif self._xform_mode == "edge-x":
                layer.sx = max(0.01, 2 * abs(lx) / w)
            elif self._xform_mode == "edge-y":
                layer.sy = max(0.01, 2 * abs(ly) / h)
        self.doc.changed.emit()

    def _end_move(self) -> None:
        mode = self._xform_mode
        self._xform_mode = None
        layer = self.doc.active
        if mode == "raster-move":
            off = self._raster_move_offset
            self._raster_move_offset = QPointF(0, 0)
            fl = self._float
            self._float = None
            if isinstance(layer, RasterLayer) and (off.x() or off.y()):
                if fl is not None:
                    self._commit_float_move(layer, fl, off)
                else:
                    full = QRect(0, 0, self.doc.width, self.doc.height)
                    before = layer.image.copy()
                    moved = new_image(self.doc.width, self.doc.height)
                    p = QPainter(moved)
                    p.drawImage(off, before)
                    p.end()
                    layer.image = moved
                    self.doc.commit_raster(layer, full, before, "Move layer pixels", masked=False)
            self.update()
            return
        if isinstance(layer, ItemLayer) and self._xform_before is not None:
            after = layer.transform_state()
            before = self._xform_before
            self._xform_before = None
            self.doc.commit_transform(layer, before, after)

    def _commit_float_move(self, layer: RasterLayer, fl: dict, off: QPointF) -> None:
        """Selected pixels moved by `off`, but kept inside the selection (they move *within* the panels)."""
        r: QRect = fl["rect"]
        m = self.doc.selection.mask[rect_slices(r)]
        before = layer.image.copy(r)
        shifted = new_image(r.width(), r.height())
        p = QPainter(shifted)
        p.drawImage(QPointF(off.x(), off.y()), fl["img"])
        p.end()
        shifted = apply_mask_to_image(shifted, m)
        result = fl["hole"].copy(r)
        p = QPainter(result)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        p.drawImage(0, 0, shifted)
        p.end()
        p = QPainter(layer.image)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(r.topLeft(), result)
        p.end()
        self.doc.commit_raster(layer, r, before, "Move selection", masked=False)
