"""
Interactive Drawing Canvas & Annotation Viewport for Livery Refinement.
Supports freehand drawing, arrows, bounding boxes, text callouts, eraser, and undo.
"""

import math
from typing import List, Optional
from PIL import Image

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QInputDialog,
    QSizePolicy,
    QWidget,
)


class ToolType:
    PEN = "pen"
    ARROW = "arrow"
    RECT = "rect"
    TEXT = "text"
    ERASER = "eraser"


class AnnotationCanvas(QWidget):
    """
    Interactive canvas displaying the 2D livery texture with a high-resolution
    transparent annotation overlay layer for user sketches, arrows, and notes.
    """

    annotationsChanged = Signal(bool)  # True if has annotations, False if empty

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet(
            """
            AnnotationCanvas {
                background-color: #121418;
                border: 1px solid #2d313b;
                border-radius: 8px;
            }
            """
        )

        # Base images
        self._base_image: Optional[Image.Image] = None
        self._base_pixmap: Optional[QPixmap] = None
        self._wireframe_image: Optional[Image.Image] = None
        self._wireframe_pixmap: Optional[QPixmap] = None
        self._show_wireframe: bool = False
        self._show_annotations: bool = True

        # Annotation layer (native 4K canvas size 4096x4096 for maximum fidelity)
        self._canvas_size = 4096
        self._overlay_image = QImage(
            self._canvas_size, self._canvas_size, QImage.Format.Format_ARGB32_Premultiplied
        )
        self._overlay_image.fill(Qt.GlobalColor.transparent)

        # Undo stack for annotations (stores copies of QImage overlay)
        self._undo_stack: List[QImage] = []
        self._max_undo = 15
        self._stroke_count: int = 0

        # Drawing state & active tool settings
        self.active_tool: str = ToolType.PEN
        self.pen_color: QColor = QColor(255, 59, 48)  # High-visibility red by default
        self.pen_size: int = 16  # in canvas space (relative to 4096)
        self.eraser_size: int = 64

        # Drag tracking
        self._is_drawing: bool = False
        self._last_canvas_pt: Optional[QPoint] = None
        self._start_canvas_pt: Optional[QPoint] = None
        self._current_canvas_pt: Optional[QPoint] = None

        self._placeholder_text = (
            "2D Texture Viewport & Annotation Canvas\n\n"
            "• Generate or Load a Livery\n"
            "• Use the Toolbar to Draw Notes & Arrows\n"
            "• Refine with AI or Commit when satisfied"
        )
        self._update_cursor()

    # ==========================================================================
    # Public Texture & Layer API
    # ==========================================================================

    def set_texture(self, pil_img: Optional[Image.Image]) -> None:
        """Sets the clean base livery texture."""
        self._base_image = pil_img
        if pil_img is not None:
            rgb = pil_img.convert("RGBA")
            data = rgb.tobytes("raw", "RGBA")
            qimg = QImage(data, rgb.width, rgb.height, QImage.Format.Format_RGBA8888)
            self._base_pixmap = QPixmap.fromImage(qimg)
        else:
            self._base_pixmap = None

        self.update()

    def get_base_image(self) -> Optional[Image.Image]:
        return self._base_image

    def set_wireframe(self, wf: Optional[Image.Image]) -> None:
        self._wireframe_image = wf
        if wf is not None:
            rgb = wf.convert("RGBA")
            data = rgb.tobytes("raw", "RGBA")
            qimg = QImage(data, rgb.width, rgb.height, QImage.Format.Format_RGBA8888)
            self._wireframe_pixmap = QPixmap.fromImage(qimg)
        else:
            self._wireframe_pixmap = None
        self.update()

    def set_show_wireframe(self, show: bool) -> None:
        self._show_wireframe = show
        self.update()

    def set_show_annotations(self, show: bool) -> None:
        self._show_annotations = show
        self.update()

    def clear_texture(self) -> None:
        self._base_image = None
        self._base_pixmap = None
        self.clear_annotations()
        self.update()

    def clear_annotations(self) -> None:
        """Clears all drawings from the annotation layer."""
        self._save_undo()
        self._overlay_image.fill(Qt.GlobalColor.transparent)
        self._stroke_count = 0
        self.annotationsChanged.emit(False)
        self.update()

    def has_annotations(self) -> bool:
        """Checks if the annotation layer has any active annotations."""
        return self._stroke_count > 0

    def undo(self) -> None:
        """Reverts the last drawn stroke."""
        if self._undo_stack:
            self._overlay_image = self._undo_stack.pop()
            self._stroke_count = max(0, self._stroke_count - 1)
            self.annotationsChanged.emit(self.has_annotations())
            self.update()

    def _save_undo(self) -> None:
        if len(self._undo_stack) >= self._max_undo:
            self._undo_stack.pop(0)
        self._undo_stack.append(self._overlay_image.copy())

    def get_annotated_image(self) -> Optional[Image.Image]:
        """
        Returns a high-resolution PIL Image compositing the base livery
        with the user's hand-drawn annotations burned in.
        """
        if self._base_image is None:
            return None

        # Build QImage composite
        w, h = self._base_image.size
        composite_qimg = QImage(w, h, QImage.Format.Format_RGBA8888)
        composite_qimg.fill(Qt.GlobalColor.transparent)

        painter = QPainter(composite_qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # 1. Base pixmap
        if self._base_pixmap:
            painter.drawPixmap(0, 0, w, h, self._base_pixmap)

        # 2. Overlay
        painter.drawImage(QRect(0, 0, w, h), self._overlay_image)
        painter.end()

        # Convert back to PIL Image
        ptr = composite_qimg.constBits()
        pil_composite = Image.frombuffer("RGBA", (w, h), bytes(ptr), "raw", "RGBA", 0, 1)
        return pil_composite

    # ==========================================================================
    # Tool Selection & Configuration
    # ==========================================================================

    def set_tool(self, tool: str) -> None:
        self.active_tool = tool
        self._update_cursor()

    def set_color(self, color: QColor) -> None:
        self.pen_color = color

    def set_brush_size(self, size: int) -> None:
        self.pen_size = max(4, min(size, 128))

    def _update_cursor(self) -> None:
        if self.active_tool in (ToolType.PEN, ToolType.ARROW, ToolType.RECT):
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.active_tool == ToolType.TEXT:
            self.setCursor(Qt.CursorShape.IBeamCursor)
        elif self.active_tool == ToolType.ERASER:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ==========================================================================
    # Coordinate Mapping (Widget <-> Canvas)
    # ==========================================================================

    def get_content_rect(self) -> QRect:
        """Returns the centered square viewport rect inside the widget."""
        w, h = self.width() - 16, self.height() - 16
        if w <= 0 or h <= 0:
            return QRect()

        side = min(w, h)
        x = (self.width() - side) // 2
        y = (self.height() - side) // 2
        return QRect(x, y, side, side)

    def _widget_to_canvas(self, pos: QPoint) -> Optional[QPoint]:
        rect = self.get_content_rect()
        if not rect.contains(pos) or rect.width() == 0 or rect.height() == 0:
            return None

        rel_x = (pos.x() - rect.x()) / rect.width()
        rel_y = (pos.y() - rect.y()) / rect.height()
        cx = int(rel_x * self._canvas_size)
        cy = int(rel_y * self._canvas_size)
        return QPoint(max(0, min(cx, self._canvas_size - 1)), max(0, min(cy, self._canvas_size - 1)))

    # ==========================================================================
    # Mouse Interaction & Drawing
    # ==========================================================================

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        c_pt = self._widget_to_canvas(event.pos())
        if not c_pt or self._base_image is None:
            return

        if self.active_tool == ToolType.TEXT:
            self._handle_text_tool(c_pt)
            return

        self._save_undo()
        self._is_drawing = True
        self._start_canvas_pt = c_pt
        self._last_canvas_pt = c_pt
        self._current_canvas_pt = c_pt

        if self.active_tool == ToolType.PEN:
            self._draw_pen_stroke(c_pt, c_pt)
        elif self.active_tool == ToolType.ERASER:
            self._erase_at(c_pt)

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        c_pt = self._widget_to_canvas(event.pos())
        if not self._is_drawing or not c_pt:
            return

        self._current_canvas_pt = c_pt

        if self.active_tool == ToolType.PEN:
            self._draw_pen_stroke(self._last_canvas_pt, c_pt)
            self._last_canvas_pt = c_pt
            self.update()
        elif self.active_tool == ToolType.ERASER:
            self._erase_at(c_pt)
            self._last_canvas_pt = c_pt
            self.update()
        elif self.active_tool in (ToolType.ARROW, ToolType.RECT):
            # Live preview updated in paintEvent
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._is_drawing:
            super().mouseReleaseEvent(event)
            return

        self._is_drawing = False
        c_pt = self._widget_to_canvas(event.pos()) or self._current_canvas_pt

        if c_pt and self._start_canvas_pt:
            if self.active_tool == ToolType.ARROW:
                self._draw_arrow(self._start_canvas_pt, c_pt)
            elif self.active_tool == ToolType.RECT:
                self._draw_rect(self._start_canvas_pt, c_pt)

        self._start_canvas_pt = None
        self._last_canvas_pt = None
        self._current_canvas_pt = None
        self.annotationsChanged.emit(True)
        self.update()

    # ==========================================================================
    # Drawing Primitives on Overlay
    # ==========================================================================

    def _draw_pen_stroke(self, p1: QPoint, p2: QPoint) -> None:
        self._stroke_count += 1
        painter = QPainter(self._overlay_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.pen_color, self.pen_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()

    def _erase_at(self, p: QPoint) -> None:
        painter = QPainter(self._overlay_image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(p, self.eraser_size, self.eraser_size)
        painter.end()

    def _draw_arrow(self, p1: QPoint, p2: QPoint) -> None:
        if (p1 - p2).manhattanLength() < 5:
            return

        self._stroke_count += 1
        painter = QPainter(self._overlay_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.pen_color, self.pen_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(self.pen_color)

        # Draw shaft
        painter.drawLine(p1, p2)

        # Draw arrow head
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        angle = math.atan2(dy, dx)

        arrow_len = max(24, self.pen_size * 3)
        arrow_width = max(16, self.pen_size * 2)

        left_x = p2.x() - arrow_len * math.cos(angle - math.pi / 6)
        left_y = p2.y() - arrow_len * math.sin(angle - math.pi / 6)
        right_x = p2.x() - arrow_len * math.cos(angle + math.pi / 6)
        right_y = p2.y() - arrow_len * math.sin(angle + math.pi / 6)

        arrow_poly = QPolygonF([
            QPoint(p2.x(), p2.y()),
            QPoint(int(left_x), int(left_y)),
            QPoint(int(right_x), int(right_y)),
        ])
        painter.drawPolygon(arrow_poly)
        painter.end()

    def _draw_rect(self, p1: QPoint, p2: QPoint) -> None:
        rect = QRect(p1, p2).normalized()
        if rect.width() < 5 and rect.height() < 5:
            return

        self._stroke_count += 1
        painter = QPainter(self._overlay_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self.pen_color, self.pen_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        painter.end()

    def _handle_text_tool(self, pt: QPoint) -> None:
        text, ok = QInputDialog.getText(
            self, "Livery Annotation Note", "Enter note or design callout to place on livery:"
        )
        if not ok or not text.strip():
            return

        self._save_undo()
        painter = QPainter(self._overlay_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Segoe UI", 48, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_rect = fm.boundingRect(text.strip())

        padding = 24
        box_rect = QRect(
            pt.x(),
            pt.y(),
            text_rect.width() + padding * 2,
            text_rect.height() + padding * 2,
        )

        # Draw dark badge background with colored border
        painter.setPen(QPen(self.pen_color, 6))
        painter.setBrush(QColor(18, 20, 24, 230))
        painter.drawRoundedRect(box_rect, 14, 14)

        # Draw text
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            box_rect.adjusted(padding, padding // 2, -padding, -padding // 2),
            Qt.AlignmentFlag.AlignCenter,
            text.strip(),
        )
        painter.end()

        self.annotationsChanged.emit(True)
        self.update()

    # ==========================================================================
    # Rendering
    # ==========================================================================

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.get_content_rect()

        # If no texture loaded, render grid and placeholder
        if self._base_pixmap is None:
            painter.setPen(QPen(QColor(35, 38, 46), 1, Qt.PenStyle.DotLine))
            step = 30
            for x in range(0, self.width(), step):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), step):
                painter.drawLine(0, y, self.width(), y)

            painter.setPen(QColor(120, 128, 142))
            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            return

        # 1. Base Livery Texture
        painter.drawPixmap(rect, self._base_pixmap)

        # 2. Semi-transparent Wireframe Guide (if enabled)
        if self._show_wireframe and self._wireframe_pixmap is not None:
            painter.setOpacity(0.35)
            painter.drawPixmap(rect, self._wireframe_pixmap)
            painter.setOpacity(1.0)

        # 3. Annotation Layer (if enabled)
        if self._show_annotations:
            painter.drawImage(rect, self._overlay_image)

            # 4. Live interactive preview during drag (Arrow / Rect)
            if self._is_drawing and self._start_canvas_pt and self._current_canvas_pt:
                p1_w = self._canvas_to_widget(self._start_canvas_pt)
                p2_w = self._canvas_to_widget(self._current_canvas_pt)

                if self.active_tool == ToolType.ARROW:
                    pen = QPen(self.pen_color, max(2, self.pen_size // 2), Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(p1_w, p2_w)
                elif self.active_tool == ToolType.RECT:
                    pen = QPen(self.pen_color, max(2, self.pen_size // 2), Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRect(p1_w, p2_w).normalized())

    def _canvas_to_widget(self, pt: QPoint) -> QPoint:
        rect = self.get_content_rect()
        wx = rect.x() + int((pt.x() / self._canvas_size) * rect.width())
        wy = rect.y() + int((pt.y() / self._canvas_size) * rect.height())
        return QPoint(wx, wy)
