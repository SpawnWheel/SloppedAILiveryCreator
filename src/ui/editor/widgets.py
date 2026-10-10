"""Small custom widgets for the Photoshop-style editor UI."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.editor import icons


class ColorSwatch(QPushButton):
    def __init__(self, color: QColor, size: int = 30, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.set_color(color)

    def set_color(self, c: QColor) -> None:
        self.color = QColor(c)
        self.setStyleSheet(
            f"QPushButton {{ background-color: {c.name()}; border: 1px solid #0d0e10; border-radius: 2px; padding: 0; }}"
            f"QPushButton:hover {{ border: 1px solid #ffffff; }}"
        )


class MiniSwatch(QPushButton):
    """Palette chip: click = foreground, right-click = background."""

    leftPicked = Signal(QColor)
    rightPicked = Signal(QColor)

    def __init__(self, c: QColor, size: int = 18, parent=None):
        super().__init__(parent)
        self.c = QColor(c)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{c.name().upper()}\nClick = foreground · Right-click = background")
        self.setStyleSheet(
            f"QPushButton {{ background-color: {c.name()}; border: 1px solid #0d0e10; border-radius: 1px; padding: 0; }}"
            f"QPushButton:hover {{ border: 1px solid #ffffff; }}"
        )
        self.clicked.connect(lambda: self.leftPicked.emit(QColor(self.c)))

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.RightButton:
            self.rightPicked.emit(QColor(self.c))
            return
        super().mousePressEvent(e)


class FgBgWidget(QWidget):
    """Photoshop's overlapping foreground / background colour squares with swap + reset corners."""

    fgClicked = Signal()
    bgClicked = Signal()
    swapClicked = Signal()
    resetClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fg = QColor(230, 30, 40)
        self.bg = QColor(255, 255, 255)
        self.setFixedSize(52, 52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Foreground / background colour\nClick a square to change it · ↔ swaps (X) · "
                        "small squares reset to black/white (D)")

    def set_colors(self, fg: QColor, bg: QColor) -> None:
        self.fg, self.bg = QColor(fg), QColor(bg)
        self.update()

    def _rects(self):
        return QRect(4, 4, 28, 28), QRect(20, 20, 28, 28), QRect(36, 2, 14, 14), QRect(2, 38, 13, 13)

    def paintEvent(self, e) -> None:
        fg_r, bg_r, swap_r, reset_r = self._rects()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for r, c in ((bg_r, self.bg), (fg_r, self.fg)):
            p.setPen(QPen(QColor(10, 10, 12), 1))
            p.setBrush(c)
            p.drawRect(r)
            p.setPen(QPen(QColor(200, 200, 205, 160), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r.adjusted(1, 1, -1, -1))
        # swap arrow
        p.setPen(QPen(QColor(205, 208, 214), 1.4))
        p.drawArc(QRectF(swap_r.x() + 1, swap_r.y() + 3, 18, 18), 90 * 16, 90 * 16)
        p.setBrush(QColor(205, 208, 214))
        p.drawPolygon(QPolygonF([QPointF(swap_r.x() + 10, swap_r.y() + 1), QPointF(swap_r.x() + 14, swap_r.y() + 3.5),
                                 QPointF(swap_r.x() + 10, swap_r.y() + 6)]))
        p.drawPolygon(QPolygonF([QPointF(swap_r.x() - 1, swap_r.y() + 10), QPointF(swap_r.x() + 1.5, swap_r.y() + 14),
                                 QPointF(swap_r.x() + 4, swap_r.y() + 10)]))
        # reset mini squares
        p.setPen(QPen(QColor(10, 10, 12), 1))
        p.setBrush(QColor(255, 255, 255))
        p.drawRect(reset_r.x() + 4, reset_r.y() + 4, 8, 8)
        p.setBrush(QColor(0, 0, 0))
        p.setPen(QPen(QColor(200, 200, 205), 1))
        p.drawRect(reset_r.x(), reset_r.y(), 8, 8)
        p.end()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        fg_r, bg_r, swap_r, reset_r = self._rects()
        pos = e.position().toPoint()
        if swap_r.adjusted(-3, -3, 3, 3).contains(pos):
            self.swapClicked.emit()
        elif reset_r.adjusted(-2, -2, 2, 2).contains(pos):
            self.resetClicked.emit()
        elif fg_r.contains(pos):
            self.fgClicked.emit()
        elif bg_r.contains(pos):
            self.bgClicked.emit()


class SliderSpin(QWidget):
    """Compact slider + spin box pair (option bar / dock rows)."""

    valueChanged = Signal(int)

    def __init__(self, lo: int, hi: int, value: int, suffix: str = "", slider_w: int = 90,
                 spin_hi: Optional[int] = None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setFixedWidth(slider_w)
        self.spin = QSpinBox()
        self.spin.setRange(lo, spin_hi if spin_hi is not None else hi)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(64)
        lay.addWidget(self.slider)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(value)

    def _from_slider(self, v: int) -> None:
        if self.spin.value() != v:
            self.spin.setValue(v)

    def _from_spin(self, v: int) -> None:
        if v <= self.slider.maximum() and self.slider.value() != v:
            self.slider.blockSignals(True)
            self.slider.setValue(v)
            self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self) -> int:
        return self.spin.value()

    def setValue(self, v: int) -> None:
        self.spin.setValue(int(v))


class FloatingToolbox(QFrame):
    """
    The tool palette, floating over the canvas like Photoshop / GIMP. Drag the grip to move it, double-click
    the grip (or the ⇔ button) to switch between one and two columns. It snaps to the canvas edges.
    """

    toolChosen = Signal(str)

    def __init__(self, tools: List[tuple], parent: QWidget):
        super().__init__(parent)
        self.setObjectName("Toolbox")
        self._tools = tools  # [(tool_id, icon_name, tooltip, key, group)]
        self._columns = 1
        self._drag_off: Optional[QPoint] = None
        self._user_moved = False
        self.buttons: Dict[str, QToolButton] = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)

        self.grip = QFrame()
        self.grip.setObjectName("ToolboxGrip")
        self.grip.setFixedHeight(14)
        self.grip.setCursor(Qt.CursorShape.SizeAllCursor)
        self.grip.setToolTip("Drag to move the toolbox · double-click for 1 / 2 columns")
        gl = QHBoxLayout(self.grip)
        gl.setContentsMargins(4, 0, 2, 0)
        dots = QLabel("· · · ·")
        dots.setStyleSheet("color: #6c7079; font-size: 9px; font-weight: bold;")
        gl.addWidget(dots)
        gl.addStretch()
        self.btn_cols = QToolButton()
        self.btn_cols.setText("»")
        self.btn_cols.setFixedSize(14, 12)
        self.btn_cols.setStyleSheet("QToolButton { padding: 0; font-size: 10px; color: #8c9099; border: none; }"
                                    "QToolButton:hover { color: white; }")
        self.btn_cols.setToolTip("One / two columns")
        self.btn_cols.clicked.connect(self.toggle_columns)
        gl.addWidget(self.btn_cols)
        outer.addWidget(self.grip)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(4, 4, 4, 2)
        self.grid.setSpacing(1)
        outer.addWidget(self.grid_host)

        for tool, icon_name, tip, key, _group in tools:
            b = QToolButton()
            b.setIcon(icons.icon(icon_name, 20))
            b.setIconSize(QSize(20, 20))
            b.setFixedSize(32, 30)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, t=tool: self.toolChosen.emit(t))
            self.group.addButton(b)
            self.buttons[tool] = b

        self.fgbg = FgBgWidget()
        self.fgbg_host = QWidget()
        fl = QHBoxLayout(self.fgbg_host)
        fl.setContentsMargins(0, 6, 0, 0)
        fl.addStretch()
        fl.addWidget(self.fgbg)
        fl.addStretch()
        outer.addWidget(self.fgbg_host)
        self._relayout()

    # ------------------------------------------------------------------ layout
    def _relayout(self) -> None:
        while self.grid.count():
            self.grid.takeAt(0)
        cols = self._columns
        row = col = 0
        last_group = None
        for tool, _i, _t, _k, group in self._tools:
            if cols == 1 and last_group is not None and group != last_group:
                sep = QFrame()
                sep.setFixedHeight(1)
                sep.setStyleSheet("background: #3a3d43; margin: 0 4px;")
                self.grid.addWidget(sep, row, 0, 1, 1)
                row += 1
            last_group = group
            self.grid.addWidget(self.buttons[tool], row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1
        self.btn_cols.setText("«" if cols == 2 else "»")
        self.adjustSize()
        self.clamp()

    def toggle_columns(self) -> None:
        self._columns = 2 if self._columns == 1 else 1
        self._relayout()

    def set_checked(self, tool: str) -> None:
        b = self.buttons.get(tool)
        if b is not None and not b.isChecked():
            b.setChecked(True)

    # ------------------------------------------------------------------ dragging
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self.grip.geometry().contains(e.position().toPoint()):
            self._drag_off = e.position().toPoint()
            self.raise_()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag_off is not None:
            new = self.mapToParent(e.position().toPoint()) - self._drag_off
            self.move(new)
            self._user_moved = True
            self.clamp(snap=True)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._drag_off = None
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if self.grip.geometry().contains(e.position().toPoint()):
            self.toggle_columns()
            return
        super().mouseDoubleClickEvent(e)

    def clamp(self, snap: bool = False) -> None:
        par = self.parentWidget()
        if par is None:
            return
        m = 8
        x = min(max(self.x(), m), max(m, par.width() - self.width() - m))
        y = min(max(self.y(), m), max(m, par.height() - self.height() - m))
        if snap:
            if x < 24:
                x = m
            if par.width() - (x + self.width()) < 24:
                x = par.width() - self.width() - m
            if y < 24:
                y = m
        self.move(x, y)

    def default_position(self) -> None:
        self.move(10, 10)
        self._user_moved = False


class CanvasHost(QWidget):
    """Holds the canvas and keeps floating overlays (toolbox, empty-state card) on top of it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas: Optional[QWidget] = None
        self.overlays: List[QWidget] = []
        self.center_overlay: Optional[QWidget] = None

    def set_canvas(self, w: QWidget) -> None:
        self.canvas = w
        w.setParent(self)
        w.lower()

    def relayout(self) -> None:
        if self.canvas is not None:
            self.canvas.setGeometry(self.rect())
        for o in self.overlays:
            if hasattr(o, "clamp"):
                o.clamp()
            o.raise_()
        if self.center_overlay is not None:
            c = self.center_overlay
            c.adjustSize()
            c.move(max(0, (self.width() - c.width()) // 2), max(0, (self.height() - c.height()) // 2))
            c.raise_()

    def resizeEvent(self, e) -> None:
        self.relayout()
        super().resizeEvent(e)
