"""
Livery Paint Editor - a Photoshop-lite window for hand-finishing AI liveries (or wheel rims).

  * Template guide drawn over your paint (opacity / blend adjustable) so you always know where panels are
  * Layers: paint layers, placed PNG decals (sponsors), live text (numbers, names)
  * Tools: move/transform, brush, eraser, line, rectangle, ellipse, polygon, bucket fill
    (fill can use the TEMPLATE outlines as its boundary - click a panel to paint it), eyedropper, text
  * Unlimited-ish undo, zoom / pan, save & reopen layered projects (.ams2paint), export PNG
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.image_io import load_any_image
from src.core.image_processor import ImageProcessor
from src.ui.editor.canvas import EditorCanvas, Tool
from src.ui.editor.document import EditorDocument
from src.ui.editor.layers import (
    BLEND_MODES,
    PROJECT_EXT,
    ImageLayer,
    ItemLayer,
    Layer,
    ProjectData,
    RasterLayer,
    TextLayer,
    _qimage_png_bytes,
    load_project,
    new_image,
    save_project,
)
from src.ui.qt_image import numpy_rgba_to_qimage, pil_to_qimage, qimage_to_numpy_rgba, qimage_to_pil

TOOLS = [
    (Tool.MOVE, "✥", "Move / Transform (V)\nDrag decals & text. Corner = scale (Shift = stretch), "
                     "side dots = stretch one way, blue dot = rotate (Shift snaps 15°). "
                     "On a paint layer, drags its pixels.", "V"),
    (Tool.BRUSH, "🖌", "Brush (B)\nShift+click draws a straight line from the last point. Alt+click picks a colour.", "B"),
    (Tool.ERASER, "⌫", "Eraser (E)", "E"),
    (Tool.LINE, "╱", "Line (L)\nShift snaps to 15° angles.", "L"),
    (Tool.RECT, "▭", "Rectangle (U)\nShift = square.", "U"),
    (Tool.ELLIPSE, "◯", "Ellipse (O)\nShift = circle.", "O"),
    (Tool.POLYGON, "⬠", "Polygon (P)\nClick points; click the first point, double-click, right-click or Enter to fill. "
                         "Backspace removes a point, Esc cancels.", "P"),
    (Tool.FILL, "🪣", "Paint Bucket (G)\nClick inside a template panel to paint that whole area.", "G"),
    (Tool.EYEDROPPER, "💧", "Eyedropper (I)", "I"),
    (Tool.TEXT, "T", "Text (T)\nClick to place a number / name.", "T"),
    (Tool.PAN, "✋", "Pan (H)  -  or hold Space / middle mouse. Mouse wheel zooms.", "H"),
]

PRESET_COLORS = ["#FFFFFF", "#000000", "#1A1A1A", "#7F7F7F", "#E10600", "#FF6A00", "#FFD200",
                 "#00A651", "#00B5E2", "#0047AB", "#6A0DAD", "#FF2D87", "#C0C0C0", "#D4AF37"]

PANEL_QSS = """
QGroupBox { font-weight: bold; font-size: 11px; color: #d0d4dc; border: 1px solid #2e323b;
            border-radius: 6px; margin-top: 10px; padding-top: 8px; background-color: #1f2229; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px;
                   color: #d0d4dc; background-color: #181a1f; }
QLabel { color: #b8beca; font-size: 11px; }
QPushButton { background-color: #2b303c; color: #e6e9ef; border: 1px solid #3c404b; border-radius: 4px;
              padding: 4px 8px; font-size: 11px; }
QPushButton:hover { background-color: #3b4252; border-color: #0078d4; }
QPushButton:disabled { color: #555c68; background-color: #1d2027; }
QToolButton { background-color: #23262e; color: #e6e9ef; border: 1px solid #343844; border-radius: 6px;
              font-size: 17px; }
QToolButton:hover { border-color: #0078d4; }
QToolButton:checked { background-color: #0078d4; border-color: #33a1ff; color: #ffffff; }
QListWidget { background-color: #16181d; border: 1px solid #2e323b; border-radius: 4px; color: #e0e0e0; }
QListWidget::item { padding: 3px; }
QListWidget::item:selected { background-color: #0a4f8a; }
QSpinBox, QDoubleSpinBox, QComboBox, QFontComboBox, QPlainTextEdit {
    background-color: #16181d; border: 1px solid #3c404b; border-radius: 3px; color: #ffffff; padding: 2px 4px; }
QCheckBox { color: #b8beca; font-size: 11px; }
"""


class ColorSwatch(QPushButton):
    def __init__(self, color: QColor, size: int = 30, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.set_color(color)

    def set_color(self, c: QColor) -> None:
        self.color = QColor(c)
        self.setStyleSheet(
            f"background-color: {c.name()}; border: 2px solid #9aa3b2; border-radius: 4px;"
        )


class LiveryPaintEditor(QDialog):
    """Layered paint editor. Call exec(); on Accepted use result_image() and project_data()."""

    applied = Signal(object)  # PIL image

    def __init__(
        self,
        base_image: Image.Image,
        guide_image: Optional[Image.Image] = None,
        title: str = "Livery Paint Editor",
        project: Optional[ProjectData] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinMaxButtonsHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setStyleSheet("QDialog { background-color: #181a1f; }" + PANEL_QSS)
        self.resize(1500, 950)
        self._result: Optional[Image.Image] = None
        self._palette_source = base_image
        self._updating_ui = False
        self._slider_before = None

        if project is not None:
            w, h = project.width, project.height
        else:
            w, h = base_image.size
        self.doc = EditorDocument(w, h, self)

        if project is not None:
            self.doc.replace_all(project.layers, len(project.layers) - 1)
            if project.guide_png:
                gi = QImage()
                gi.loadFromData(project.guide_png, "PNG")
                self.doc.set_guide(gi)
        else:
            base = RasterLayer("Base Livery", pil_to_qimage(base_image))
            paint = RasterLayer("Paint 1", new_image(w, h))
            self.doc.replace_all([base, paint], 1)
        if guide_image is not None:
            self.doc.set_guide(pil_to_qimage(guide_image))

        self._build_ui()
        self._install_shortcuts()
        self.doc.layersChanged.connect(self._refresh_layers)
        self.doc.activeChanged.connect(lambda _: self._on_doc_active_changed())
        self.doc.changed.connect(self._refresh_properties_live)
        self.doc.undo.on_change = self._refresh_undo_buttons
        self._refresh_layers()
        self._on_doc_active_changed()
        self._refresh_undo_buttons()
        self._set_tool(Tool.BRUSH)
        self.doc.dirty = False

    # ================================================================== UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addLayout(self._build_top_bar())

        mid = QHBoxLayout()
        mid.setSpacing(6)
        mid.addWidget(self._build_tool_palette())

        self.canvas = EditorCanvas(self.doc, self)
        self.canvas.statusMessage.connect(self._flash)
        self.canvas.colorPicked.connect(lambda c: self._set_primary(c))
        self.canvas.textRequested.connect(self._on_text_requested)
        self.canvas.needPaintLayer.connect(self._auto_paint_layer)
        self.canvas.zoomChanged.connect(lambda z: self.lbl_zoom.setText(f"{z * 100:.0f}%"))
        self.canvas.cursorMoved.connect(lambda x, y: self.lbl_pos.setText(f"x {x}  y {y}"))
        self.canvas.filesDropped.connect(self._on_files_dropped)
        frame = QFrame()
        frame.setStyleSheet("QFrame { border: 1px solid #2d313b; border-radius: 6px; }")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(1, 1, 1, 1)
        fl.addWidget(self.canvas)
        mid.addWidget(frame, 1)

        mid.addWidget(self._build_side_panel())
        root.addLayout(mid, 1)

        # bottom bar
        bottom = QHBoxLayout()
        self.lbl_status = QLabel("Tip: pick the Paint Bucket and click inside a template panel to paint it.")
        self.lbl_status.setStyleSheet("color: #8fa3bf; font-size: 11px;")
        bottom.addWidget(self.lbl_status, 1)
        self.lbl_pos = QLabel("")
        self.lbl_pos.setStyleSheet("color: #6c7486; font-size: 11px; min-width: 110px;")
        bottom.addWidget(self.lbl_pos)
        self.lbl_zoom = QLabel("")
        self.lbl_zoom.setStyleSheet("color: #6c7486; font-size: 11px; min-width: 50px;")
        bottom.addWidget(self.lbl_zoom)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_apply = QPushButton("✅ Apply to Livery")
        btn_apply.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border: none;"
            " padding: 7px 18px; border-radius: 4px; font-size: 12px; } QPushButton:hover { background-color: #388e3c; }"
        )
        btn_apply.setToolTip("Flatten all visible layers (the template guide is NOT included) and add it as a new version")
        btn_apply.clicked.connect(self._apply)
        bottom.addWidget(btn_apply)
        root.addLayout(bottom)

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(4)

        def btn(text, tip, slot):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        btn("📂 Open Project", "Open a saved layered project (.ams2paint)", self._open_project)
        btn("💾 Save Project", "Save all layers so you can come back and move sponsors later (Ctrl+S)", self._save_project)
        bar.addSpacing(10)
        btn("🖼 Add Decal / PNG…", "Place a sponsor logo or any image as a movable layer (also: drag & drop, Ctrl+V)",
            self._import_decal)
        btn("🔤 Add Text…", "Add a race number or name as editable text", lambda: self._add_text(None))
        btn("➕ Paint Layer", "New empty paint layer above the current one", self._new_paint_layer)
        bar.addSpacing(10)
        self.btn_undo = btn("↩ Undo", "Undo (Ctrl+Z)", self._undo)
        self.btn_redo = btn("↪ Redo", "Redo (Ctrl+Y)", self._redo)
        bar.addSpacing(10)
        btn("⤢ Fit", "Fit to window (Ctrl+0)", lambda: self.canvas.fit())
        btn("1:1", "Actual pixels (Ctrl+1)", lambda: self.canvas.set_zoom(1.0))
        btn("＋", "Zoom in", lambda: self.canvas.set_zoom(self.canvas.scale * 1.25))
        btn("－", "Zoom out", lambda: self.canvas.set_zoom(self.canvas.scale / 1.25))
        bar.addStretch()
        btn("⬇ Export PNG…", "Save the flattened texture as a PNG", self._export_png)
        return bar

    def _build_tool_palette(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(52)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        for tool, icon, tip, _ in TOOLS:
            b = QToolButton()
            b.setText(icon)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setFixedSize(46, 40)
            b.clicked.connect(lambda _=False, t=tool: self._set_tool(t))
            self.tool_group.addButton(b)
            self.tool_buttons[tool] = b
            lay.addWidget(b)
        lay.addSpacing(10)

        self.sw_primary = ColorSwatch(QColor(230, 30, 40), 34)
        self.sw_primary.setToolTip("Primary colour (paint / fill). Click to change.")
        self.sw_primary.clicked.connect(lambda: self._choose_color(True))
        self.sw_secondary = ColorSwatch(QColor(255, 255, 255), 26)
        self.sw_secondary.setToolTip("Secondary colour (shape outlines). Click to change.")
        self.sw_secondary.clicked.connect(lambda: self._choose_color(False))
        lay.addWidget(self.sw_primary, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.sw_secondary, 0, Qt.AlignmentFlag.AlignHCenter)
        swap = QPushButton("⇅")
        swap.setToolTip("Swap colours (X)")
        swap.setFixedSize(46, 22)
        swap.clicked.connect(self._swap_colors)
        lay.addWidget(swap)
        lay.addStretch()
        return w

    def _build_side_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(372)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 14, 0)
        lay.setSpacing(6)
        lay.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

        lay.addWidget(self._build_colors_group())
        lay.addWidget(self._build_tool_options_group())
        lay.addWidget(self._build_guide_group())
        lay.addWidget(self._build_layers_group(), 1)
        lay.addWidget(self._build_props_group())
        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _build_colors_group(self) -> QGroupBox:
        g = QGroupBox("Colours")
        v = QVBoxLayout(g)
        v.setSpacing(4)
        v.addWidget(QLabel("From this livery:"))
        self.palette_row = QHBoxLayout()
        self.palette_row.setSpacing(3)
        try:
            cols = ImageProcessor.dominant_colors(self._palette_source, 8)
        except Exception:
            cols = []
        for c in cols:
            self.palette_row.addWidget(self._mini_swatch(QColor(*c)))
        self.palette_row.addStretch()
        v.addLayout(self.palette_row)
        grid = QGridLayout()
        grid.setSpacing(3)
        for i, hexc in enumerate(PRESET_COLORS):
            grid.addWidget(self._mini_swatch(QColor(hexc)), i // 7, i % 7)
        v.addLayout(grid)
        return g

    def _mini_swatch(self, c: QColor) -> QPushButton:
        b = QPushButton()
        b.setFixedSize(22, 22)
        b.setToolTip(f"{c.name().upper()}  (click = primary, right-click = secondary)")
        b.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555; border-radius: 3px; padding: 0;")
        b.clicked.connect(lambda _=False, col=QColor(c): self._set_primary(col))
        b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        b.customContextMenuRequested.connect(lambda _p, col=QColor(c): self._set_secondary(col))
        return b

    def _build_tool_options_group(self) -> QGroupBox:
        g = QGroupBox("Tool Options")
        f = QFormLayout(g)
        f.setSpacing(5)

        size_row = QHBoxLayout()
        self.sl_size = QSlider(Qt.Orientation.Horizontal)
        self.sl_size.setRange(1, 600)
        self.sl_size.setValue(24)
        self.sp_size = QSpinBox()
        self.sp_size.setRange(1, 2000)
        self.sp_size.setValue(24)
        self.sp_size.setSuffix(" px")
        self.sl_size.valueChanged.connect(self.sp_size.setValue)
        self.sp_size.valueChanged.connect(self._on_size_changed)
        size_row.addWidget(self.sl_size, 1)
        size_row.addWidget(self.sp_size)
        f.addRow("Size:", size_row)

        op_row = QHBoxLayout()
        self.sl_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sl_opacity.setRange(1, 100)
        self.sl_opacity.setValue(100)
        self.lbl_opacity = QLabel("100%")
        self.sl_opacity.valueChanged.connect(self._on_brush_opacity)
        op_row.addWidget(self.sl_opacity, 1)
        op_row.addWidget(self.lbl_opacity)
        f.addRow("Opacity:", op_row)

        shape_row = QHBoxLayout()
        self.chk_fill = QCheckBox("Fill (primary)")
        self.chk_fill.setChecked(True)
        self.chk_outline = QCheckBox("Outline (secondary)")
        self.chk_fill.toggled.connect(lambda v: setattr(self.canvas, "shape_fill", v))
        self.chk_outline.toggled.connect(lambda v: setattr(self.canvas, "shape_outline", v))
        shape_row.addWidget(self.chk_fill)
        shape_row.addWidget(self.chk_outline)
        self.shape_row_w = QWidget()
        self.shape_row_w.setLayout(shape_row)
        f.addRow("Shapes:", self.shape_row_w)

        self.cmb_fill_sample = QComboBox()
        self.cmb_fill_sample.addItem("Template outlines", "guide")
        self.cmb_fill_sample.addItem("Current layer", "layer")
        self.cmb_fill_sample.addItem("Everything visible", "all")
        self.cmb_fill_sample.setToolTip("What the bucket looks at to find the area's edges")
        self.cmb_fill_sample.currentIndexChanged.connect(
            lambda _: setattr(self.canvas, "fill_sample", self.cmb_fill_sample.currentData())
        )
        f.addRow("Fill uses:", self.cmb_fill_sample)

        tol_row = QHBoxLayout()
        self.sp_tol = QSpinBox()
        self.sp_tol.setRange(0, 255)
        self.sp_tol.setValue(40)
        self.sp_tol.setToolTip("How different a colour can be and still be filled")
        self.sp_tol.valueChanged.connect(lambda v: setattr(self.canvas, "fill_tolerance", v))
        self.sp_grow = QSpinBox()
        self.sp_grow.setRange(0, 20)
        self.sp_grow.setValue(2)
        self.sp_grow.setSuffix(" px")
        self.sp_grow.setToolTip("Expand the fill so it slides under the template's lines (no gaps)")
        self.sp_grow.valueChanged.connect(lambda v: setattr(self.canvas, "fill_grow", v))
        tol_row.addWidget(QLabel("Tol"))
        tol_row.addWidget(self.sp_tol)
        tol_row.addWidget(QLabel("Grow"))
        tol_row.addWidget(self.sp_grow)
        self.fill_row_w = QWidget()
        self.fill_row_w.setLayout(tol_row)
        f.addRow("", self.fill_row_w)
        self.chk_contig = QCheckBox("Contiguous (only the connected area)")
        self.chk_contig.setChecked(True)
        self.chk_contig.toggled.connect(lambda v: setattr(self.canvas, "fill_contiguous", v))
        f.addRow("", self.chk_contig)
        self._fill_widgets = [self.cmb_fill_sample, self.fill_row_w, self.chk_contig]
        return g

    def _build_guide_group(self) -> QGroupBox:
        g = QGroupBox("Template Guide (not exported)")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.chk_guide = QCheckBox("Show")
        self.chk_guide.setChecked(True)
        self.chk_guide.toggled.connect(self._on_guide_toggle)
        row.addWidget(self.chk_guide)
        self.cmb_guide_blend = QComboBox()
        self.cmb_guide_blend.addItems(["Multiply", "Normal", "Screen", "Overlay", "Darken", "Lighten"])
        self.cmb_guide_blend.setToolTip("Multiply suits dark lines on white. Use Screen for light lines on black.")
        self.cmb_guide_blend.currentTextChanged.connect(self._on_guide_blend)
        row.addWidget(self.cmb_guide_blend)
        v.addLayout(row)
        op = QHBoxLayout()
        op.addWidget(QLabel("Opacity"))
        self.sl_guide = QSlider(Qt.Orientation.Horizontal)
        self.sl_guide.setRange(0, 100)
        self.sl_guide.setValue(int(self.doc.guide_opacity * 100))
        self.sl_guide.valueChanged.connect(self._on_guide_opacity)
        op.addWidget(self.sl_guide, 1)
        v.addLayout(op)
        btn = QPushButton("Load / change template image…")
        btn.clicked.connect(self._load_guide)
        v.addWidget(btn)
        self.lbl_guide = QLabel("")
        self.lbl_guide.setWordWrap(True)
        v.addWidget(self.lbl_guide)
        self._update_guide_label()
        return g

    def _build_layers_group(self) -> QGroupBox:
        g = QGroupBox("Layers")
        v = QVBoxLayout(g)
        v.setSpacing(4)
        self.list_layers = QListWidget()
        self.list_layers.setIconSize(QSize(40, 40))
        self.list_layers.setMinimumHeight(190)
        self.list_layers.currentRowChanged.connect(self._on_layer_row_changed)
        self.list_layers.itemChanged.connect(self._on_layer_item_changed)
        v.addWidget(self.list_layers, 1)

        r1 = QHBoxLayout()
        for text, tip, slot in [
            ("▲", "Move layer up", lambda: self.doc.move_layer(self.doc.active_index, +1)),
            ("▼", "Move layer down", lambda: self.doc.move_layer(self.doc.active_index, -1)),
            ("⧉", "Duplicate (Ctrl+J)", lambda: self.doc.duplicate_layer(self.doc.active_index)),
            ("⤓", "Merge down", lambda: self.doc.merge_down(self.doc.active_index)),
            ("🗑", "Delete layer (Del)", lambda: self.doc.remove_layer(self.doc.active_index)),
        ]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            r1.addWidget(b)
        v.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Opacity"))
        self.sl_layer_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sl_layer_opacity.setRange(0, 100)
        self.sl_layer_opacity.sliderPressed.connect(self._layer_slider_pressed)
        self.sl_layer_opacity.sliderReleased.connect(self._layer_slider_released)
        self.sl_layer_opacity.valueChanged.connect(self._layer_opacity_changed)
        r2.addWidget(self.sl_layer_opacity, 1)
        self.cmb_layer_blend = QComboBox()
        self.cmb_layer_blend.addItems(list(BLEND_MODES.keys()))
        self.cmb_layer_blend.currentTextChanged.connect(self._layer_blend_changed)
        r2.addWidget(self.cmb_layer_blend)
        v.addLayout(r2)
        self.chk_lock = QCheckBox("Lock layer")
        self.chk_lock.toggled.connect(self._layer_lock_changed)
        v.addWidget(self.chk_lock)
        return g

    def _build_props_group(self) -> QGroupBox:
        self.props_group = QGroupBox("Decal / Text Properties")
        f = QFormLayout(self.props_group)
        f.setSpacing(4)

        def dspin(lo, hi, dec, suffix="", step=1.0):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            return s

        pos = QHBoxLayout()
        self.sp_x = dspin(-20000, 20000, 0)
        self.sp_y = dspin(-20000, 20000, 0)
        pos.addWidget(QLabel("X"))
        pos.addWidget(self.sp_x)
        pos.addWidget(QLabel("Y"))
        pos.addWidget(self.sp_y)
        f.addRow("Position:", pos)

        sc = QHBoxLayout()
        self.sp_sx = dspin(1, 5000, 1, "%", 5)
        self.sp_sy = dspin(1, 5000, 1, "%", 5)
        self.chk_link = QCheckBox("🔗")
        self.chk_link.setChecked(True)
        self.chk_link.setToolTip("Keep proportions")
        sc.addWidget(self.sp_sx)
        sc.addWidget(self.sp_sy)
        sc.addWidget(self.chk_link)
        f.addRow("Scale W/H:", sc)

        rot = QHBoxLayout()
        self.sp_rot = dspin(-360, 360, 1, "°", 1)
        rot.addWidget(self.sp_rot)
        for text, tip, slot in [
            ("⟲90", "Rotate -90°", lambda: self._rotate_by(-90)),
            ("⟳90", "Rotate +90°", lambda: self._rotate_by(90)),
            ("⇋", "Flip horizontally (mirror for the other side of the car)", lambda: self._toggle_flip("flip_h")),
            ("⇵", "Flip vertically", lambda: self._toggle_flip("flip_v")),
        ]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedWidth(40)
            b.clicked.connect(slot)
            rot.addWidget(b)
        f.addRow("Rotate:", rot)

        for s in (self.sp_x, self.sp_y, self.sp_sx, self.sp_sy, self.sp_rot):
            s.valueChanged.connect(self._on_transform_spin)

        # decal-only
        self.decal_row = QWidget()
        dr = QGridLayout(self.decal_row)
        dr.setContentsMargins(0, 0, 0, 0)
        b1 = QPushButton("Remove white bg")
        b1.setToolTip("Make white areas transparent (for JPG logos)")
        b1.clicked.connect(lambda: self._key_out((255, 255, 255)))
        b2 = QPushButton("Remove black bg")
        b2.clicked.connect(lambda: self._key_out((0, 0, 0)))
        b3 = QPushButton("Colourise (primary)")
        b3.setToolTip("Turn the logo into a single colour (your primary colour), keeping its shape")
        b3.clicked.connect(self._colorize_decal)
        b4 = QPushButton("Rasterize")
        b4.setToolTip("Convert into a normal paint layer so you can paint / erase on it")
        b4.clicked.connect(lambda: self.doc.rasterize_layer(self.doc.active_index))
        dr.addWidget(b1, 0, 0)
        dr.addWidget(b2, 0, 1)
        dr.addWidget(b3, 1, 0)
        dr.addWidget(b4, 1, 1)
        f.addRow(self.decal_row)

        # text-only
        self.text_box = QWidget()
        tf = QFormLayout(self.text_box)
        tf.setContentsMargins(0, 0, 0, 0)
        self.ed_text = QPlainTextEdit()
        self.ed_text.setFixedHeight(48)
        self._text_timer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(350)
        self._text_timer.timeout.connect(lambda: self._apply_text_prop("text", self.ed_text.toPlainText()))
        self.ed_text.textChanged.connect(self._text_timer.start)
        tf.addRow("Text:", self.ed_text)
        self.cmb_font = QFontComboBox()
        self.cmb_font.currentFontChanged.connect(lambda f_: self._apply_text_prop("font_family", f_.family()))
        tf.addRow("Font:", self.cmb_font)
        fs = QHBoxLayout()
        self.sp_font = QSpinBox()
        self.sp_font.setRange(6, 3000)
        self.sp_font.setSuffix(" px")
        self.sp_font.setKeyboardTracking(False)
        self.sp_font.valueChanged.connect(lambda v: self._apply_text_prop("font_size", v))
        self.chk_bold = QCheckBox("B")
        self.chk_bold.toggled.connect(lambda v: self._apply_text_prop("bold", v))
        self.chk_italic = QCheckBox("I")
        self.chk_italic.toggled.connect(lambda v: self._apply_text_prop("italic", v))
        fs.addWidget(self.sp_font)
        fs.addWidget(self.chk_bold)
        fs.addWidget(self.chk_italic)
        tf.addRow("Size:", fs)
        cl = QHBoxLayout()
        self.sw_text = ColorSwatch(QColor("white"), 22)
        self.sw_text.clicked.connect(lambda: self._pick_text_color("color"))
        self.sw_outline = ColorSwatch(QColor("black"), 22)
        self.sw_outline.clicked.connect(lambda: self._pick_text_color("outline_color"))
        self.sp_outline = QSpinBox()
        self.sp_outline.setRange(0, 200)
        self.sp_outline.setSuffix(" px")
        self.sp_outline.setKeyboardTracking(False)
        self.sp_outline.valueChanged.connect(lambda v: self._apply_text_prop("outline_width", v))
        cl.addWidget(QLabel("Fill"))
        cl.addWidget(self.sw_text)
        cl.addWidget(QLabel("Outline"))
        cl.addWidget(self.sw_outline)
        cl.addWidget(self.sp_outline)
        cl.addStretch()
        tf.addRow("Colour:", cl)
        f.addRow(self.text_box)
        return self.props_group

    # ================================================================== shortcuts
    def _install_shortcuts(self) -> None:
        for tool, _, _, key in TOOLS:
            QShortcut(QKeySequence(key), self, activated=lambda t=tool: self._set_tool(t))
        pairs = [
            (QKeySequence.StandardKey.Undo, self._undo),
            (QKeySequence("Ctrl+Y"), self._redo),
            (QKeySequence("Ctrl+Shift+Z"), self._redo),
            (QKeySequence("Ctrl+0"), lambda: self.canvas.fit()),
            (QKeySequence("Ctrl+1"), lambda: self.canvas.set_zoom(1.0)),
            (QKeySequence("Ctrl+S"), self._save_project),
            (QKeySequence("Ctrl+J"), lambda: self.doc.duplicate_layer(self.doc.active_index)),
            (QKeySequence.StandardKey.Paste, self._paste_image),
            (QKeySequence("X"), self._swap_colors),
            (QKeySequence("["), lambda: self.sp_size.setValue(max(1, int(self.sp_size.value() / 1.2)))),
            (QKeySequence("]"), lambda: self.sp_size.setValue(int(self.sp_size.value() * 1.2) + 1)),
            (QKeySequence("Delete"), lambda: self.doc.remove_layer(self.doc.active_index)),
        ]
        for seq, fn in pairs:
            QShortcut(seq, self, activated=fn)

    # ================================================================== tool / colour handlers
    def _set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        b = self.tool_buttons.get(tool)
        if b and not b.isChecked():
            b.setChecked(True)
        is_fill = tool == Tool.FILL
        for w in self._fill_widgets:
            w.setEnabled(is_fill)
        self.shape_row_w.setEnabled(tool in (Tool.RECT, Tool.ELLIPSE, Tool.POLYGON))
        hints = {
            Tool.FILL: "Paint Bucket: click inside a panel. 'Template outlines' uses the guide's lines as walls.",
            Tool.MOVE: "Move: drag decals/text; handles scale & rotate. Arrow keys nudge (Shift = 10px).",
            Tool.POLYGON: "Polygon: click corners, then click the first point / double-click / Enter to paint it.",
            Tool.BRUSH: "Brush: Shift+click for straight lines. [ and ] change size. Alt+click picks colour.",
            Tool.TEXT: "Text: click where the number / name should go.",
        }
        if tool in hints:
            self.lbl_status.setText(hints[tool])

    def _on_size_changed(self, v: int) -> None:
        self.canvas.brush_size = v
        if self.sl_size.value() != v and v <= self.sl_size.maximum():
            self.sl_size.blockSignals(True)
            self.sl_size.setValue(v)
            self.sl_size.blockSignals(False)
        self.canvas._update_cursor()
        self.canvas.update()

    def _on_brush_opacity(self, v: int) -> None:
        self.canvas.brush_opacity = v / 100.0
        self.lbl_opacity.setText(f"{v}%")

    def _set_primary(self, c: QColor) -> None:
        self.canvas.primary = QColor(c)
        self.sw_primary.set_color(c)

    def _set_secondary(self, c: QColor) -> None:
        self.canvas.secondary = QColor(c)
        self.sw_secondary.set_color(c)

    def _choose_color(self, primary: bool) -> None:
        cur = self.canvas.primary if primary else self.canvas.secondary
        c = QColorDialog.getColor(cur, self, "Choose colour")
        if c.isValid():
            (self._set_primary if primary else self._set_secondary)(c)

    def _swap_colors(self) -> None:
        p, s = QColor(self.canvas.primary), QColor(self.canvas.secondary)
        self._set_primary(s)
        self._set_secondary(p)

    def _flash(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    # ================================================================== guide
    def _on_guide_toggle(self, v: bool) -> None:
        self.doc.guide_visible = v
        self.canvas.update()

    def _on_guide_opacity(self, v: int) -> None:
        self.doc.guide_opacity = v / 100.0
        self.canvas.update()

    def _on_guide_blend(self, name: str) -> None:
        self.doc.guide_blend = name
        self.canvas.update()

    def _load_guide(self) -> None:
        from src.utils.constants import IMAGE_OPEN_FILTER

        path, _ = QFileDialog.getOpenFileName(self, "Choose template / wireframe", "", IMAGE_OPEN_FILTER + ";;ZIP (*.zip)")
        if not path:
            return
        try:
            img = load_any_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Template", f"Could not open template:\n{e}")
            return
        self.doc.set_guide(pil_to_qimage(img))
        self.chk_guide.setChecked(True)
        self._update_guide_label()

    def _update_guide_label(self) -> None:
        if self.doc.guide is None:
            self.lbl_guide.setText("No template loaded. Load your car's template (PNG / PSD / DDS) to see the panel "
                                   "outlines while you paint, and to use the bucket's 'Template outlines' mode.")
            self.lbl_guide.setStyleSheet("color: #ffb74d; font-size: 10px;")
        else:
            self.lbl_guide.setText("Template loaded ✓  - shown on top of your paint only here, never exported.")
            self.lbl_guide.setStyleSheet("color: #81c784; font-size: 10px;")

    # ================================================================== layers panel
    def _layer_icon(self, layer: Layer) -> QIcon:
        try:
            if isinstance(layer, RasterLayer):
                src = layer.image
            else:
                src = layer.source()  # type: ignore[attr-defined]
            pm = QPixmap(40, 40)
            pm.fill(QColor(60, 62, 68))
            from PySide6.QtGui import QPainter

            p = QPainter(pm)
            thumb = src.scaled(40, 40, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
            p.drawImage((40 - thumb.width()) // 2, (40 - thumb.height()) // 2, thumb)
            p.end()
            return QIcon(pm)
        except Exception:
            return QIcon()

    def _refresh_layers(self) -> None:
        self._updating_ui = True
        self.list_layers.blockSignals(True)
        self.list_layers.clear()
        kinds = {"raster": "🖌", "image": "🖼", "text": "🔤"}
        for i in range(len(self.doc.layers) - 1, -1, -1):
            layer = self.doc.layers[i]
            item = QListWidgetItem(self._layer_icon(layer), f"{kinds.get(layer.kind, '')} {layer.name}"
                                   + ("  🔒" if layer.locked else ""))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip("Tick = visible. Double-click to rename.")
            self.list_layers.addItem(item)
        self.list_layers.setCurrentRow(len(self.doc.layers) - 1 - self.doc.active_index)
        self.list_layers.blockSignals(False)
        self._updating_ui = False
        self._sync_layer_controls()

    def _on_layer_row_changed(self, row: int) -> None:
        if self._updating_ui or row < 0:
            return
        idx = len(self.doc.layers) - 1 - row
        self.doc.set_active(idx)

    def _on_layer_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating_ui:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or not (0 <= idx < len(self.doc.layers)):
            return
        layer = self.doc.layers[idx]
        before = {"visible": layer.visible, "name": layer.name}
        name = item.text()
        for prefix in ("🖌 ", "🖼 ", "🔤 "):
            if name.startswith(prefix):
                name = name[len(prefix):]
        name = name.replace("  🔒", "").strip() or layer.name
        after = {"visible": item.checkState() == Qt.CheckState.Checked, "name": name}
        QTimer.singleShot(0, lambda: self.doc.set_layer_props(layer, before, after))

    def _on_doc_active_changed(self) -> None:
        if not self._updating_ui:
            self._updating_ui = True
            self.list_layers.blockSignals(True)
            self.list_layers.setCurrentRow(len(self.doc.layers) - 1 - self.doc.active_index)
            self.list_layers.blockSignals(False)
            self._updating_ui = False
        self._sync_layer_controls()

    def _sync_layer_controls(self) -> None:
        layer = self.doc.active
        self._updating_ui = True
        if layer is not None:
            self.sl_layer_opacity.setValue(int(layer.opacity * 100))
            self.cmb_layer_blend.setCurrentText(layer.blend)
            self.chk_lock.setChecked(layer.locked)
        self._updating_ui = False
        self._refresh_properties()

    def _layer_slider_pressed(self) -> None:
        if self.doc.active is not None:
            self._slider_before = self.doc.active.props()

    def _layer_opacity_changed(self, v: int) -> None:
        layer = self.doc.active
        if self._updating_ui or layer is None:
            return
        if self._slider_before is None:  # keyboard / click on groove
            before = layer.props()
            after = dict(before, opacity=v / 100.0)
            self.doc.set_layer_props(layer, before, after, "Layer opacity")
        else:
            layer.opacity = v / 100.0
            self.doc.invalidate()

    def _layer_slider_released(self) -> None:
        layer = self.doc.active
        if layer is not None and self._slider_before is not None:
            before = self._slider_before
            self._slider_before = None
            self.doc.set_layer_props(layer, before, layer.props(), "Layer opacity")

    def _layer_blend_changed(self, name: str) -> None:
        layer = self.doc.active
        if self._updating_ui or layer is None or layer.blend == name:
            return
        before = layer.props()
        self.doc.set_layer_props(layer, before, dict(before, blend=name), "Blend mode")

    def _layer_lock_changed(self, v: bool) -> None:
        layer = self.doc.active
        if self._updating_ui or layer is None:
            return
        before = layer.props()
        self.doc.set_layer_props(layer, before, dict(before, locked=v), "Lock layer")

    # ================================================================== properties panel
    def _refresh_properties_live(self) -> None:
        if isinstance(self.doc.active, ItemLayer) and self.props_group.isVisible():
            self._refresh_properties()

    def _refresh_properties(self) -> None:
        layer = self.doc.active
        is_item = isinstance(layer, ItemLayer)
        self.props_group.setVisible(is_item)
        if not is_item:
            return
        self._updating_ui = True
        for sp, val in (
            (self.sp_x, layer.x), (self.sp_y, layer.y),
            (self.sp_sx, abs(layer.sx) * 100), (self.sp_sy, abs(layer.sy) * 100),
            (self.sp_rot, ((layer.rotation + 180) % 360) - 180),
        ):
            if not sp.hasFocus():
                sp.setValue(val)
        is_text = isinstance(layer, TextLayer)
        self.text_box.setVisible(is_text)
        self.decal_row.setVisible(isinstance(layer, ImageLayer))
        if is_text:
            if not self.ed_text.hasFocus() and self.ed_text.toPlainText() != layer.text:
                self.ed_text.setPlainText(layer.text)
            self.cmb_font.setCurrentFont(QFont(layer.font_family))
            if not self.sp_font.hasFocus():
                self.sp_font.setValue(int(layer.font_size))
            self.chk_bold.setChecked(layer.bold)
            self.chk_italic.setChecked(layer.italic)
            self.sw_text.set_color(layer.color)
            self.sw_outline.set_color(layer.outline_color)
            if not self.sp_outline.hasFocus():
                self.sp_outline.setValue(int(layer.outline_width))
        self._updating_ui = False

    def _on_transform_spin(self, _v=None) -> None:
        layer = self.doc.active
        if self._updating_ui or not isinstance(layer, ItemLayer):
            return
        before = layer.transform_state()
        sx, sy = self.sp_sx.value() / 100.0, self.sp_sy.value() / 100.0
        if self.chk_link.isChecked():
            sender = self.sender()
            ratio = (abs(before["sy"]) / abs(before["sx"])) if before["sx"] else 1.0
            if sender is self.sp_sx:
                sy = sx * ratio
            elif sender is self.sp_sy:
                sx = sy / ratio if ratio else sy
        after = dict(before, x=self.sp_x.value(), y=self.sp_y.value(), sx=sx, sy=sy, rotation=self.sp_rot.value())
        self.doc.commit_transform(layer, before, after)
        self._refresh_properties()

    def _rotate_by(self, deg: float) -> None:
        layer = self.doc.active
        if isinstance(layer, ItemLayer):
            before = layer.transform_state()
            self.doc.commit_transform(layer, before, dict(before, rotation=before["rotation"] + deg))

    def _toggle_flip(self, key: str) -> None:
        layer = self.doc.active
        if isinstance(layer, ItemLayer):
            before = layer.transform_state()
            self.doc.commit_transform(layer, before, dict(before, **{key: not before[key]}))

    def _apply_text_prop(self, key: str, value) -> None:
        layer = self.doc.active
        if self._updating_ui or not isinstance(layer, TextLayer):
            return
        before = layer.props()
        if before.get(key) == value:
            return
        after = dict(before, **{key: value})
        if key == "text":
            after["name"] = (value.splitlines()[0] if value else "Text")[:24]
        self.doc.set_layer_props(layer, before, after, "Edit text")

    def _pick_text_color(self, key: str) -> None:
        layer = self.doc.active
        if not isinstance(layer, TextLayer):
            return
        cur = layer.color if key == "color" else layer.outline_color
        c = QColorDialog.getColor(cur, self, "Text colour")
        if c.isValid():
            self._apply_text_prop(key, c.name(QColor.NameFormat.HexArgb))
            self._refresh_properties()

    def _key_out(self, rgb) -> None:
        layer = self.doc.active
        if not isinstance(layer, ImageLayer):
            return
        arr = qimage_to_numpy_rgba(layer.image).astype(np.int16)
        dist = np.abs(arr[..., :3] - np.array(rgb, np.int16)).max(axis=2)
        # soft key: fully transparent within 18, fade out to 60
        alpha_factor = np.clip((dist - 18) / 42.0, 0, 1)
        arr[..., 3] = (arr[..., 3] * alpha_factor).astype(np.int16)
        self._replace_decal_image(layer, numpy_rgba_to_qimage(arr.astype(np.uint8)), "Remove background")

    def _colorize_decal(self) -> None:
        layer = self.doc.active
        if not isinstance(layer, ImageLayer):
            return
        arr = qimage_to_numpy_rgba(layer.image)
        c = self.canvas.primary
        arr[..., 0], arr[..., 1], arr[..., 2] = c.red(), c.green(), c.blue()
        self._replace_decal_image(layer, numpy_rgba_to_qimage(arr), "Colourise decal")

    def _replace_decal_image(self, layer: ImageLayer, new_img: QImage, label: str) -> None:
        from src.ui.editor.undo import UndoCommand

        old = layer.image

        def put(img):
            layer.image = img
            self.doc.notify_props_changed()

        put(new_img)
        self.doc.undo.push(UndoCommand(label, lambda: put(old), lambda: put(new_img),
                                       old.sizeInBytes() + new_img.sizeInBytes()))

    # ================================================================== layer creation
    def _new_paint_layer(self) -> None:
        n = sum(1 for l in self.doc.layers if isinstance(l, RasterLayer))
        self.doc.add_layer(RasterLayer(f"Paint {n}", new_image(self.doc.width, self.doc.height)), label="New paint layer")

    def _auto_paint_layer(self) -> None:
        layer = self.doc.active
        name = layer.name if layer else "layer"
        self._new_paint_layer()
        self._flash(f"'{name}' is a decal/text layer, so a new paint layer was added above it for your painting.")

    def _place_image(self, img: QImage, name: str, pos: Optional[QPointF] = None) -> None:
        if img.isNull():
            return
        layer = ImageLayer(name, img)
        max_w = self.doc.width * 0.3
        s = min(1.0, max_w / max(1, img.width()), (self.doc.height * 0.3) / max(1, img.height()))
        layer.sx = layer.sy = max(0.02, s)
        if pos is None or pos.x() < 0:
            c = self.canvas.to_doc(QPointF(self.canvas.width() / 2, self.canvas.height() / 2))
            pos = QPointF(min(max(c.x(), 0), self.doc.width), min(max(c.y(), 0), self.doc.height))
        layer.x, layer.y = pos.x(), pos.y()
        self.doc.add_layer(layer, label="Add decal")
        self._set_tool(Tool.MOVE)
        self._flash("Decal placed. Drag to move, corner handles to scale, blue dot to rotate. ⇋ mirrors it for the other side.")

    def _load_decal_file(self, path: str, pos: Optional[QPointF] = None) -> None:
        p = Path(path)
        try:
            if p.suffix.lower() == ".svg":
                qimg = QImage(str(p))
                if qimg.isNull():
                    raise ValueError("SVG support not available")
            else:
                qimg = pil_to_qimage(load_any_image(p))
        except Exception as e:
            QMessageBox.warning(self, "Decal", f"Could not open '{p.name}':\n{e}")
            return
        self._place_image(qimg, p.stem, pos)

    def _import_decal(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add decal / sponsor logo", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tga *.psd *.dds *.svg);;All Files (*.*)",
        )
        for p in paths:
            self._load_decal_file(p)

    def _on_files_dropped(self, paths: List[str], pos: QPointF) -> None:
        for p in paths:
            if p.lower().endswith(PROJECT_EXT):
                self._open_project(p)
                return
            self._load_decal_file(p, pos)

    def _paste_image(self) -> None:
        cb = QApplication.clipboard()
        img = cb.image()
        if not img.isNull():
            self._place_image(img, "Pasted image")
        else:
            self._flash("Clipboard has no image.")

    def _on_text_requested(self, pos: QPointF) -> None:
        if pos.x() < 0 and isinstance(self.doc.active, TextLayer):
            self.ed_text.setFocus()
            return
        self._add_text(pos)

    def _add_text(self, pos: Optional[QPointF]) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Add text", "Number / name:", "42")
        if not ok or not text.strip():
            return
        layer = TextLayer(text.splitlines()[0][:24], text)
        layer.font_size = max(24, int(self.doc.width * 0.06))
        layer.color = QColor(self.canvas.primary)
        layer.outline_color = QColor(self.canvas.secondary)
        layer.outline_width = 0
        if pos is None or pos.x() < 0:
            c = self.canvas.to_doc(QPointF(self.canvas.width() / 2, self.canvas.height() / 2))
            pos = c
        layer.x, layer.y = pos.x(), pos.y()
        self.doc.add_layer(layer, label="Add text")
        self._set_tool(Tool.MOVE)

    # ================================================================== undo
    def _undo(self) -> None:
        label = self.doc.undo.undo()
        if label:
            self._flash(f"Undo: {label}")

    def _redo(self) -> None:
        label = self.doc.undo.redo()
        if label:
            self._flash(f"Redo: {label}")

    def _refresh_undo_buttons(self) -> None:
        self.btn_undo.setEnabled(self.doc.undo.can_undo())
        self.btn_redo.setEnabled(self.doc.undo.can_redo())

    # ================================================================== files
    def project_data(self) -> ProjectData:
        guide_png = _qimage_png_bytes(self.doc.guide) if self.doc.guide is not None else None
        return ProjectData(self.doc.width, self.doc.height, list(self.doc.layers), guide_png)

    def _save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save layered project", "livery" + PROJECT_EXT,
                                              f"Livery Paint Project (*{PROJECT_EXT})")
        if not path:
            return
        if not path.lower().endswith(PROJECT_EXT):
            path += PROJECT_EXT
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            save_project(path, self.project_data())
            self.doc.dirty = False
            self._flash(f"Saved project: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _open_project(self, path: Optional[str] = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open layered project", "",
                                                  f"Livery Paint Project (*{PROJECT_EXT})")
        if not path:
            return
        try:
            data = load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", f"Could not open project:\n{e}")
            return
        if (data.width, data.height) != (self.doc.width, self.doc.height):
            QMessageBox.warning(self, "Size mismatch",
                                f"That project is {data.width}×{data.height}, this texture is "
                                f"{self.doc.width}×{self.doc.height}. Layers will be placed at their original size.")
        self.doc.replace_all(data.layers, len(data.layers) - 1)
        if data.guide_png:
            gi = QImage()
            gi.loadFromData(data.guide_png, "PNG")
            self.doc.set_guide(gi)
            self._update_guide_label()
        self._flash(f"Opened project: {Path(path).name}")

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export flattened PNG", "livery.png", "PNG (*.png)")
        if not path:
            return
        self.doc.flatten().save(path, "PNG")
        self._flash(f"Exported {Path(path).name}")

    # ================================================================== finish
    def result_image(self) -> Optional[Image.Image]:
        return self._result

    def _apply(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._result = qimage_to_pil(self.doc.flatten())
        finally:
            QApplication.restoreOverrideCursor()
        self.applied.emit(self._result)
        self.doc.dirty = False
        self.accept()

    def reject(self) -> None:
        if self.doc.dirty and self.doc.undo.can_undo():
            res = QMessageBox.question(
                self, "Discard changes?", "You have unapplied edits. Close the editor and discard them?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Discard:
                return
        super().reject()
