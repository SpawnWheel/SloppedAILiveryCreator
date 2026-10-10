"""
Livery Paint Editor - a Photoshop / GIMP-style workspace for hand-finishing liveries (or wheel rims).

  * Floating toolbox over the canvas, context-sensitive options bar, dockable / floatable panels
  * Template PANELS: select one or several UV panels (Panel Select tool, Panels dock) and every paint tool -
    brush, eraser, shapes, gradient, bucket, moving pixels - stays inside them. Decals and text placed while
    panels are selected are clipped to those panels, so they can be moved / scaled but never spill over.
  * Marquee + lasso selections, grow / shrink / feather / invert, copy / paste-in-place, layer via copy
  * Layers: paint layers, placed PNG decals, live text; soft brushes, gradients, template-aware bucket fill
  * Undo, zoom / pan, layered projects (.ams2paint), PNG export

PaintEditorWidget is the workspace (embedded in the main window's "Paint Editor" tab);
LiveryPaintEditor wraps it in a dialog (used for wheel textures).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QByteArray, QPointF, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.image_io import load_any_image
from src.core.image_processor import ImageProcessor
from src.ui.editor import icons
from src.ui.editor.canvas import RASTER_TOOLS, SELECT_TOOLS, EditorCanvas, Tool
from src.ui.editor.document import EditorDocument
from src.ui.editor.layers import (
    BLEND_MODES,
    PROJECT_EXT,
    ImageLayer,
    ItemLayer,
    Layer,
    LayerClip,
    ProjectData,
    RasterLayer,
    TextLayer,
    _qimage_png_bytes,
    load_project,
    new_image,
    save_project,
    snapshot_layer,
)
from src.ui.editor.masks import apply_mask_to_image, feather, grow, rect_slices
from src.ui.editor.theme import EDITOR_QSS, PASTEBOARD
from src.ui.editor.widgets import CanvasHost, ColorSwatch, FloatingToolbox, MiniSwatch, SliderSpin
from src.ui.qt_image import numpy_rgba_to_qimage, pil_to_qimage, qimage_to_numpy_rgba, qimage_to_pil

# (tool, icon, tooltip, shortcut, group)
TOOLS = [
    (Tool.MOVE, "move", "Move / Transform (V)\nDrag decals & text: corners scale (Shift = stretch), side dots "
                        "stretch, blue dot rotates.\nOn a paint layer it moves the pixels - only the selected "
                        "ones, kept inside the selection, if something is selected.", "V", "select"),
    (Tool.PANEL, "panel", "Panel Select (W)\nClick a template panel to select it. Shift+click adds, Alt+click "
                          "removes, drag across panels to pick several.\nAfter that, painting and pasted PNGs stay "
                          "inside the selected panels.", "W", "select"),
    (Tool.MARQUEE, "marquee", "Marquee Select (M)\nDrag a rectangle or ellipse. Shift = add, Alt = subtract.",
     "M", "select"),
    (Tool.LASSO, "lasso", "Lasso Select (L)\nDraw around an area. Shift = add, Alt = subtract.", "L", "select"),
    (Tool.BRUSH, "brush", "Brush (B)\nShift+click draws a straight line from the last point. Alt+click picks a "
                          "colour. [ ] size, Shift+[ ] hardness.", "B", "paint"),
    (Tool.ERASER, "eraser", "Eraser (E)", "E", "paint"),
    (Tool.GRADIENT, "gradient", "Gradient (Shift+G)\nDrag to fade foreground → background (or → transparent) "
                                "across the selection.", "Shift+G", "paint"),
    (Tool.FILL, "fill", "Paint Bucket (G)\nClick inside a template panel to paint that whole area.", "G", "paint"),
    (Tool.LINE, "line", "Line (N)\nShift snaps to 15° angles.", "N", "shape"),
    (Tool.RECT, "rect", "Rectangle (U)\nShift = square.", "U", "shape"),
    (Tool.ELLIPSE, "ellipse", "Ellipse (O)\nShift = circle.", "O", "shape"),
    (Tool.POLYGON, "polygon", "Polygon (P)\nClick points; click the first point, double-click, right-click or Enter "
                              "to fill. Backspace removes a point, Esc cancels.", "P", "shape"),
    (Tool.TEXT, "text", "Text (T)\nClick to place a number / name.", "T", "other"),
    (Tool.EYEDROPPER, "eyedropper", "Eyedropper (I)", "I", "other"),
    (Tool.PAN, "pan", "Hand / Pan (H)  -  or hold Space / middle mouse. Mouse wheel zooms.", "H", "other"),
]
TOOL_NAMES = {Tool.MOVE: "Move", Tool.PANEL: "Panel Select", Tool.MARQUEE: "Marquee", Tool.LASSO: "Lasso",
              Tool.BRUSH: "Brush", Tool.ERASER: "Eraser", Tool.GRADIENT: "Gradient", Tool.FILL: "Paint Bucket",
              Tool.LINE: "Line", Tool.RECT: "Rectangle", Tool.ELLIPSE: "Ellipse", Tool.POLYGON: "Polygon",
              Tool.TEXT: "Text", Tool.EYEDROPPER: "Eyedropper", Tool.PAN: "Hand"}

TOOL_HINTS = {
    Tool.MOVE: "Drag decals / text; handles scale & rotate; arrow keys nudge (Shift = 10 px). "
               "Clipped items stay inside their panels.",
    Tool.PANEL: "Click a panel to select it · Shift+click add · Alt+click remove · drag across several panels.",
    Tool.MARQUEE: "Drag to select · Shift = add · Alt = subtract · click empty = deselect.",
    Tool.LASSO: "Draw around the area to select · Shift = add · Alt = subtract.",
    Tool.BRUSH: "Shift+click = straight line · Alt+click = pick colour · [ ] size · 1-0 opacity.",
    Tool.ERASER: "Erases on the current paint layer (inside the selection, if any).",
    Tool.GRADIENT: "Drag across the selection. Shift snaps the angle.",
    Tool.FILL: "Click inside a panel. 'Template outlines' uses the guide's lines as walls.",
    Tool.LINE: "Drag to draw · Shift snaps to 15°.",
    Tool.RECT: "Drag to draw · Shift = square.",
    Tool.ELLIPSE: "Drag to draw · Shift = circle.",
    Tool.POLYGON: "Click corners, then click the first point / double-click / Enter to paint it.",
    Tool.TEXT: "Click where the number / name should go (it's clipped to the selected panels, if any).",
    Tool.EYEDROPPER: "Click to pick the foreground colour from the picture.",
    Tool.PAN: "Drag to pan · wheel zooms · Ctrl+0 fits.",
}

PRESET_COLORS = ["#FFFFFF", "#000000", "#1A1A1A", "#7F7F7F", "#C0C0C0", "#E10600", "#FF6A00", "#FFD200",
                 "#00A651", "#00B5E2", "#0047AB", "#6A0DAD", "#FF2D87", "#D4AF37"]

WORKSPACE_VERSION = 3


def _hsep() -> QFrame:
    f = QFrame()
    f.setFixedWidth(1)
    f.setStyleSheet("background: #3a3d43; margin: 3px 4px;")
    return f


class PaintEditorWidget(QMainWindow):
    """The paint workspace. Embed it (embedded=True) or let LiveryPaintEditor wrap it in a dialog."""

    applied = Signal(object)            # PIL image (flattened, guide not included)
    cancelRequested = Signal()          # dialog mode "Cancel"
    openLiveryRequested = Signal()      # empty-state button
    useCurrentVersionRequested = Signal()
    statusMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, embedded: bool = True):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Widget)
        self.setObjectName("PaintEditor")
        self.setStyleSheet(EDITOR_QSS)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks
                            | QMainWindow.DockOption.AllowNestedDocks)
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self.embedded = embedded
        self._result: Optional[Image.Image] = None
        self._palette_source: Optional[Image.Image] = None
        self._updating_ui = False
        self._slider_before = None
        self._has_content = False
        self.source_key = None
        self._recent: List[QColor] = []
        self._clipboard_origin: Optional[QRect] = None
        self._last_selection: Optional[np.ndarray] = None
        self._hidden_floating: List[QDockWidget] = []
        self._actions: Dict[str, QAction] = {}
        self._restored = False

        self.doc = EditorDocument(1024, 1024, self)

        self._build_central()
        self._build_options_bar()
        self._build_docks()
        self._build_status_bar()
        self._build_actions_and_menus()

        self.doc.layersChanged.connect(self._refresh_layers)
        self.doc.activeChanged.connect(lambda _: self._on_doc_active_changed())
        self.doc.changed.connect(self._refresh_properties_live)
        self.doc.changed.connect(self._thumb_timer.start)
        self.doc.selectionChanged.connect(self._on_selection_changed)
        self.doc.panelsChanged.connect(self._refresh_panels)
        self.doc.guideChanged.connect(self._update_guide_label)
        self.doc.undo.on_change = self._refresh_undo_buttons

        self._refresh_layers()
        self._refresh_panels()
        self._on_doc_active_changed()
        self._refresh_undo_buttons()
        self._set_tool(Tool.BRUSH)
        self._set_empty(embedded)
        self._default_state = None
        self.doc.dirty = False

    # ================================================================== loading
    def load(self, base_image: Optional[Image.Image], guide_image: Optional[Image.Image] = None,
             project: Optional[ProjectData] = None, source_key=None) -> None:
        """Opens a texture (or a saved layered project) for editing. Clears undo history."""
        if project is not None:
            w, h = project.width, project.height
            layers = [snapshot_layer(l) for l in project.layers]  # never edit the version's stored layers
            self.doc.reset(w, h, layers, len(layers) - 1)
            if project.guide_png and guide_image is None:
                gi = QImage()
                gi.loadFromData(project.guide_png, "PNG")
                self.doc.set_guide(gi)
            self.doc.set_panel_names((project.meta or {}).get("panel_names"))
        else:
            w, h = base_image.size
            base = RasterLayer("Base Livery", pil_to_qimage(base_image))
            paint = RasterLayer("Paint 1", new_image(w, h))
            self.doc.reset(w, h, [base, paint], 1)
        if guide_image is not None:
            self.doc.set_guide(pil_to_qimage(guide_image))
        elif project is None:
            self.doc.set_guide(None)
        self._palette_source = base_image if base_image is not None else qimage_to_pil(self.doc.flatten())
        self._rebuild_livery_palette()
        self.source_key = source_key
        self._last_selection = None
        self._set_empty(False)
        self.canvas.fit()
        self._update_guide_label()
        self.doc.dirty = False

    def new_canvas(self, w: int = 4096, h: int = 4096, color: Optional[QColor] = QColor(255, 255, 255),
                   guide_image: Optional[Image.Image] = None) -> None:
        base_img = new_image(w, h)
        if color is not None:
            base_img.fill(color)
        base = RasterLayer("Background", base_img)
        paint = RasterLayer("Paint 1", new_image(w, h))
        self.doc.reset(w, h, [base, paint], 1)
        if guide_image is not None:
            self.doc.set_guide(pil_to_qimage(guide_image))
        self._palette_source = None
        self._rebuild_livery_palette()
        self.source_key = "blank"
        self._set_empty(False)
        self.canvas.fit()
        self.doc.dirty = True

    def set_guide_image(self, guide_image: Optional[Image.Image]) -> None:
        """The main window's template changed."""
        if self._has_content:
            self.doc.set_guide(pil_to_qimage(guide_image) if guide_image is not None else None)

    def has_content(self) -> bool:
        return self._has_content

    def is_dirty(self) -> bool:
        return self._has_content and self.doc.dirty and self.doc.undo.can_undo()

    def mark_clean(self, source_key=None) -> None:
        self.doc.dirty = False
        if source_key is not None:
            self.source_key = source_key

    def _set_empty(self, empty: bool) -> None:
        self._has_content = not empty
        self.canvas.setVisible(not empty)
        self.toolbox.setVisible(not empty and self.act_toolbox.isChecked())
        self.empty_card.setVisible(empty)
        self.host.relayout()
        self._actions["apply"].setEnabled(not empty)
        self.btn_apply.setEnabled(not empty)

    # ================================================================== UI: central
    def _build_central(self) -> None:
        self.host = CanvasHost()
        self.host.setObjectName("EditorRoot")
        self.host.setStyleSheet(f"QWidget#EditorRoot {{ background: {PASTEBOARD}; }}")
        self.host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.canvas = EditorCanvas(self.doc, self.host)
        self.host.set_canvas(self.canvas)
        self.canvas.statusMessage.connect(self._flash)
        self.canvas.colorPicked.connect(lambda c: self._set_primary(c))
        self.canvas.textRequested.connect(self._on_text_requested)
        self.canvas.needPaintLayer.connect(self._auto_paint_layer)
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        self.canvas.cursorMoved.connect(lambda x, y: self.lbl_pos.setText(f"{x}, {y} px"))
        self.canvas.filesDropped.connect(self._on_files_dropped)
        self.canvas.hoverPanelChanged.connect(self._on_canvas_hover_panel)
        self.canvas.colorUsed.connect(self._add_recent)
        self.canvas.contextMenuAt.connect(self._show_context_menu)

        self.toolbox = FloatingToolbox(TOOLS, self.host)
        self.toolbox.toolChosen.connect(self._set_tool)
        self.toolbox.fgbg.fgClicked.connect(lambda: self._choose_color(True))
        self.toolbox.fgbg.bgClicked.connect(lambda: self._choose_color(False))
        self.toolbox.fgbg.swapClicked.connect(self._swap_colors)
        self.toolbox.fgbg.resetClicked.connect(self._reset_colors)
        self.toolbox.move(10, 10)
        self.host.overlays.append(self.toolbox)

        self.empty_card = self._build_empty_card()
        self.empty_card.setParent(self.host)
        self.host.center_overlay = self.empty_card
        self.setCentralWidget(self.host)

    def _build_empty_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("EmptyState")
        card.setFixedWidth(440)
        v = QVBoxLayout(card)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(10)
        ic = QLabel()
        ic.setPixmap(icons.icon("brush", 40).pixmap(40, 40))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(ic)
        t = QLabel("Nothing open in the Paint Editor yet")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("font-size: 15px; font-weight: bold; color: white;")
        v.addWidget(t)
        d = QLabel("Generate a livery in the AI Generation tab, open an existing livery, or start from a blank "
                   "canvas and paint it yourself. Load your car's template first so you can select its panels.")
        d.setWordWrap(True)
        d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        d.setProperty("dim", True)
        v.addWidget(d)
        v.addSpacing(6)
        self.btn_use_current = QPushButton("Edit Current AI Version")
        self.btn_use_current.setProperty("primary", True)
        self.btn_use_current.clicked.connect(self.useCurrentVersionRequested.emit)
        self.btn_use_current.setVisible(False)
        v.addWidget(self.btn_use_current)
        b1 = QPushButton("Open Existing Livery…")
        b1.clicked.connect(self.openLiveryRequested.emit)
        v.addWidget(b1)
        b2 = QPushButton("New Blank Canvas…")
        b2.clicked.connect(self._new_canvas_dialog)
        v.addWidget(b2)
        b3 = QPushButton("Open Layered Project (.ams2paint)…")
        b3.clicked.connect(lambda: self._open_project())
        v.addWidget(b3)
        return card

    def set_current_version_available(self, label: Optional[str]) -> None:
        self.btn_use_current.setVisible(bool(label))
        if label:
            self.btn_use_current.setText(f"Edit {label}")

    # ================================================================== UI: options bar
    def _build_options_bar(self) -> None:
        bar = QToolBar("Tool Options")
        bar.setObjectName("OptionsBar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setIconSize(QSize(18, 18))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)
        self.options_bar = bar
        self._opt_groups: Dict[str, QAction] = {}

        self.lbl_tool_icon = QLabel()
        self.lbl_tool_name = QLabel()
        self.lbl_tool_name.setStyleSheet("color: #e6e7ea; font-weight: bold; min-width: 78px;")
        bar.addWidget(self.lbl_tool_icon)
        bar.addWidget(self.lbl_tool_name)
        bar.addSeparator()

        def group(name: str, *widgets) -> None:
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 6, 0)
            lay.setSpacing(5)
            for x in widgets:
                if isinstance(x, str):
                    lab = QLabel(x)
                    lay.addWidget(lab)
                else:
                    lay.addWidget(x)
            self._opt_groups[name] = bar.addWidget(w)

        # selection mode
        self.selop_group = QButtonGroup(self)
        selop_w = QWidget()
        sl = QHBoxLayout(selop_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(1)
        self.selop_buttons = {}
        for op, glyph, tip in (("replace", "▢", "New selection"), ("add", "⊞", "Add to selection (Shift)"),
                               ("subtract", "⊟", "Subtract from selection (Alt)"),
                               ("intersect", "⊠", "Intersect with selection (Shift+Alt)")):
            b = QToolButton()
            b.setText(glyph)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setFixedSize(26, 24)
            b.setStyleSheet("font-size: 15px;")
            b.clicked.connect(lambda _=False, o=op: setattr(self.canvas, "select_op", o))
            self.selop_group.addButton(b)
            self.selop_buttons[op] = b
            sl.addWidget(b)
        self.selop_buttons["replace"].setChecked(True)
        group("selop", selop_w)

        self.cmb_panel_by = QComboBox()
        self.cmb_panel_by.addItem("Template panels", "panels")
        self.cmb_panel_by.addItem("Colour (magic wand)", "colour")
        self.cmb_panel_by.setToolTip("Template panels: click picks a whole UV panel of the template.\n"
                                     "Colour: picks the connected area of similar colour, like a magic wand.")
        self.cmb_panel_by.currentIndexChanged.connect(self._on_panel_by_changed)
        self.sp_wand_tol = QSpinBox()
        self.sp_wand_tol.setRange(0, 255)
        self.sp_wand_tol.setValue(32)
        self.sp_wand_tol.setToolTip("Magic wand tolerance")
        self.sp_wand_tol.valueChanged.connect(lambda v: setattr(self.canvas, "wand_tolerance", v))
        self.lbl_wand_tol = QLabel("Tolerance")
        btn_all_panels = QPushButton("All Panels")
        btn_all_panels.clicked.connect(self._select_all_panels)
        btn_desel = QPushButton("Deselect")
        btn_desel.clicked.connect(self._deselect)
        group("panel", "Select by", self.cmb_panel_by, self.lbl_wand_tol, self.sp_wand_tol, btn_all_panels, btn_desel)

        self.cmb_marquee = QComboBox()
        self.cmb_marquee.addItem("Rectangle", "rect")
        self.cmb_marquee.addItem("Ellipse", "ellipse")
        self.cmb_marquee.currentIndexChanged.connect(
            lambda _: setattr(self.canvas, "marquee_shape", self.cmb_marquee.currentData()))
        group("marquee", "Shape", self.cmb_marquee)

        self.ss_size = SliderSpin(1, 600, 24, " px", 110, spin_hi=2000)
        self.ss_size.valueChanged.connect(self._on_size_changed)
        group("size", "Size", self.ss_size)
        self.ss_hard = SliderSpin(0, 100, 100, "%", 70)
        self.ss_hard.setToolTip("Hardness - lower it for a soft airbrush edge")
        self.ss_hard.valueChanged.connect(self._on_hardness_changed)
        group("hardness", "Hardness", self.ss_hard)
        self.ss_opacity = SliderSpin(1, 100, 100, "%", 80)
        self.ss_opacity.valueChanged.connect(self._on_brush_opacity)
        group("opacity", "Opacity", self.ss_opacity)

        self.chk_fill = QCheckBox("Fill")
        self.chk_fill.setChecked(True)
        self.chk_fill.setToolTip("Fill with the foreground colour")
        self.chk_outline = QCheckBox("Outline")
        self.chk_outline.setToolTip("Outline with the background colour (Size = line width)")
        self.chk_fill.toggled.connect(lambda v: setattr(self.canvas, "shape_fill", v))
        self.chk_outline.toggled.connect(lambda v: setattr(self.canvas, "shape_outline", v))
        group("shape", self.chk_fill, self.chk_outline)

        self.cmb_fill_sample = QComboBox()
        self.cmb_fill_sample.addItem("Template outlines", "guide")
        self.cmb_fill_sample.addItem("Current layer", "layer")
        self.cmb_fill_sample.addItem("Everything visible", "all")
        self.cmb_fill_sample.setToolTip("What the bucket looks at to find the area's edges")
        self.cmb_fill_sample.currentIndexChanged.connect(
            lambda _: setattr(self.canvas, "fill_sample", self.cmb_fill_sample.currentData()))
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
        self.chk_contig = QCheckBox("Contiguous")
        self.chk_contig.setChecked(True)
        self.chk_contig.setToolTip("Only the connected area (off = every matching pixel)")
        self.chk_contig.toggled.connect(lambda v: setattr(self.canvas, "fill_contiguous", v))
        group("fill", "Fill uses", self.cmb_fill_sample, "Tolerance", self.sp_tol, "Grow", self.sp_grow,
              self.chk_contig)

        self.cmb_grad_type = QComboBox()
        self.cmb_grad_type.addItem("Linear", "linear")
        self.cmb_grad_type.addItem("Radial", "radial")
        self.cmb_grad_type.currentIndexChanged.connect(
            lambda _: setattr(self.canvas, "gradient_type", self.cmb_grad_type.currentData()))
        self.cmb_grad_to = QComboBox()
        self.cmb_grad_to.addItem("Foreground → Background", "secondary")
        self.cmb_grad_to.addItem("Foreground → Transparent", "transparent")
        self.cmb_grad_to.currentIndexChanged.connect(
            lambda _: setattr(self.canvas, "gradient_to", self.cmb_grad_to.currentData()))
        group("gradient", self.cmb_grad_type, self.cmb_grad_to)

        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: #8c9099;")
        group("hint", self.lbl_hint)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        def tbtn(icon_name, tip, slot):
            b = QToolButton()
            b.setIcon(icons.icon(icon_name, 18))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        self.btn_undo = tbtn("undo", "Undo (Ctrl+Z)", self._undo)
        self.btn_redo = tbtn("redo", "Redo (Ctrl+Y)", self._redo)
        bar.addSeparator()
        tbtn("fit", "Fit on screen (Ctrl+0)", lambda: self.canvas.fit())
        self.cmb_zoom = QComboBox()
        self.cmb_zoom.setEditable(True)
        self.cmb_zoom.setFixedWidth(74)
        for z in ("10%", "25%", "50%", "100%", "200%", "400%"):
            self.cmb_zoom.addItem(z)
        self.cmb_zoom.setToolTip("Zoom - type a value or pick one")
        self.cmb_zoom.lineEdit().returnPressed.connect(self._zoom_from_combo)
        self.cmb_zoom.activated.connect(lambda _: self._zoom_from_combo())
        bar.addWidget(self.cmb_zoom)
        bar.addSeparator()

        if not self.embedded:
            b = QPushButton("Cancel")
            b.clicked.connect(self.cancelRequested.emit)
            bar.addWidget(b)
        self.btn_apply = QPushButton("Apply as New Version" if self.embedded else "Apply")
        self.btn_apply.setProperty("apply", True)
        self.btn_apply.setToolTip("Flatten all visible layers (the template guide is NOT included) and add the "
                                  "result as a new livery version (Ctrl+Enter)")
        self.btn_apply.clicked.connect(self._apply)
        bar.addWidget(self.btn_apply)

    # ================================================================== UI: docks
    def _dock(self, title: str, name: str, widget: QWidget, scroll: bool = False) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                      | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        if scroll:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            sa.setWidget(widget)
            d.setWidget(sa)
        else:
            d.setWidget(widget)
        return d

    def _build_docks(self) -> None:
        self.dock_color = self._dock("Colour", "dock_color", self._build_colors_panel())
        self.dock_guide = self._dock("Template", "dock_guide", self._build_guide_panel())
        self.dock_panels = self._dock("Panels", "dock_panels", self._build_panels_panel())
        self.dock_props = self._dock("Properties", "dock_props", self._build_props_panel(), scroll=True)
        self.dock_layers = self._dock("Layers", "dock_layers", self._build_layers_panel())

        right = Qt.DockWidgetArea.RightDockWidgetArea
        # Columns now; the tabbing happens on first show (_arrange_docks): Qt 6 leaves orphan tab bars over the
        # canvas when docks are tabified before the window has its real geometry.
        self.addDockWidget(right, self.dock_color)
        self.addDockWidget(right, self.dock_panels)
        self.addDockWidget(right, self.dock_layers)
        self.addDockWidget(right, self.dock_guide)
        self.addDockWidget(right, self.dock_props)
        self.dock_guide.hide()
        self.dock_props.hide()
        for d in self._all_docks():
            d.setMinimumWidth(300)
            d.dockLocationChanged.connect(self._schedule_tabbar_sweep)
            d.topLevelChanged.connect(self._schedule_tabbar_sweep)
            d.visibilityChanged.connect(self._schedule_tabbar_sweep)
        self._docks_arranged = False

    def _all_docks(self) -> List[QDockWidget]:
        return [self.dock_color, self.dock_guide, self.dock_panels, self.dock_props, self.dock_layers]

    def _arrange_docks(self) -> None:
        """Default workspace: Colour|Template tabs, Panels|Properties tabs, Layers."""
        for d in self._all_docks():
            d.setFloating(False)
        self.tabifyDockWidget(self.dock_color, self.dock_guide)
        self.tabifyDockWidget(self.dock_panels, self.dock_props)
        for d in self._all_docks():
            d.show()
        self.dock_color.raise_()
        self.dock_panels.raise_()
        self._size_docks()
        self._schedule_tabbar_sweep()

    def _schedule_tabbar_sweep(self, *_a) -> None:
        QTimer.singleShot(0, self._sweep_tabbars)

    def _sweep_tabbars(self) -> None:
        """Hide orphan dock tab bars that Qt leaves floating over the central area."""
        from PySide6.QtWidgets import QTabBar

        central = self.centralWidget().geometry() if self.centralWidget() else QRect()
        for tb in self.findChildren(QTabBar):
            if tb.parentWidget() is self and tb.isVisible() and central.contains(tb.geometry().center()):
                tb.hide()

    def _size_docks(self) -> None:
        """Initial dock sizes - only meaningful once the window has its real size."""
        h = max(600, self.height())
        self.resizeDocks([self.dock_color, self.dock_panels, self.dock_layers],
                         [int(h * 0.26), int(h * 0.34), int(h * 0.40)], Qt.Orientation.Vertical)
        self.resizeDocks([self.dock_layers], [330], Qt.Orientation.Horizontal)

    # ---- colour
    def _build_colors_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.sw_primary = ColorSwatch(QColor(230, 30, 40), 30)
        self.sw_primary.setToolTip("Foreground colour (paint / fill / text). Click to change.")
        self.sw_primary.clicked.connect(lambda: self._choose_color(True))
        self.sw_secondary = ColorSwatch(QColor(255, 255, 255), 22)
        self.sw_secondary.setToolTip("Background colour (outlines, gradient end). Click to change.")
        self.sw_secondary.clicked.connect(lambda: self._choose_color(False))
        row.addWidget(self.sw_primary)
        row.addWidget(self.sw_secondary, 0, Qt.AlignmentFlag.AlignBottom)
        self.ed_hex = QLineEdit()
        self.ed_hex.setPlaceholderText("#RRGGBB")
        self.ed_hex.setMaxLength(9)
        self.ed_hex.setFixedWidth(84)
        self.ed_hex.setToolTip("Foreground colour as hex - type and press Enter")
        self.ed_hex.returnPressed.connect(self._hex_entered)
        row.addSpacing(6)
        row.addWidget(self.ed_hex)
        row.addStretch()
        v.addLayout(row)

        lab = QLabel("FROM THIS LIVERY")
        lab.setProperty("section", True)
        v.addWidget(lab)
        self.palette_row = QHBoxLayout()
        self.palette_row.setSpacing(3)
        v.addLayout(self.palette_row)

        lab = QLabel("SWATCHES")
        lab.setProperty("section", True)
        v.addWidget(lab)
        grid = QGridLayout()
        grid.setSpacing(3)
        for i, hexc in enumerate(PRESET_COLORS):
            sw = MiniSwatch(QColor(hexc))
            sw.leftPicked.connect(self._set_primary)
            sw.rightPicked.connect(self._set_secondary)
            grid.addWidget(sw, i // 7, i % 7)
        v.addLayout(grid)

        lab = QLabel("RECENT")
        lab.setProperty("section", True)
        v.addWidget(lab)
        self.recent_row = QHBoxLayout()
        self.recent_row.setSpacing(3)
        v.addLayout(self.recent_row)
        self._rebuild_recent()
        v.addStretch()
        return w

    def _clear_layout(self, lay) -> None:
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

    def _rebuild_livery_palette(self) -> None:
        self._clear_layout(self.palette_row)
        cols = []
        if self._palette_source is not None:
            try:
                cols = ImageProcessor.dominant_colors(self._palette_source, 8)
            except Exception:
                cols = []
        for c in cols:
            sw = MiniSwatch(QColor(*c))
            sw.leftPicked.connect(self._set_primary)
            sw.rightPicked.connect(self._set_secondary)
            self.palette_row.addWidget(sw)
        if not cols:
            lab = QLabel("—")
            lab.setProperty("dim", True)
            self.palette_row.addWidget(lab)
        self.palette_row.addStretch()

    def _rebuild_recent(self) -> None:
        self._clear_layout(self.recent_row)
        for c in self._recent:
            sw = MiniSwatch(c)
            sw.leftPicked.connect(self._set_primary)
            sw.rightPicked.connect(self._set_secondary)
            self.recent_row.addWidget(sw)
        if not self._recent:
            lab = QLabel("Colours you paint with appear here")
            lab.setProperty("dim", True)
            self.recent_row.addWidget(lab)
        self.recent_row.addStretch()

    def _add_recent(self, c: QColor) -> None:
        c = QColor(c)
        c.setAlpha(255)
        self._recent = [x for x in self._recent if x.rgb() != c.rgb()]
        self._recent.insert(0, c)
        self._recent = self._recent[:10]
        self._rebuild_recent()

    # ---- template guide
    def _build_guide_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.chk_guide = QCheckBox("Show template")
        self.chk_guide.setChecked(True)
        self.chk_guide.toggled.connect(self._on_guide_toggle)
        row.addWidget(self.chk_guide)
        row.addStretch()
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
        btn = QPushButton("Load / Change Template Image…")
        btn.clicked.connect(self._load_guide)
        v.addWidget(btn)
        self.lbl_guide = QLabel("")
        self.lbl_guide.setWordWrap(True)
        v.addWidget(self.lbl_guide)
        v.addStretch()
        self._update_guide_label()
        return w

    # ---- panels
    def _build_panels_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        self.lbl_panels = QLabel("")
        self.lbl_panels.setWordWrap(True)
        self.lbl_panels.setProperty("dim", True)
        v.addWidget(self.lbl_panels)
        self.list_panels = QListWidget()
        self.list_panels.setMouseTracking(True)
        self.list_panels.setMinimumHeight(96)
        self.list_panels.setToolTip("Tick panels to select them (you can tick several).\n"
                                    "Double-click a name to rename it (e.g. 'Left door').")
        self.list_panels.itemChanged.connect(self._on_panel_item_changed)
        self.list_panels.itemEntered.connect(
            lambda it: self.canvas.set_highlight_panel(int(it.data(Qt.ItemDataRole.UserRole))))
        self.list_panels.viewport().installEventFilter(self)
        v.addWidget(self.list_panels, 1)
        r = QHBoxLayout()
        r.setSpacing(4)
        for text, tip, slot in (("All", "Select every panel (Ctrl+Alt+A)", self._select_all_panels),
                                ("None", "Deselect (Ctrl+D)", self._deselect),
                                ("Invert", "Invert the selection (Ctrl+Shift+I)", self._invert_selection),
                                ("Zoom", "Zoom to the selection", self._zoom_to_selection)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            r.addWidget(b)
        v.addLayout(r)
        self.chk_panel_labels = QCheckBox("Show panel numbers on the canvas")
        self.chk_panel_labels.toggled.connect(self._on_panel_labels_toggled)
        v.addWidget(self.chk_panel_labels)
        return w

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.list_panels.viewport() and ev.type() == ev.Type.Leave:
            self.canvas.set_highlight_panel(-1)
        return super().eventFilter(obj, ev)

    # ---- layers
    def _build_layers_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(5)
        r = QHBoxLayout()
        self.cmb_layer_blend = QComboBox()
        self.cmb_layer_blend.addItems(list(BLEND_MODES.keys()))
        self.cmb_layer_blend.setToolTip("Blend mode")
        self.cmb_layer_blend.currentTextChanged.connect(self._layer_blend_changed)
        r.addWidget(self.cmb_layer_blend, 1)
        r.addWidget(QLabel("Opacity"))
        self.sl_layer_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sl_layer_opacity.setRange(0, 100)
        self.sl_layer_opacity.setFixedWidth(80)
        self.sl_layer_opacity.sliderPressed.connect(self._layer_slider_pressed)
        self.sl_layer_opacity.sliderReleased.connect(self._layer_slider_released)
        self.sl_layer_opacity.valueChanged.connect(self._layer_opacity_changed)
        r.addWidget(self.sl_layer_opacity)
        self.lbl_layer_opacity = QLabel("100%")
        self.lbl_layer_opacity.setFixedWidth(32)
        r.addWidget(self.lbl_layer_opacity)
        v.addLayout(r)
        r2 = QHBoxLayout()
        self.chk_lock = QCheckBox("Lock")
        self.chk_lock.setToolTip("Protect this layer from painting and moving")
        self.chk_lock.toggled.connect(self._layer_lock_changed)
        r2.addWidget(self.chk_lock)
        r2.addStretch()
        self.lbl_layer_clip = QLabel("")
        self.lbl_layer_clip.setStyleSheet("color: #ffc400;")
        r2.addWidget(self.lbl_layer_clip)
        v.addLayout(r2)

        self.list_layers = QListWidget()
        self.list_layers.setIconSize(QSize(44, 44))
        self.list_layers.setMinimumHeight(170)
        self.list_layers.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_layers.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.list_layers.currentRowChanged.connect(self._on_layer_row_changed)
        self.list_layers.itemChanged.connect(self._on_layer_item_changed)
        self.list_layers.model().rowsMoved.connect(self._on_layer_rows_moved)
        v.addWidget(self.list_layers, 1)

        bar = QHBoxLayout()
        bar.setSpacing(1)
        for icon_name, tip, slot in [
            ("plus", "New paint layer (Ctrl+Shift+N)", self._new_paint_layer),
            ("image", "Place image / decal… (also drag & drop, Ctrl+V)", self._import_decal),
            ("text", "Add text…", lambda: self._add_text(None)),
            ("clip", "Clip layer to the selection (Ctrl+Alt+G) - it will only show inside the selected panels",
             self._clip_layer_to_selection),
            ("duplicate", "Duplicate layer / layer via copy (Ctrl+J)", self._layer_via_copy),
            ("merge", "Merge down (Ctrl+E)", lambda: self.doc.merge_down(self.doc.active_index)),
            ("up", "Move layer up (Ctrl+])", lambda: self.doc.move_layer(self.doc.active_index, +1)),
            ("down", "Move layer down (Ctrl+[)", lambda: self.doc.move_layer(self.doc.active_index, -1)),
            ("trash", "Delete layer", lambda: self.doc.remove_layer(self.doc.active_index)),
        ]:
            b = QToolButton()
            b.setIcon(icons.icon(icon_name, 16))
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch()
        v.addLayout(bar)

        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(450)
        self._thumb_timer.timeout.connect(self._refresh_active_thumb)
        return w

    # ---- properties
    def _build_props_panel(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self.lbl_props_title = QLabel("")
        self.lbl_props_title.setStyleSheet("font-weight: bold; color: white;")
        outer.addWidget(self.lbl_props_title)

        # clip row (all layer kinds)
        clip_box = QFrame()
        cl = QVBoxLayout(clip_box)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)
        lab = QLabel("CONFINED TO PANELS")
        lab.setProperty("section", True)
        cl.addWidget(lab)
        self.lbl_clip = QLabel("")
        self.lbl_clip.setWordWrap(True)
        cl.addWidget(self.lbl_clip)
        cr = QHBoxLayout()
        self.btn_clip = QPushButton("Clip to Selection")
        self.btn_clip.setToolTip("Only show this layer inside the current selection (Ctrl+Alt+G)")
        self.btn_clip.clicked.connect(self._clip_layer_to_selection)
        self.btn_unclip = QPushButton("Release")
        self.btn_unclip.setToolTip("Let this layer show everywhere again")
        self.btn_unclip.clicked.connect(self._release_clip)
        cr.addWidget(self.btn_clip)
        cr.addWidget(self.btn_unclip)
        cl.addLayout(cr)
        outer.addWidget(clip_box)

        self.item_box = QWidget()
        f = QFormLayout(self.item_box)
        f.setContentsMargins(0, 6, 0, 0)
        f.setSpacing(5)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        lab = QLabel("TRANSFORM")
        lab.setProperty("section", True)
        f.addRow(lab)

        def dspin(lo, hi, dec, suffix="", step=1.0):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setSuffix(suffix)
            s.setKeyboardTracking(False)
            s.setMaximumWidth(84)
            return s

        pos = QHBoxLayout()
        self.sp_x = dspin(-20000, 20000, 0)
        self.sp_y = dspin(-20000, 20000, 0)
        pos.addWidget(QLabel("X"))
        pos.addWidget(self.sp_x)
        pos.addWidget(QLabel("Y"))
        pos.addWidget(self.sp_y)
        pos.addStretch()
        f.addRow("Position", pos)

        sc = QHBoxLayout()
        self.sp_sx = dspin(1, 5000, 1, "%", 5)
        self.sp_sy = dspin(1, 5000, 1, "%", 5)
        self.chk_link = QCheckBox("Link")
        self.chk_link.setChecked(True)
        self.chk_link.setToolTip("Keep proportions")
        sc.addWidget(self.sp_sx)
        sc.addWidget(self.sp_sy)
        sc.addWidget(self.chk_link)
        sc.addStretch()
        f.addRow("Scale", sc)

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
            b.setFixedWidth(34)
            b.setStyleSheet("padding: 4px 2px;")
            b.clicked.connect(slot)
            rot.addWidget(b)
        rot.addStretch()
        f.addRow("Rotate", rot)
        fit_row = QHBoxLayout()
        b_fit = QPushButton("Fit to Panels")
        b_fit.setToolTip("Scale and centre this item to fill the area it's clipped to (or the selection)")
        b_fit.clicked.connect(self._fit_item_to_area)
        b_centre = QPushButton("Centre")
        b_centre.setToolTip("Centre the item in its panels / the selection")
        b_centre.clicked.connect(lambda: self._fit_item_to_area(scale=False))
        fit_row.addWidget(b_fit)
        fit_row.addWidget(b_centre)
        f.addRow("", fit_row)

        for s in (self.sp_x, self.sp_y, self.sp_sx, self.sp_sy, self.sp_rot):
            s.valueChanged.connect(self._on_transform_spin)

        # decal-only
        self.decal_row = QWidget()
        dr = QGridLayout(self.decal_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.setSpacing(4)
        b1 = QPushButton("Remove White BG")
        b1.setToolTip("Make white areas transparent (for JPG logos)")
        b1.clicked.connect(lambda: self._key_out((255, 255, 255)))
        b2 = QPushButton("Remove Black BG")
        b2.clicked.connect(lambda: self._key_out((0, 0, 0)))
        b3 = QPushButton("Colourise (FG)")
        b3.setToolTip("Turn the logo into a single colour (your foreground colour), keeping its shape")
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
        tf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.ed_text = QPlainTextEdit()
        self.ed_text.setFixedHeight(48)
        self._text_timer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(350)
        self._text_timer.timeout.connect(lambda: self._apply_text_prop("text", self.ed_text.toPlainText()))
        self.ed_text.textChanged.connect(self._text_timer.start)
        tf.addRow("Text", self.ed_text)
        self.cmb_font = QFontComboBox()
        self.cmb_font.currentFontChanged.connect(lambda f_: self._apply_text_prop("font_family", f_.family()))
        tf.addRow("Font", self.cmb_font)
        fs = QHBoxLayout()
        self.sp_font = QSpinBox()
        self.sp_font.setRange(6, 3000)
        self.sp_font.setSuffix(" px")
        self.sp_font.setKeyboardTracking(False)
        self.sp_font.valueChanged.connect(lambda v: self._apply_text_prop("font_size", v))
        self.chk_bold = QCheckBox("Bold")
        self.chk_bold.toggled.connect(lambda v: self._apply_text_prop("bold", v))
        self.chk_italic = QCheckBox("Italic")
        self.chk_italic.toggled.connect(lambda v: self._apply_text_prop("italic", v))
        fs.addWidget(self.sp_font)
        fs.addWidget(self.chk_bold)
        fs.addWidget(self.chk_italic)
        tf.addRow("Size", fs)
        cl2 = QHBoxLayout()
        self.sw_text = ColorSwatch(QColor("white"), 20)
        self.sw_text.clicked.connect(lambda: self._pick_text_color("color"))
        self.sw_outline = ColorSwatch(QColor("black"), 20)
        self.sw_outline.clicked.connect(lambda: self._pick_text_color("outline_color"))
        self.sp_outline = QSpinBox()
        self.sp_outline.setRange(0, 200)
        self.sp_outline.setSuffix(" px")
        self.sp_outline.setKeyboardTracking(False)
        self.sp_outline.valueChanged.connect(lambda v: self._apply_text_prop("outline_width", v))
        cl2.addWidget(QLabel("Fill"))
        cl2.addWidget(self.sw_text)
        cl2.addWidget(QLabel("Outline"))
        cl2.addWidget(self.sw_outline)
        cl2.addWidget(self.sp_outline)
        cl2.addStretch()
        tf.addRow("Colour", cl2)
        f.addRow(self.text_box)
        outer.addWidget(self.item_box)
        outer.addStretch()
        return w

    # ================================================================== UI: status bar
    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        self.lbl_status = QLabel("Tip: press W and click a template panel - painting and pasted PNGs then stay "
                                 "inside it.")
        sb.addWidget(self.lbl_status, 1)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setStyleSheet("color: #ffc400;")
        sb.addPermanentWidget(self.lbl_sel)
        self.lbl_pos = QLabel("")
        self.lbl_pos.setMinimumWidth(96)
        sb.addPermanentWidget(self.lbl_pos)
        self.lbl_size = QLabel("")
        sb.addPermanentWidget(self.lbl_size)
        self.lbl_zoom = QLabel("")
        self.lbl_zoom.setMinimumWidth(46)
        sb.addPermanentWidget(self.lbl_zoom)

    # ================================================================== actions + menus
    def _act(self, key: str, text: str, slot, shortcut=None, checkable: bool = False, checked: bool = False,
             icon_name: Optional[str] = None, tip: str = "") -> QAction:
        a = QAction(text, self)
        if icon_name:
            a.setIcon(icons.icon(icon_name, 16))
        if shortcut:
            if isinstance(shortcut, (list, tuple)):
                a.setShortcuts([QKeySequence(s) if isinstance(s, str) else QKeySequence(s) for s in shortcut])
            else:
                a.setShortcut(QKeySequence(shortcut) if isinstance(shortcut, str) else QKeySequence(shortcut))
        a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        if checkable:
            a.setCheckable(True)
            a.setChecked(checked)
            a.toggled.connect(slot)
        else:
            a.triggered.connect(lambda _=False: slot())
        if tip:
            a.setStatusTip(tip)
            a.setToolTip(tip)
        self.addAction(a)  # scoped to the editor: shortcuts only work while it has focus
        self._actions[key] = a
        return a

    def _build_actions_and_menus(self) -> None:
        A = self._act
        # tools
        for tool, _i, _t, key, _g in TOOLS:
            A(f"tool_{tool}", TOOL_NAMES[tool], lambda t=tool: self._set_tool(t), key)
        # misc keys
        A("swap", "Swap Colours", self._swap_colors, "X")
        A("reset_colors", "Default Colours", self._reset_colors, "D")
        A("size_down", "Smaller Brush", lambda: self.ss_size.setValue(max(1, int(self.ss_size.value() / 1.2))), "[")
        A("size_up", "Bigger Brush", lambda: self.ss_size.setValue(int(self.ss_size.value() * 1.2) + 1), "]")
        A("hard_down", "Softer Brush", lambda: self.ss_hard.setValue(max(0, self.ss_hard.value() - 25)), "Shift+[")
        A("hard_up", "Harder Brush", lambda: self.ss_hard.setValue(min(100, self.ss_hard.value() + 25)), "Shift+]")
        for d in range(10):
            A(f"op_{d}", f"Opacity {d * 10 or 100}%", lambda v=(d * 10 or 100): self.ss_opacity.setValue(v), str(d))

        m_doc = QMenu("&Paint", self)
        m_doc.addAction(A("new", "New Blank Canvas…", self._new_canvas_dialog))
        m_doc.addAction(A("open_proj", "Open Layered Project…", lambda: self._open_project()))
        m_doc.addAction(A("save_proj", "Save Layered Project…", self._save_project, "Ctrl+S"))
        m_doc.addSeparator()
        m_doc.addAction(A("place", "Place Image / Decal…", self._import_decal, "Ctrl+Shift+P"))
        m_doc.addAction(A("export_png", "Export Flattened PNG…", self._export_png, "Ctrl+Shift+E"))
        m_doc.addSeparator()
        m_doc.addAction(A("apply", "Apply as New Version", self._apply, ["Ctrl+Return", "Ctrl+Enter"]))

        m_edit = QMenu("&Edit", self)
        m_edit.addAction(A("undo", "Undo", self._undo, QKeySequence.StandardKey.Undo, icon_name="undo"))
        m_edit.addAction(A("redo", "Redo", self._redo, ["Ctrl+Y", "Ctrl+Shift+Z"], icon_name="redo"))
        m_edit.addSeparator()
        m_edit.addAction(A("cut", "Cut", self._cut, QKeySequence.StandardKey.Cut))
        m_edit.addAction(A("copy", "Copy", self._copy, QKeySequence.StandardKey.Copy))
        m_edit.addAction(A("copy_merged", "Copy Merged", lambda: self._copy(merged=True), "Ctrl+Shift+C"))
        m_edit.addAction(A("paste", "Paste (into selected panels)", self._paste_image, QKeySequence.StandardKey.Paste))
        m_edit.addAction(A("paste_in_place", "Paste in Place", lambda: self._paste_image(in_place=True),
                           "Ctrl+Shift+V"))
        m_edit.addSeparator()
        m_edit.addAction(A("fill_fg", "Fill Selection with Foreground", lambda: self._fill_selection(True),
                           ["Alt+Backspace", "Shift+F5"]))
        m_edit.addAction(A("fill_bg", "Fill Selection with Background", lambda: self._fill_selection(False),
                           "Ctrl+Backspace"))
        m_edit.addAction(A("clear", "Clear Selected Pixels", self._delete_key, QKeySequence.StandardKey.Delete))

        m_sel = QMenu("&Select", self)
        m_sel.addAction(A("sel_all", "All", self._select_all, QKeySequence.StandardKey.SelectAll))
        m_sel.addAction(A("sel_panels", "All Panels", self._select_all_panels, "Ctrl+Alt+A"))
        m_sel.addAction(A("deselect", "Deselect", self._deselect, "Ctrl+D"))
        m_sel.addAction(A("reselect", "Reselect", self._reselect, "Ctrl+Shift+D"))
        m_sel.addAction(A("invert", "Inverse", self._invert_selection, "Ctrl+Shift+I"))
        m_sel.addSeparator()
        m_sel.addAction(A("grow", "Grow…", lambda: self._modify_selection("grow")))
        m_sel.addAction(A("shrink", "Shrink…", lambda: self._modify_selection("shrink")))
        m_sel.addAction(A("feather", "Feather…", lambda: self._modify_selection("feather"), "Shift+F6"))
        m_sel.addSeparator()
        m_sel.addAction(A("sel_from_layer", "Select Layer's Pixels", self._select_layer_pixels))

        m_layer = QMenu("&Layer", self)
        m_layer.addAction(A("new_layer", "New Paint Layer", self._new_paint_layer, "Ctrl+Shift+N", icon_name="plus"))
        m_layer.addAction(A("via_copy", "Duplicate / Layer via Copy", self._layer_via_copy, "Ctrl+J",
                            icon_name="duplicate"))
        m_layer.addAction(A("via_cut", "Layer via Cut", lambda: self._layer_via_copy(cut=True), "Ctrl+Shift+J"))
        m_layer.addAction(A("add_text", "Add Text…", lambda: self._add_text(None), icon_name="text"))
        m_layer.addSeparator()
        m_layer.addAction(A("clip_layer", "Clip Layer to Selection", self._clip_layer_to_selection, "Ctrl+Alt+G",
                            icon_name="clip"))
        m_layer.addAction(A("release_clip", "Release Clip", self._release_clip))
        m_layer.addAction(A("rasterize", "Rasterize Layer", lambda: self.doc.rasterize_layer(self.doc.active_index)))
        m_layer.addSeparator()
        m_layer.addAction(A("layer_up", "Bring Forward", lambda: self.doc.move_layer(self.doc.active_index, +1),
                            "Ctrl+]"))
        m_layer.addAction(A("layer_down", "Send Backward", lambda: self.doc.move_layer(self.doc.active_index, -1),
                            "Ctrl+["))
        m_layer.addAction(A("merge_down", "Merge Down", lambda: self.doc.merge_down(self.doc.active_index), "Ctrl+E"))
        m_layer.addAction(A("delete_layer", "Delete Layer", lambda: self.doc.remove_layer(self.doc.active_index),
                            icon_name="trash"))

        m_view = QMenu("&View", self)
        m_view.addAction(A("zoom_in", "Zoom In", lambda: self.canvas.set_zoom(self.canvas.scale * 1.25),
                           ["Ctrl+=", "Ctrl++"]))
        m_view.addAction(A("zoom_out", "Zoom Out", lambda: self.canvas.set_zoom(self.canvas.scale / 1.25), "Ctrl+-"))
        m_view.addAction(A("fit", "Fit on Screen", lambda: self.canvas.fit(), "Ctrl+0"))
        m_view.addAction(A("actual", "100%", lambda: self.canvas.set_zoom(1.0), "Ctrl+1"))
        m_view.addAction(A("zoom_sel", "Zoom to Selection", self._zoom_to_selection, "Ctrl+Alt+0"))
        m_view.addSeparator()
        self.act_show_guide = A("show_guide", "Show Template Guide", lambda v: self.chk_guide.setChecked(v),
                                "Ctrl+;", checkable=True, checked=True)
        m_view.addAction(self.act_show_guide)
        m_view.addAction(A("show_edges", "Show Selection Edges", self._toggle_edges, "Ctrl+H", checkable=True,
                           checked=True))
        m_view.addAction(A("show_veil", "Dim Outside Selection", self._toggle_veil, checkable=True, checked=True))
        self.act_panel_labels = A("show_labels", "Show Panel Numbers", lambda v: self.chk_panel_labels.setChecked(v),
                                  checkable=True)
        m_view.addAction(self.act_panel_labels)

        m_win = QMenu("&Window", self)
        self.act_toolbox = A("toolbox", "Tools", self._toggle_toolbox, checkable=True, checked=True)
        m_win.addAction(self.act_toolbox)
        for d in (self.dock_color, self.dock_guide, self.dock_panels, self.dock_props, self.dock_layers):
            ta = d.toggleViewAction()
            ta.setText(d.windowTitle())
            m_win.addAction(ta)
        m_win.addSeparator()
        m_win.addAction(A("reset_ws", "Reset Workspace", self._reset_workspace))

        self._menus = [m_doc, m_edit, m_sel, m_layer, m_view, m_win]
        if not self.embedded:
            mb = QMenuBar(self)
            for m in self._menus:
                mb.addMenu(m)
            self.setMenuBar(mb)

    def menus(self) -> List[QMenu]:
        return list(self._menus)

    def _show_context_menu(self, dpos: QPointF, gpos) -> None:
        """Right-click menu with what makes sense here (like Photoshop's canvas menu)."""
        A = self._actions
        m = QMenu(self)
        pm = self.doc.panels
        idx = pm.index_at(int(dpos.x()), int(dpos.y())) if pm is not None else -1
        if idx >= 0:
            info = pm.infos[idx]
            m.addAction(f"Select {info.title}", lambda: self.doc.set_selection(pm.mask([idx]), info.title))
            m.addAction(f"Add {info.code} to Selection",
                        lambda: self.doc.apply_selection(pm.mask([idx]), "add", info.title))
            m.addAction(f"Rename {info.code}…", lambda: self._rename_panel_dialog(idx))
            m.addSeparator()
        if self.doc.selection.active:
            for k in ("deselect", "invert", "fill_fg", "clear", "via_copy", "copy", "zoom_sel"):
                m.addAction(A[k])
            m.addSeparator()
        layer = self.doc.active
        if isinstance(layer, ItemLayer):
            m.addAction("Flip Horizontal", lambda: self._toggle_flip("flip_h"))
            m.addAction("Fit to Panels", self._fit_item_to_area)
            m.addAction(A["clip_layer"])
            if layer.clip is not None:
                m.addAction(A["release_clip"])
            m.addAction(A["rasterize"])
            m.addSeparator()
        m.addAction(A["paste"])
        m.addAction(A["place"])
        m.addAction(A["add_text"])
        m.addAction(A["new_layer"])
        m.exec(gpos)

    def _rename_panel_dialog(self, idx: int) -> None:
        pm = self.doc.panels
        if pm is None:
            return
        info = pm.infos[idx]
        name, ok = QInputDialog.getText(self, "Rename panel", f"Name for {info.code}:", text=info.name)
        if ok:
            self.doc.rename_panel(idx, name)

    # ================================================================== tool / colour handlers
    def _set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self.toolbox.set_checked(tool)
        self.lbl_tool_icon.setPixmap(icons.icon(dict((t[0], t[1]) for t in TOOLS)[tool], 18).pixmap(18, 18))
        self.lbl_tool_name.setText(TOOL_NAMES.get(tool, tool))
        vis = {
            Tool.MOVE: ["hint"], Tool.PANEL: ["selop", "panel"], Tool.MARQUEE: ["selop", "marquee"],
            Tool.LASSO: ["selop", "hint"], Tool.BRUSH: ["size", "hardness", "opacity"],
            Tool.ERASER: ["size", "hardness", "opacity"], Tool.GRADIENT: ["gradient", "opacity"],
            Tool.FILL: ["fill", "opacity"], Tool.LINE: ["size", "opacity"],
            Tool.RECT: ["size", "shape", "opacity"], Tool.ELLIPSE: ["size", "shape", "opacity"],
            Tool.POLYGON: ["size", "shape", "opacity"], Tool.TEXT: ["hint"], Tool.EYEDROPPER: ["hint"],
            Tool.PAN: ["hint"],
        }.get(tool, ["hint"])
        for name, act in self._opt_groups.items():
            act.setVisible(name in vis)
        self.lbl_hint.setText(TOOL_HINTS.get(tool, ""))
        self._on_panel_by_changed()
        if tool in TOOL_HINTS:
            self.lbl_status.setText(TOOL_HINTS[tool])
        if tool in SELECT_TOOLS:
            if tool == Tool.PANEL:
                self.dock_panels.raise_()
                if self.doc.panels_state == "analysing":
                    self._flash("Finding the template's panels…")
        elif tool == Tool.MOVE and isinstance(self.doc.active, ItemLayer):
            self.dock_props.raise_()

    def _on_panel_by_changed(self, _=None) -> None:
        by = self.cmb_panel_by.currentData()
        self.canvas.panel_by = by
        colour = by == "colour"
        self.sp_wand_tol.setVisible(colour)
        self.lbl_wand_tol.setVisible(colour)

    def _on_size_changed(self, v: int) -> None:
        self.canvas.brush_size = v
        self.canvas._update_cursor()
        self.canvas.update()

    def _on_hardness_changed(self, v: int) -> None:
        self.canvas.brush_hardness = v
        self.canvas.update()

    def _on_brush_opacity(self, v: int) -> None:
        self.canvas.brush_opacity = v / 100.0

    def _set_primary(self, c: QColor) -> None:
        c = QColor(c)
        self.canvas.primary = c
        self.sw_primary.set_color(c)
        self.toolbox.fgbg.set_colors(c, self.canvas.secondary)
        if not self.ed_hex.hasFocus():
            self.ed_hex.setText(c.name().upper())

    def _set_secondary(self, c: QColor) -> None:
        c = QColor(c)
        self.canvas.secondary = c
        self.sw_secondary.set_color(c)
        self.toolbox.fgbg.set_colors(self.canvas.primary, c)

    def _hex_entered(self) -> None:
        txt = self.ed_hex.text().strip()
        if not txt.startswith("#"):
            txt = "#" + txt
        c = QColor(txt)
        if c.isValid():
            self._set_primary(c)
            self.canvas.setFocus()
        else:
            self._flash(f"'{self.ed_hex.text()}' isn't a colour - use #RRGGBB.")

    def _choose_color(self, primary: bool) -> None:
        cur = self.canvas.primary if primary else self.canvas.secondary
        c = QColorDialog.getColor(cur, self, "Foreground colour" if primary else "Background colour")
        if c.isValid():
            (self._set_primary if primary else self._set_secondary)(c)

    def _swap_colors(self) -> None:
        p, s = QColor(self.canvas.primary), QColor(self.canvas.secondary)
        self._set_primary(s)
        self._set_secondary(p)

    def _reset_colors(self) -> None:
        self._set_primary(QColor(0, 0, 0))
        self._set_secondary(QColor(255, 255, 255))

    def _flash(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        self.statusMessage.emit(msg)

    def _on_zoom_changed(self, z: float) -> None:
        txt = f"{z * 100:.0f}%" if z >= 0.1 else f"{z * 100:.1f}%"
        self.lbl_zoom.setText(txt)
        if not self.cmb_zoom.lineEdit().hasFocus():
            self.cmb_zoom.setEditText(txt)

    def _zoom_from_combo(self) -> None:
        txt = self.cmb_zoom.currentText().replace("%", "").strip()
        try:
            v = float(txt) / 100.0
        except ValueError:
            return
        self.canvas.set_zoom(v)
        self.canvas.setFocus()

    def _toggle_toolbox(self, v: bool) -> None:
        self.toolbox.setVisible(v and self._has_content)

    def _toggle_edges(self, v: bool) -> None:
        self.canvas.show_selection_edges = v
        self.canvas.update()

    def _toggle_veil(self, v: bool) -> None:
        self.canvas.show_selection_veil = v
        self.canvas.update()

    def _on_panel_labels_toggled(self, v: bool) -> None:
        self.canvas.show_panel_labels = v
        if self.act_panel_labels.isChecked() != v:
            self.act_panel_labels.setChecked(v)
        self.canvas.update()

    # ================================================================== guide
    def _on_guide_toggle(self, v: bool) -> None:
        self.doc.guide_visible = v
        if hasattr(self, "act_show_guide") and self.act_show_guide.isChecked() != v:
            self.act_show_guide.setChecked(v)
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
        if not hasattr(self, "lbl_guide"):
            return
        if self.doc.guide is None:
            self.lbl_guide.setText("No template loaded. Load your car's template (PNG / PSD / DDS) to see the panel "
                                   "outlines while you paint and to select panels.")
            self.lbl_guide.setStyleSheet("color: #ffb74d;")
        else:
            self.lbl_guide.setText("Template loaded ✓  - shown over your paint only here, never exported.")
            self.lbl_guide.setStyleSheet("color: #81c784;")

    # ================================================================== panels + selection
    def _refresh_panels(self) -> None:
        pm = self.doc.panels
        st = self.doc.panels_state
        self._updating_ui = True
        self.list_panels.blockSignals(True)
        self.list_panels.clear()
        if pm is not None:
            cov = pm.coverage(self.doc.selection.mask)
            for info in pm.infos:
                it = QListWidgetItem(info.name or f"{info.where.replace('-', ' ').capitalize()} panel")
                it.setData(Qt.ItemDataRole.UserRole, info.index)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable)
                it.setCheckState(Qt.CheckState.Checked if cov[info.index] >= 0.9 else Qt.CheckState.Unchecked)
                it.setIcon(self._panel_icon(info.code))
                it.setToolTip(f"{info.code} · {info.where} · {info.area * 100:.1f}% of the texture\n"
                              "Tick to select · double-click to rename")
                self.list_panels.addItem(it)
            self.lbl_panels.setText(f"{len(pm)} panels · tick to select (several allowed), or press W and click "
                                    "them. Painting and pasting then stay inside.")
        elif st == "analysing":
            self.lbl_panels.setText("Finding the template's panels…")
        elif st == "failed":
            self.lbl_panels.setText("Couldn't detect separate panels in this template. The Panel Select tool will "
                                    "pick the area enclosed by the template's lines instead.")
        else:
            self.lbl_panels.setText("No template loaded. Load the car's template (Template tab, or the AI "
                                    "Generation tab) to select its panels.")
        self.list_panels.blockSignals(False)
        self._updating_ui = False

    @staticmethod
    def _panel_icon(code: str) -> QIcon:
        pm = QPixmap(52, 36)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(47, 143, 240))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(2, 6, 46, 24, 5, 5)
        f = QFont()
        f.setPixelSize(15)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255))
        p.drawText(2, 6, 46, 24, Qt.AlignmentFlag.AlignCenter, code)
        p.end()
        return QIcon(pm)

    def _sync_panel_checks(self) -> None:
        pm = self.doc.panels
        if pm is None or self.list_panels.count() != len(pm):
            return
        cov = pm.coverage(self.doc.selection.mask)
        self.list_panels.blockSignals(True)
        for i in range(self.list_panels.count()):
            it = self.list_panels.item(i)
            idx = int(it.data(Qt.ItemDataRole.UserRole))
            st = Qt.CheckState.Checked if cov[idx] >= 0.9 else (
                Qt.CheckState.PartiallyChecked if cov[idx] > 0.02 else Qt.CheckState.Unchecked)
            if it.checkState() != st:
                it.setCheckState(st)
        self.list_panels.blockSignals(False)

    def _on_panel_item_changed(self, it: QListWidgetItem) -> None:
        if self._updating_ui:
            return
        pm = self.doc.panels
        if pm is None:
            return
        idx = int(it.data(Qt.ItemDataRole.UserRole))
        info = pm.infos[idx]
        default = f"{info.where.replace('-', ' ').capitalize()} panel"
        name = it.text().strip()
        if name != (info.name or default):
            self.doc.rename_panel(idx, "" if name == default else name)
            return
        checked = it.checkState() == Qt.CheckState.Checked
        op = "add" if checked else "subtract"
        QTimer.singleShot(0, lambda: self.doc.apply_selection(pm.mask([idx]), op, info.title))

    def _on_canvas_hover_panel(self, idx: int) -> None:
        for i in range(self.list_panels.count()):
            it = self.list_panels.item(i)
            if int(it.data(Qt.ItemDataRole.UserRole)) == idx:
                self.list_panels.setCurrentItem(it)
                self.list_panels.scrollToItem(it)
                return
        self.list_panels.setCurrentRow(-1)

    def _on_selection_changed(self) -> None:
        self._sync_panel_checks()
        sel = self.doc.selection
        if sel.active:
            self._last_selection = sel.copy_mask()
            r = sel.bbox()
            lab = self.doc.selection_label()
            self.lbl_sel.setText(f"◌ {lab}  ({r.width()}×{r.height()})" if r else f"◌ {lab}")
        else:
            self.lbl_sel.setText("")
        self._refresh_properties()

    def _select_all(self) -> None:
        self.doc.set_selection(np.full((self.doc.height, self.doc.width), 255, np.uint8), "All")

    def _select_all_panels(self) -> None:
        pm = self.doc.ensure_panels()
        if pm is None:
            self._flash("No template panels found - load the car's template first.")
            return
        self.doc.set_selection(pm.all_mask(), "All panels")

    def _deselect(self) -> None:
        if self.doc.selection.active:
            self.doc.set_selection(None, "Deselect")

    def _reselect(self) -> None:
        if self._last_selection is not None and not self.doc.selection.active:
            if self._last_selection.shape == (self.doc.height, self.doc.width):
                self.doc.set_selection(self._last_selection.copy(), "Reselect")

    def _invert_selection(self) -> None:
        sel = self.doc.selection
        if sel.active:
            self.doc.set_selection(255 - sel.mask, "Inverse")
        else:
            self._flash("Nothing selected to invert.")

    def _modify_selection(self, how: str) -> None:
        sel = self.doc.selection
        if not sel.active:
            self._flash("Select something first (W = Panel Select).")
            return
        title = {"grow": "Grow selection", "shrink": "Shrink selection", "feather": "Feather selection"}[how]
        px, ok = QInputDialog.getInt(self, title, "Pixels:", 4 if how != "feather" else 6, 1, 200)
        if not ok:
            return
        m = sel.mask
        if how == "grow":
            m = grow(m, px)
        elif how == "shrink":
            m = grow(m, -px)
        else:
            m = feather(m, px)
        self.doc.set_selection(m, f"{title.split()[0]} {px}px")

    def _select_layer_pixels(self) -> None:
        layer = self.doc.active
        if layer is None:
            return
        from src.ui.editor.layers import bake_layer
        from src.ui.editor.masks import qimage_alpha

        a = qimage_alpha(bake_layer(layer, self.doc.width, self.doc.height))
        self.doc.set_selection(a if a.any() else None, f"Pixels of {layer.name}")

    def _zoom_to_selection(self) -> None:
        r = self.doc.selection.bbox()
        if r is None:
            self.canvas.fit()
        else:
            self.canvas.zoom_to_rect(r.adjusted(-40, -40, 40, 40))

    # ================================================================== layers panel
    def _layer_icon(self, layer: Layer) -> QIcon:
        try:
            src = layer.image if isinstance(layer, RasterLayer) else layer.source()  # type: ignore[attr-defined]
            s = 44
            pm = QPixmap(s, s)
            pm.fill(QColor(70, 72, 78))
            p = QPainter(pm)
            for yy in range(0, s, 8):
                for xx in range(0, s, 8):
                    if (xx // 8 + yy // 8) % 2:
                        p.fillRect(xx, yy, 8, 8, QColor(92, 94, 100))
            thumb = src.scaled(s, s, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            p.drawImage((s - thumb.width()) // 2, (s - thumb.height()) // 2, thumb)
            p.setPen(QColor(20, 20, 22))
            p.drawRect(0, 0, s - 1, s - 1)
            badge_x = s - 15
            if layer.clip is not None:
                p.drawPixmap(badge_x, s - 15, icons.icon("clip", 13).pixmap(13, 13))
                badge_x -= 14
            if layer.locked:
                p.drawPixmap(badge_x, s - 15, icons.icon("lock", 13).pixmap(13, 13))
            p.end()
            return QIcon(pm)
        except Exception:
            return QIcon()

    def _refresh_layers(self) -> None:
        self._updating_ui = True
        self.list_layers.blockSignals(True)
        self.list_layers.clear()
        kinds = {"raster": "Paint layer", "image": "Image / decal", "text": "Text"}
        for i in range(len(self.doc.layers) - 1, -1, -1):
            layer = self.doc.layers[i]
            item = QListWidgetItem(self._layer_icon(layer), layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
                          | Qt.ItemFlag.ItemIsDragEnabled)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDropEnabled)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, i)
            tip = [kinds.get(layer.kind, layer.kind)]
            if layer.clip is not None:
                tip.append(f"Clipped to {layer.clip.label or 'a selection'}")
            if layer.locked:
                tip.append("Locked")
            tip.append("Tick = visible · double-click to rename · drag to reorder")
            item.setToolTip("\n".join(tip))
            if not layer.visible:
                item.setForeground(QColor(120, 124, 132))
            self.list_layers.addItem(item)
        self.list_layers.setCurrentRow(len(self.doc.layers) - 1 - self.doc.active_index)
        self.list_layers.blockSignals(False)
        self._updating_ui = False
        self._sync_layer_controls()

    def _refresh_active_thumb(self) -> None:
        row = len(self.doc.layers) - 1 - self.doc.active_index
        it = self.list_layers.item(row)
        layer = self.doc.active
        if it is not None and layer is not None:
            self.list_layers.blockSignals(True)
            it.setIcon(self._layer_icon(layer))
            self.list_layers.blockSignals(False)

    def _on_layer_row_changed(self, row: int) -> None:
        if self._updating_ui or row < 0:
            return
        idx = len(self.doc.layers) - 1 - row
        self.doc.set_active(idx)

    def _on_layer_rows_moved(self, _parent, start, _end, _dest, row) -> None:
        if self._updating_ui:
            return
        n = len(self.doc.layers)
        dst_row = row if row < start else row - 1
        src_idx, dst_idx = n - 1 - start, n - 1 - dst_row
        QTimer.singleShot(0, lambda: self.doc.reorder_layer(src_idx, dst_idx))

    def _on_layer_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating_ui:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None or not (0 <= idx < len(self.doc.layers)):
            return
        layer = self.doc.layers[idx]
        before = {"visible": layer.visible, "name": layer.name}
        name = item.text().strip() or layer.name
        after = {"visible": item.checkState() == Qt.CheckState.Checked, "name": name}
        if before != after:
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
            self.lbl_layer_opacity.setText(f"{int(layer.opacity * 100)}%")
            self.cmb_layer_blend.setCurrentText(layer.blend)
            self.chk_lock.setChecked(layer.locked)
            self.lbl_layer_clip.setText(f"◫ {layer.clip.label}" if layer.clip is not None else "")
        self.lbl_size.setText(f"{self.doc.width}×{self.doc.height}")
        self._updating_ui = False
        self._refresh_properties()

    def _layer_slider_pressed(self) -> None:
        if self.doc.active is not None:
            self._slider_before = self.doc.active.props()

    def _layer_opacity_changed(self, v: int) -> None:
        layer = self.doc.active
        self.lbl_layer_opacity.setText(f"{v}%")
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

    # ================================================================== clip
    def _selection_clip(self) -> Optional[LayerClip]:
        sel = self.doc.selection
        if not sel.active:
            return None
        return LayerClip(sel.mask.copy(), self.doc.selection_label())

    def _clip_layer_to_selection(self) -> None:
        layer = self.doc.active
        clip = self._selection_clip()
        if layer is None:
            return
        if clip is None:
            self._flash("Select the panels first (W), then clip the layer to them.")
            return
        self.doc.set_layer_clip(layer, clip, "Clip to selection")
        self._flash(f"'{layer.name}' now only shows inside {clip.label}.")

    def _release_clip(self) -> None:
        layer = self.doc.active
        if layer is not None and layer.clip is not None:
            self.doc.set_layer_clip(layer, None, "Release clip")

    # ================================================================== properties panel
    def _refresh_properties_live(self) -> None:
        if isinstance(self.doc.active, ItemLayer) and self.dock_props.isVisible():
            self._refresh_properties()

    def _refresh_properties(self) -> None:
        layer = self.doc.active
        if not hasattr(self, "item_box"):
            return
        kinds = {"raster": "Paint layer", "image": "Image / decal", "text": "Text"}
        self.lbl_props_title.setText(f"{kinds.get(layer.kind, '')}: {layer.name}" if layer else "No layer")
        if layer is not None and layer.clip is not None:
            self.lbl_clip.setText(f"Only visible inside <b>{layer.clip.label or 'its clip area'}</b>.")
        else:
            self.lbl_clip.setText("Shows everywhere." + ("" if self.doc.selection.active else
                                                         " Select panels to confine it to them."))
        self.btn_clip.setEnabled(layer is not None and self.doc.selection.active)
        self.btn_unclip.setEnabled(layer is not None and layer.clip is not None)
        is_item = isinstance(layer, ItemLayer)
        self.item_box.setVisible(is_item)
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

    def _area_rect(self, layer: Optional[Layer] = None) -> Optional[QRect]:
        if layer is not None and layer.clip is not None:
            return layer.clip.bbox()
        return self.doc.selection.bbox()

    def _fit_item_to_area(self, scale: bool = True) -> None:
        layer = self.doc.active
        if not isinstance(layer, ItemLayer):
            return
        r = self._area_rect(layer)
        if r is None:
            r = self.doc.rect()
        before = layer.transform_state()
        after = dict(before, x=r.center().x() + 0.5, y=r.center().y() + 0.5)
        if scale:
            w, h = layer.source_size()
            rad = np.radians(layer.rotation)
            bw = abs(w * np.cos(rad)) + abs(h * np.sin(rad))
            bh = abs(w * np.sin(rad)) + abs(h * np.cos(rad))
            s = 0.9 * min(r.width() / max(1.0, bw), r.height() / max(1.0, bh))
            after["sx"] = after["sy"] = max(0.01, s)
        self.doc.commit_transform(layer, before, after)

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

    def _layer_via_copy(self, cut: bool = False) -> None:
        if self.doc.selection.active and isinstance(self.doc.active, RasterLayer):
            self.doc.layer_via_copy(self.doc.active_index, cut=cut)
            self._flash("Selected pixels copied to a new layer." if not cut else "Selected pixels cut to a new layer.")
        elif not cut:
            self.doc.duplicate_layer(self.doc.active_index)

    def _place_image(self, img: QImage, name: str, pos: Optional[QPointF] = None,
                     clip_to_selection: bool = True, fit: bool = True) -> None:
        if img.isNull():
            return
        layer = ImageLayer(name, img)
        clip = self._selection_clip() if clip_to_selection else None
        area = clip.bbox() if clip is not None else None
        if fit:
            if area is not None:
                s = 0.8 * min(area.width() / max(1, img.width()), area.height() / max(1, img.height()))
                s = min(s, 1.0) if img.width() > 64 else s
            else:
                max_w = self.doc.width * 0.3
                s = min(1.0, max_w / max(1, img.width()), (self.doc.height * 0.3) / max(1, img.height()))
            layer.sx = layer.sy = max(0.02, s)
        inside = pos is not None and pos.x() >= 0 and (clip is None or self.doc.selection.contains(int(pos.x()), int(pos.y())))
        if not inside:
            if area is not None:
                pos = QPointF(area.center().x() + 0.5, area.center().y() + 0.5)
            else:
                c = self.canvas.to_doc(QPointF(self.canvas.width() / 2, self.canvas.height() / 2))
                pos = QPointF(min(max(c.x(), 0), self.doc.width), min(max(c.y(), 0), self.doc.height))
        layer.x, layer.y = pos.x(), pos.y()
        layer.clip = clip
        self.doc.add_layer(layer, label="Add decal")
        self._set_tool(Tool.MOVE)
        if clip is not None:
            self._flash(f"Placed inside {clip.label} - it can only show there. Drag to move, corners to scale. "
                        "(Layer ▸ Release Clip to free it.)")
        else:
            self._flash("Decal placed. Drag to move, corner handles to scale, blue dot to rotate. ⇋ mirrors it for "
                        "the other side. Tip: select panels first (W) to keep it inside them.")

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
            self, "Place image / sponsor logo", "",
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

    # ---- clipboard
    def _copy_source(self, merged: bool) -> QImage:
        layer = self.doc.active
        if merged or not isinstance(layer, RasterLayer):
            return self.doc.flatten()
        return layer.image

    def _copy(self, merged: bool = False) -> bool:
        src = self._copy_source(merged)
        sel = self.doc.selection
        if sel.active:
            r = sel.bbox()
            img = apply_mask_to_image(src.copy(r), sel.mask[rect_slices(r)])
        else:
            r = self.doc.rect()
            img = src.copy()
        QApplication.clipboard().setImage(img.convertToFormat(QImage.Format.Format_ARGB32))
        self._clipboard_origin = QRect(r)
        merged_txt = " (all visible layers)" if src is not getattr(self.doc.active, "image", None) else ""
        self._flash(f"Copied {r.width()}×{r.height()} px{merged_txt}. Ctrl+V pastes into the selected panels, "
                    "Ctrl+Shift+V pastes in place.")
        return True

    def _cut(self) -> None:
        if not isinstance(self.doc.active, RasterLayer):
            self._flash("Cut works on paint layers. For decals / text, delete the layer instead.")
            return
        if self._copy():
            self._clear_selected_pixels()

    def _paste_image(self, in_place: bool = False) -> None:
        img = QApplication.clipboard().image()
        if img.isNull():
            self._flash("Clipboard has no image.")
            return
        origin = self._clipboard_origin
        if in_place and origin is not None and img.size() == origin.size():
            layer_pos = QPointF(origin.x() + origin.width() / 2.0, origin.y() + origin.height() / 2.0)
            self._place_image(img, "Pasted image", layer_pos, clip_to_selection=True, fit=False)
            if self.doc.active is not None and isinstance(self.doc.active, ItemLayer):
                self.doc.active.x, self.doc.active.y = layer_pos.x(), layer_pos.y()
                self.doc.invalidate()
        else:
            self._place_image(img, "Pasted image")

    # ---- fill / clear
    def _fill_selection(self, foreground: bool = True) -> None:
        target = self.canvas._paint_target()
        if target is None:
            return
        sel = self.doc.selection
        rect = sel.bbox() if sel.active else self.doc.rect()
        c = QColor(self.canvas.primary if foreground else self.canvas.secondary)
        c.setAlpha(255)
        before = target.image.copy(rect)
        p = QPainter(target.image)
        p.setOpacity(self.canvas.brush_opacity)
        p.fillRect(rect, c)
        p.end()
        self.doc.commit_raster(target, rect, before, "Fill selection")
        self._add_recent(c)
        what = self.doc.selection_label() if sel.active else "the whole layer"
        self._flash(f"Filled {what} with {c.name().upper()}.")

    def _clear_selected_pixels(self) -> None:
        target = self.canvas._paint_target()
        sel = self.doc.selection
        if target is None or not sel.active:
            return
        rect = sel.bbox()
        before = target.image.copy(rect)
        p = QPainter(target.image)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        p.fillRect(rect, Qt.GlobalColor.transparent)
        p.end()
        self.doc.commit_raster(target, rect, before, "Clear")

    def _delete_key(self) -> None:
        layer = self.doc.active
        if self.doc.selection.active and isinstance(layer, RasterLayer):
            self._clear_selected_pixels()
        elif isinstance(layer, ItemLayer):
            self.doc.remove_layer(self.doc.active_index)
        else:
            self._flash("Nothing selected to clear. (Layer ▸ Delete Layer removes the whole layer.)")

    # ---- text
    def _on_text_requested(self, pos: QPointF) -> None:
        if pos.x() < 0 and isinstance(self.doc.active, TextLayer):
            self.dock_props.raise_()
            self.ed_text.setFocus()
            return
        self._add_text(pos)

    def _add_text(self, pos: Optional[QPointF]) -> None:
        text, ok = QInputDialog.getMultiLineText(self, "Add text", "Number / name:", "42")
        if not ok or not text.strip():
            return
        layer = TextLayer(text.splitlines()[0][:24], text)
        clip = self._selection_clip()
        area = clip.bbox() if clip is not None else None
        if area is not None:
            layer.font_size = max(24, int(min(area.height() * 0.45, area.width() * 0.35)))
        else:
            layer.font_size = max(24, int(self.doc.width * 0.06))
        layer.color = QColor(self.canvas.primary)
        layer.outline_color = QColor(self.canvas.secondary)
        layer.outline_width = 0
        inside = pos is not None and pos.x() >= 0 and (clip is None or self.doc.selection.contains(int(pos.x()), int(pos.y())))
        if not inside:
            if area is not None:
                pos = QPointF(area.center().x() + 0.5, area.center().y() + 0.5)
            else:
                pos = self.canvas.to_doc(QPointF(self.canvas.width() / 2, self.canvas.height() / 2))
        layer.x, layer.y = pos.x(), pos.y()
        layer.clip = clip
        self.doc.add_layer(layer, label="Add text")
        self._set_tool(Tool.MOVE)
        self.dock_props.raise_()

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
        return ProjectData(self.doc.width, self.doc.height, [snapshot_layer(l) for l in self.doc.layers], guide_png,
                           {"panel_names": self.doc.panel_names()})

    def _save_project(self) -> None:
        if not self._has_content:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save layered project", "livery" + PROJECT_EXT,
                                              f"Livery Paint Project (*{PROJECT_EXT})")
        if not path:
            return
        if not path.lower().endswith(PROJECT_EXT):
            path += PROJECT_EXT
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            save_project(path, self.project_data())
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
        if self._has_content and (data.width, data.height) != (self.doc.width, self.doc.height):
            QMessageBox.information(self, "Size changed",
                                    f"That project is {data.width}×{data.height}; the canvas now uses that size.")
        guide = self.doc.guide if (self.doc.guide is not None and not data.guide_png) else None
        self.load(None, None, data, source_key="project")
        if guide is not None:
            self.doc.set_guide(guide)
        self.doc.dirty = True
        self._flash(f"Opened project: {Path(path).name}")

    def _export_png(self) -> None:
        if not self._has_content:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export flattened PNG", "livery.png", "PNG (*.png)")
        if not path:
            return
        self.doc.flatten().save(path, "PNG")
        self._flash(f"Exported {Path(path).name}")

    def _new_canvas_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("New Blank Canvas")
        dlg.setStyleSheet(EDITOR_QSS + "QDialog { background: #2b2d31; }")
        f = QFormLayout(dlg)
        size = QComboBox()
        for s in (4096, 2048, 1024):
            size.addItem(f"{s} × {s}", s)
        f.addRow("Size", size)
        bg = QComboBox()
        bg.addItem("White", QColor(255, 255, 255))
        bg.addItem("Background colour", QColor(self.canvas.secondary))
        bg.addItem("Foreground colour", QColor(self.canvas.primary))
        bg.addItem("Transparent", None)
        f.addRow("Fill", bg)
        hint = QLabel("The template from the AI Generation tab (or Template dock) is used as the guide.")
        hint.setWordWrap(True)
        hint.setProperty("dim", True)
        f.addRow(hint)
        row = QHBoxLayout()
        ok = QPushButton("Create")
        ok.setProperty("primary", True)
        ok.clicked.connect(dlg.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        row.addStretch()
        row.addWidget(cancel)
        row.addWidget(ok)
        f.addRow(row)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if self.is_dirty():
            res = QMessageBox.question(self, "Discard edits?", "Start a new canvas and discard the current edits?",
                                       QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if res != QMessageBox.StandardButton.Discard:
                return
        s = size.currentData()
        guide = qimage_to_pil(self.doc.guide) if self.doc.guide is not None else self._external_guide()
        self.new_canvas(s, s, bg.currentData(), guide)
        self._flash("New canvas ready. Press W and click a panel, then paint - it stays inside the panel.")

    def _external_guide(self) -> Optional[Image.Image]:
        """Hook for the host window to provide its template (set by the main window)."""
        fn = getattr(self, "guide_provider", None)
        try:
            return fn() if fn else None
        except Exception:
            return None

    # ================================================================== workspace
    def _first_show_layout(self) -> None:
        self._arrange_docks()
        self._default_state = self.saveState(WORKSPACE_VERSION)
        if self.embedded:
            self._restore_workspace()
        self._schedule_tabbar_sweep()

    def _restore_workspace(self) -> None:
        try:
            from src.core.config import config

            raw = config.get("editor_workspace", "")
            if raw:
                self._restored = bool(self.restoreState(QByteArray(base64.b64decode(raw)), WORKSPACE_VERSION))
        except Exception:
            pass
        self._schedule_tabbar_sweep()

    def save_workspace(self) -> None:
        if not self.embedded or not self._docks_arranged:
            return
        try:
            from src.core.config import config

            config.set("editor_workspace", base64.b64encode(bytes(self.saveState(WORKSPACE_VERSION))).decode())
        except Exception:
            pass

    def _reset_workspace(self) -> None:
        for d in self._all_docks():
            d.setFloating(False)
            d.show()
        if self._default_state is not None:
            self.restoreState(self._default_state, WORKSPACE_VERSION)
        else:
            self._arrange_docks()
        self._size_docks()
        self._schedule_tabbar_sweep()
        self.toolbox.default_position()
        self.act_toolbox.setChecked(True)
        if self.toolbox.isHidden() and self._has_content:
            self.toolbox.show()

    def hideEvent(self, e) -> None:
        # floating docks are separate windows: hide them while the Paint tab isn't showing
        self._hidden_floating = [d for d in (self.dock_color, self.dock_guide, self.dock_panels, self.dock_props,
                                             self.dock_layers) if d.isFloating() and d.isVisible()]
        for d in self._hidden_floating:
            d.hide()
        self.save_workspace()
        super().hideEvent(e)

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if not self._docks_arranged:
            self._docks_arranged = True
            QTimer.singleShot(0, self._first_show_layout)
        for d in self._hidden_floating:
            d.show()
        self._hidden_floating = []
        if self._has_content:
            QTimer.singleShot(0, self.canvas.setFocus)

    # ================================================================== finish
    def result_image(self) -> Optional[Image.Image]:
        return self._result

    def _apply(self) -> None:
        if not self._has_content:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._result = qimage_to_pil(self.doc.flatten())
        finally:
            QApplication.restoreOverrideCursor()
        self.doc.dirty = False
        self.applied.emit(self._result)


class LiveryPaintEditor(QDialog):
    """Dialog wrapper around PaintEditorWidget (wheel textures). Call exec(); on Accepted use result_image()."""

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
        self.setStyleSheet("QDialog { background-color: #232427; }")
        self.resize(1500, 950)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.editor = PaintEditorWidget(self, embedded=False)
        lay.addWidget(self.editor)
        self.editor.load(base_image, guide_image, project)
        self.editor.applied.connect(self._on_applied)
        self.editor.cancelRequested.connect(self.reject)

    def __getattr__(self, name):
        # forward the editor API (doc, canvas, _set_tool, _place_image, sl_guide, _flash, ...) for callers/tests
        if name == "editor" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.editor, name)

    def _on_applied(self, img) -> None:
        self.applied.emit(img)
        self.accept()

    def result_image(self) -> Optional[Image.Image]:
        return self.editor.result_image()

    def project_data(self) -> ProjectData:
        return self.editor.project_data()

    def _apply(self) -> None:
        self.editor._apply()

    def reject(self) -> None:
        if self.editor.is_dirty():
            res = QMessageBox.question(
                self, "Discard changes?", "You have unapplied edits. Close the editor and discard them?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Discard:
                return
        super().reject()
