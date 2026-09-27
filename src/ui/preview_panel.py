"""
Panel for Section 3: Livery Viewport, Interactive Annotation Canvas,
AI Refinement Loop, Version History, Finish Selector, and Custom Preview Thumbnail.
"""

from typing import List, Optional
from PIL import Image

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.annotation_canvas import AnnotationCanvas, ToolType
from src.ui.widgets.preview_thumbnail_box import PreviewThumbnailBox
from src.core.config import config
from src.utils.constants import DEFAULT_FINISH, FINISH_ALPHA_MAP, normalize_finish


class LiveryRevision:
    """Represents a single livery version in the multi-turn refinement history."""

    def __init__(self, image: Image.Image, notes: str = "", label: str = ""):
        self.image = image
        self.notes = notes
        self.label = label
        self.committed = False
        self.project = None  # layered Paint Editor project (ProjectData) if this version was hand-painted


class PreviewPanel(QGroupBox):
    """
    Panel displaying the 2D livery preview with interactive annotation tools,
    multi-version revision tracking, AI refinement controls, and custom preview upload.
    """

    generateRequested = Signal()
    promptPackRequested = Signal()  # "Copy for other AI": export prompt + images instead of generating
    refineRequested = Signal(str)  # refinement_notes
    commitRequested = Signal()
    finishChanged = Signal(str)
    loadLiveryRequested = Signal()
    paintEditorRequested = Signal()
    customPreviewChanged = Signal(object)  # Optional[Image.Image]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("3. PREVIEW, ANNOTATION & REFINEMENT", parent)

        self.revisions: List[LiveryRevision] = []
        self.current_rev_idx: int = -1
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Interactive Annotation Viewport Canvas
        self.canvas = AnnotationCanvas(self)

        # 2. Top Drawing & Annotation Toolbar
        toolbar = self._build_annotation_toolbar()
        layout.addLayout(toolbar)
        layout.addWidget(self.canvas, 1)

        # 3. Viewport Controls Row: Finish + Resolution + Wireframe Overlay + Load Existing
        vp_controls = QHBoxLayout()
        vp_controls.setSpacing(8)

        finish_label = QLabel("Gloss:")
        finish_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        vp_controls.addWidget(finish_label)

        self.combo_finish = QComboBox()
        for finish in FINISH_ALPHA_MAP.keys():
            self.combo_finish.addItem(finish)
        self.combo_finish.setCurrentText(normalize_finish(config.get("default_finish", DEFAULT_FINISH)))
        self.combo_finish.setToolTip(
            "How glossy the paint looks in game (written into the texture's alpha channel on install).\n\n"
            "This is NOT the same as the Base Livery in section 4:\n"
            "  • Base Livery = which stock paint shader the car uses (normal, Matte, Chrome, Metallic...\n"
            "    whatever that car offers) plus every texture you don't replace (rims, glass...).\n"
            "  • Gloss = how strongly that base paint's shine shows on your design.\n\n"
            "Gloss 255 · Semi-Gloss 210 · Satin 170 · Matte 80.\n"
            "Want chrome or metallic? Pick a Chrome/Metallic base livery if the car has one and leave this on Gloss.\n"
            "It does not change the AI image - the AI always paints flat colour."
        )
        self.combo_finish.currentTextChanged.connect(self._on_finish_changed)
        self.combo_finish.setStyleSheet(
            """
            QComboBox {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 3px 8px;
                color: #ffffff;
                font-size: 11px;
                min-width: 90px;
            }
            """
        )
        vp_controls.addWidget(self.combo_finish)

        # Shows which base paint the gloss is applied on, so the two settings read as one thing
        self.lbl_base_hint = QLabel("")
        self.lbl_base_hint.setStyleSheet("color: #7d8594; font-size: 10px;")
        self.lbl_base_hint.setToolTip(self.combo_finish.toolTip())
        vp_controls.addWidget(self.lbl_base_hint)

        res_badge = QLabel("4K (4096px)")
        res_badge.setStyleSheet(
            """
            QLabel {
                background-color: #1a2736;
                color: #64b5f6;
                border: 1px solid #234b73;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 10px;
                font-weight: bold;
            }
            """
        )
        vp_controls.addWidget(res_badge)

        self.check_overlay = QCheckBox("Overlay Wireframe")
        self.check_overlay.setToolTip("Overlay semi-transparent wireframe guide lines over the generated livery texture")
        self.check_overlay.setStyleSheet("color: #b0b8c6; font-size: 11px;")
        self.check_overlay.toggled.connect(self.canvas.set_show_wireframe)
        vp_controls.addWidget(self.check_overlay)

        vp_controls.addStretch()

        self.btn_paint = QPushButton("🖌 Paint Editor")
        self.btn_paint.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_paint.setToolTip(
            "Open the layered Paint Editor on this version: add sponsor PNGs, numbers, lines and paint panels\n"
            "with the template shown underneath. The result is added as a new version."
        )
        self.btn_paint.setStyleSheet(
            """
            QPushButton {
                background-color: #3a2b47;
                border: 1px solid #5d3d70;
                border-radius: 4px;
                color: #e1b3ff;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #4b3763; color: #ffffff; }
            QPushButton:disabled { background-color: #1d1b22; color: #4a4552; border-color: #2a2730; }
            """
        )
        self.btn_paint.setEnabled(False)
        self.btn_paint.clicked.connect(self.paintEditorRequested.emit)
        vp_controls.addWidget(self.btn_paint)

        self.btn_load_existing = QPushButton("📂 Load Livery to Edit...")
        self.btn_load_existing.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_existing.setToolTip("Load an existing livery from AMS2 Overrides or disk to edit and refine")
        self.btn_load_existing.setStyleSheet(
            """
            QPushButton {
                background-color: #2b3547;
                border: 1px solid #3d5070;
                border-radius: 4px;
                color: #82b1ff;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #374763;
                color: #ffffff;
            }
            """
        )
        self.btn_load_existing.clicked.connect(self.loadLiveryRequested.emit)
        vp_controls.addWidget(self.btn_load_existing)

        layout.addLayout(vp_controls)

        # 4. In-Game UI Menu Preview Thumbnail Box (Custom upload / Flatpack)
        self.preview_box = PreviewThumbnailBox(self)
        self.preview_box.customPreviewChanged.connect(self.customPreviewChanged.emit)
        layout.addWidget(self.preview_box)

        # 5. Version History & Commit Bar
        history_bar = QHBoxLayout()
        history_bar.setSpacing(6)

        self.btn_prev_rev = QPushButton("◀ Prev")
        self.btn_prev_rev.setFixedWidth(55)
        self.btn_prev_rev.setStyleSheet(self._btn_secondary_style())
        self.btn_prev_rev.setEnabled(False)
        self.btn_prev_rev.clicked.connect(self._prev_revision)
        history_bar.addWidget(self.btn_prev_rev)

        self.lbl_version = QLabel("No Livery Active")
        self.lbl_version.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        history_bar.addWidget(self.lbl_version)

        self.btn_next_rev = QPushButton("Next ▶")
        self.btn_next_rev.setFixedWidth(55)
        self.btn_next_rev.setStyleSheet(self._btn_secondary_style())
        self.btn_next_rev.setEnabled(False)
        self.btn_next_rev.clicked.connect(self._next_revision)
        history_bar.addWidget(self.btn_next_rev)

        self.lbl_commit_status = QLabel("Uncommitted")
        self.lbl_commit_status.setStyleSheet(
            "color: #ffb74d; background-color: #2d2417; border: 1px solid #5a401c; border-radius: 3px; padding: 2px 6px; font-size: 10px; font-weight: 500;"
        )
        self.lbl_commit_status.hide()
        history_bar.addWidget(self.lbl_commit_status)

        history_bar.addStretch()

        self.btn_commit = QPushButton("✅ Commit Livery")
        self.btn_commit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_commit.setToolTip("Lock in this version as the approved livery for deployment")
        self.btn_commit.setStyleSheet(
            """
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #252830;
                color: #555c68;
            }
            """
        )
        self.btn_commit.setEnabled(False)
        self.btn_commit.clicked.connect(self._on_commit_clicked)
        history_bar.addWidget(self.btn_commit)

        layout.addLayout(history_bar)

        # 6. Refinement Notes Input Box
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText(
            "Refinement Notes for AI: Describe specific adjustments (e.g. 'Make the center stripe wider, "
            "shift sponsor logo to front door as drawn in red, change numbers to #42, make roof carbon black...')"
        )
        self.notes_input.setFixedHeight(50)
        self.notes_input.setStyleSheet(
            """
            QTextEdit {
                background-color: #181a20;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 6px;
                color: #ffffff;
                font-size: 11px;
            }
            QTextEdit:focus {
                border: 1px solid #0078d4;
            }
            """
        )
        layout.addWidget(self.notes_input)

        # 7. Main Action Buttons Row: Generate, Refine
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_generate = QPushButton("⚡ Generate Livery")
        self.btn_generate.setFixedHeight(36)
        self.btn_generate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_generate.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d4, stop:1 #0099ff);
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1084e0, stop:1 #20a6ff);
            }
            QPushButton:disabled {
                background: #333945;
                color: #777c86;
            }
            """
        )
        self.btn_generate.clicked.connect(self.generateRequested.emit)
        btn_row.addWidget(self.btn_generate, 2)

        self.btn_prompt_pack = QPushButton("📋 Copy for Other AI")
        self.btn_prompt_pack.setFixedHeight(36)
        self.btn_prompt_pack.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prompt_pack.setToolTip(
            "Don't generate here: save the prompt + template + reference images to a folder and copy them to the "
            "clipboard, ready to paste into ChatGPT, Gemini or any other AI website"
        )
        self.btn_prompt_pack.setStyleSheet(
            """
            QPushButton {
                background-color: #37474f;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: 1px solid #0078d4;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #455a64;
            }
            QPushButton:disabled {
                background: #252830;
                color: #555c68;
                border-color: #333945;
            }
            """
        )
        self.btn_prompt_pack.clicked.connect(self.promptPackRequested.emit)
        btn_row.addWidget(self.btn_prompt_pack, 2)

        self.btn_refine = QPushButton("🔄 Refine with AI")
        self.btn_refine.setFixedHeight(36)
        self.btn_refine.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refine.setToolTip("Send your drawings, notes, and current livery to AI for in-place modifications")
        self.btn_refine.setStyleSheet(
            """
            QPushButton {
                background-color: #7b1fa2;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 5px;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background-color: #8e24aa;
            }
            QPushButton:disabled {
                background: #252830;
                color: #555c68;
            }
            """
        )
        self.btn_refine.setEnabled(False)
        self.btn_refine.clicked.connect(self._on_refine_clicked)
        btn_row.addWidget(self.btn_refine, 2)

        layout.addLayout(btn_row)

    # ==========================================================================
    # Annotation Toolbar UI
    # ==========================================================================

    def _build_annotation_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(4)
        bar.setContentsMargins(0, 0, 0, 2)

        lbl_tools = QLabel("Markup Tools:")
        lbl_tools.setStyleSheet("color: #8c96a5; font-size: 10px; font-weight: bold;")
        bar.addWidget(lbl_tools)

        self.tool_group = QButtonGroup(self)

        tools = [
            ("🖌️ Pen", ToolType.PEN, "Draw freehand annotations"),
            ("➡️ Arrow", ToolType.ARROW, "Draw directional arrow pointing to panels"),
            ("🔲 Box", ToolType.RECT, "Draw rectangular highlight box"),
            ("🔤 Text", ToolType.TEXT, "Place text callout note on texture"),
            ("🧹 Eraser", ToolType.ERASER, "Erase annotations"),
        ]

        for label, tool_type, tip in tools:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setStyleSheet(self._tool_btn_style())
            if tool_type == ToolType.PEN:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, t=tool_type: self.canvas.set_tool(t))
            self.tool_group.addButton(btn)
            bar.addWidget(btn)

        bar.addSpacing(6)

        # Color presets
        colors = [
            ("#FF3B30", "Red"),
            ("#FFCC00", "Yellow"),
            ("#34C759", "Neon Green"),
            ("#007AFF", "Electric Blue"),
            ("#FFFFFF", "White"),
            ("#000000", "Black"),
        ]
        for hex_code, name in colors:
            c_btn = QPushButton()
            c_btn.setFixedSize(18, 18)
            c_btn.setToolTip(f"Markup Color: {name}")
            c_btn.setStyleSheet(
                f"background-color: {hex_code}; border: 1px solid #555; border-radius: 9px;"
            )
            c_btn.clicked.connect(lambda _, h=hex_code: self.canvas.set_color(QColor(h)))
            bar.addWidget(c_btn)

        btn_more_colors = QPushButton("🎨")
        btn_more_colors.setFixedSize(20, 20)
        btn_more_colors.setToolTip("Pick custom color...")
        btn_more_colors.setStyleSheet("background-color: #2b303c; border: 1px solid #3c404b; border-radius: 4px;")
        btn_more_colors.clicked.connect(self._pick_custom_color)
        bar.addWidget(btn_more_colors)

        bar.addSpacing(6)

        # Brush size buttons
        for label, size in [("S", 8), ("M", 16), ("L", 32)]:
            s_btn = QPushButton(label)
            s_btn.setFixedSize(20, 20)
            s_btn.setStyleSheet("background-color: #232730; color: #bbb; border: 1px solid #3c404b; border-radius: 3px; font-size: 9px;")
            s_btn.clicked.connect(lambda _, s=size: self.canvas.set_brush_size(s))
            bar.addWidget(s_btn)

        bar.addSpacing(6)

        # Undo & Clear
        btn_undo = QPushButton("↩️ Undo")
        btn_undo.setStyleSheet(self._btn_secondary_style(padding="2px 6px"))
        btn_undo.clicked.connect(self.canvas.undo)
        bar.addWidget(btn_undo)

        btn_clear = QPushButton("🗑️ Clear")
        btn_clear.setStyleSheet(self._btn_secondary_style(padding="2px 6px"))
        btn_clear.clicked.connect(self.canvas.clear_annotations)
        bar.addWidget(btn_clear)

        check_show_annot = QCheckBox("Show")
        check_show_annot.setChecked(True)
        check_show_annot.setToolTip("Show/hide annotations overlay")
        check_show_annot.setStyleSheet("color: #8c96a5; font-size: 10px;")
        check_show_annot.toggled.connect(self.canvas.set_show_annotations)
        bar.addWidget(check_show_annot)

        bar.addStretch()
        return bar

    def _pick_custom_color(self):
        col = QColorDialog.getColor(self.canvas.pen_color, self, "Pick Annotation Color")
        if col.isValid():
            self.canvas.set_color(col)

    # ==========================================================================
    # Revision & Iteration Management
    # ==========================================================================

    def reset_revisions(self, initial_image: Image.Image, label: str = "v1 (Initial)"):
        """Resets the revision stack with a new initial base livery."""
        self.revisions = [LiveryRevision(initial_image, label=label)]
        self.current_rev_idx = 0
        self.canvas.clear_annotations()
        self._update_revision_ui()

    def add_revision(self, new_image: Image.Image, notes: str = "", label: str = "", project=None):
        """Adds a new livery version (AI refinement or hand-painted) to the history stack."""
        rev_num = len(self.revisions) + 1
        rev_label = label or f"v{rev_num}"
        rev = LiveryRevision(new_image, notes=notes, label=rev_label)
        rev.project = project
        self.revisions.append(rev)
        self.current_rev_idx = len(self.revisions) - 1
        self.canvas.clear_annotations()
        self._update_revision_ui()

    def _prev_revision(self):
        if self.current_rev_idx > 0:
            self.current_rev_idx -= 1
            self._update_revision_ui()

    def _next_revision(self):
        if self.current_rev_idx < len(self.revisions) - 1:
            self.current_rev_idx += 1
            self._update_revision_ui()

    def _on_commit_clicked(self):
        if 0 <= self.current_rev_idx < len(self.revisions):
            self.revisions[self.current_rev_idx].committed = True
            self._update_revision_ui()
            self.commitRequested.emit()

    def _update_revision_ui(self):
        if not self.revisions or self.current_rev_idx < 0:
            self.lbl_version.setText("No Livery Active")
            self.btn_prev_rev.setEnabled(False)
            self.btn_next_rev.setEnabled(False)
            self.btn_commit.setEnabled(False)
            self.btn_refine.setEnabled(False)
            self.btn_paint.setEnabled(False)
            self.lbl_commit_status.hide()
            self.canvas.set_texture(None)
            self.preview_box.set_auto_texture(None)
            return

        rev = self.revisions[self.current_rev_idx]
        total = len(self.revisions)
        curr = self.current_rev_idx + 1

        self.lbl_version.setText(f"{rev.label} ({curr}/{total})")
        self.btn_prev_rev.setEnabled(self.current_rev_idx > 0)
        self.btn_next_rev.setEnabled(self.current_rev_idx < total - 1)
        self.btn_commit.setEnabled(True)
        self.btn_refine.setEnabled(True)
        self.btn_paint.setEnabled(True)

        # Update status badge
        self.lbl_commit_status.show()
        if rev.committed:
            self.lbl_commit_status.setText(f"✅ Committed ({rev.label})")
            self.lbl_commit_status.setStyleSheet(
                "color: #81c784; background-color: #172d1a; border: 1px solid #28542c; border-radius: 3px; padding: 2px 6px; font-size: 10px; font-weight: bold;"
            )
        else:
            self.lbl_commit_status.setText(f"Draft ({rev.label})")
            self.lbl_commit_status.setStyleSheet(
                "color: #ffb74d; background-color: #2d2417; border: 1px solid #5a401c; border-radius: 3px; padding: 2px 6px; font-size: 10px; font-weight: 500;"
            )

        # Update viewport & preview
        self.canvas.set_texture(rev.image)
        self.preview_box.set_auto_texture(rev.image)

    def _on_refine_clicked(self):
        notes = self.notes_input.toPlainText().strip()
        self.refineRequested.emit(notes)

    # ==========================================================================
    # Public Getters & Setters
    # ==========================================================================

    def set_wireframe(self, wf: Optional[Image.Image]):
        self.canvas.set_wireframe(wf)

    def set_generated_image(self, image: Image.Image, is_refinement: bool = False, notes: str = ""):
        """Called when AI generation completes."""
        if is_refinement and self.revisions:
            self.add_revision(image, notes=notes)
        else:
            self.reset_revisions(image, label="v1 (Initial)")

    def get_generated_image(self) -> Optional[Image.Image]:
        """Returns current active revision's clean texture image."""
        if 0 <= self.current_rev_idx < len(self.revisions):
            return self.revisions[self.current_rev_idx].image
        return self.canvas.get_base_image()

    def current_revision(self) -> Optional[LiveryRevision]:
        if 0 <= self.current_rev_idx < len(self.revisions):
            return self.revisions[self.current_rev_idx]
        return None

    def get_annotated_image(self) -> Optional[Image.Image]:
        """Returns base texture composited with user's drawings."""
        return self.canvas.get_annotated_image()

    def get_refinement_notes(self) -> str:
        return self.notes_input.toPlainText().strip()

    def clear_refinement_notes(self) -> None:
        self.notes_input.clear()

    def get_custom_preview_image(self) -> Optional[Image.Image]:
        return self.preview_box.get_custom_preview()

    def set_custom_preview_image(self, img: Optional[Image.Image]) -> None:
        self.preview_box.set_custom_preview(img)

    def get_finish(self) -> str:
        return self.combo_finish.currentText()

    def _on_finish_changed(self, name: str) -> None:
        config.set("default_finish", name)
        self.finishChanged.emit(name)

    def set_base_livery_hint(self, base_name: str) -> None:
        """Called when the Base Livery (section 4) changes: shows what the gloss level is applied on."""
        base_name = (base_name or "").strip()
        self.lbl_base_hint.setText(f"on '{base_name}' base paint" if base_name else "")
        lowered = base_name.lower()
        if any(k in lowered for k in ("matte", "chrome", "metal", "satin", "pearl")):
            self.lbl_base_hint.setStyleSheet("color: #e0a040; font-size: 10px;")
            self.lbl_base_hint.setToolTip(
                f"'{base_name}' has its own special paint shader. Gloss scales how much of it shows:\n"
                "keep Gloss for the full effect, lower values tone it down."
            )
        else:
            self.lbl_base_hint.setStyleSheet("color: #7d8594; font-size: 10px;")
            self.lbl_base_hint.setToolTip(self.combo_finish.toolTip())

    def set_busy(self, is_busy: bool, message: str = ""):
        self.btn_generate.setEnabled(not is_busy)
        self.btn_prompt_pack.setEnabled(not is_busy)
        self.btn_refine.setEnabled(not is_busy and len(self.revisions) > 0)
        self.btn_commit.setEnabled(not is_busy and len(self.revisions) > 0)
        self.btn_load_existing.setEnabled(not is_busy)
        self.btn_paint.setEnabled(not is_busy and len(self.revisions) > 0)

        if is_busy:
            if "Refin" in message:
                self.btn_refine.setText("🔄 Refining...")
            else:
                self.btn_generate.setText("⚡ Generating...")
        else:
            self.btn_generate.setText("⚡ Generate Livery")
            self.btn_refine.setText("🔄 Refine with AI")

    # ==========================================================================
    # Styles
    # ==========================================================================

    @staticmethod
    def _tool_btn_style() -> str:
        return """
            QPushButton {
                background-color: #1e2026;
                color: #d0d4dc;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #2b303c;
                border-color: #0078d4;
            }
            QPushButton:checked {
                background-color: #0078d4;
                color: #ffffff;
                font-weight: bold;
                border-color: #0099ff;
            }
        """

    @staticmethod
    def _btn_secondary_style(padding: str = "4px 8px") -> str:
        return f"""
            QPushButton {{
                background-color: #2b303c;
                color: #ffffff;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: {padding};
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: #3b4252;
                border-color: #0078d4;
            }}
            QPushButton:disabled {{
                background-color: #1a1c22;
                color: #4a505c;
                border-color: #232730;
            }}
        """
