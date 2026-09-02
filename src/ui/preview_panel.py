"""
Panel for Section 3: 2D Generated Texture Viewport, Finish Selector, and Generation Controls.
"""

from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.utils.constants import DEFAULT_FINISH, FINISH_ALPHA_MAP


class TextureViewport(QLabel):
    """Interactive viewport displaying the generated 2D livery texture."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            """
            TextureViewport {
                background-color: #121418;
                border: 1px solid #2d313b;
                border-radius: 8px;
            }
            """
        )
        self._current_pixmap: Optional[QPixmap] = None
        self._base_image: Optional[Image.Image] = None
        self._wireframe_image: Optional[Image.Image] = None
        self._show_overlay: bool = False
        self._placeholder_text = (
            "2D Generated Texture Viewport\n\n"
            "1. Load a Wireframe Template\n"
            "2. Add Reference Images & Prompt\n"
            "3. Click '⚡ Generate Livery'"
        )

    def set_texture(self, pil_image: Image.Image) -> None:
        """Sets and renders the generated texture."""
        self._base_image = pil_image
        self._rebuild_pixmap()

    def set_wireframe(self, wf: Optional[Image.Image]) -> None:
        """Sets the spatial reference wireframe for overlay."""
        self._wireframe_image = wf
        if self._show_overlay:
            self._rebuild_pixmap()

    def set_show_wireframe(self, show: bool) -> None:
        """Toggles the wireframe guide overlay on the generated texture."""
        self._show_overlay = show
        self._rebuild_pixmap()

    def clear_texture(self) -> None:
        self._base_image = None
        self._current_pixmap = None
        self.update()

    def _rebuild_pixmap(self) -> None:
        if self._base_image is None:
            self._current_pixmap = None
            self._update_display()
            return

        display_img = self._base_image
        if self._show_overlay and self._wireframe_image is not None:
            try:
                base_rgb = self._base_image.convert("RGB")
                wf_resized = self._wireframe_image.resize(base_rgb.size, Image.Resampling.LANCZOS).convert("RGB")
                display_img = Image.blend(base_rgb, wf_resized, alpha=0.35).convert("RGBA")
            except Exception:
                display_img = self._base_image

        rgb = display_img.convert("RGBA")
        data = rgb.tobytes("raw", "RGBA")
        qimg = QImage(data, rgb.width, rgb.height, QImage.Format.Format_RGBA8888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        if self._current_pixmap:
            scaled = self._current_pixmap.scaled(
                self.size().width() - 16,
                self.size().height() - 16,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        else:
            self.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._current_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Draw subtle grid
            painter.setPen(QPen(QColor(35, 38, 46), 1, Qt.PenStyle.DotLine))
            step = 30
            for x in range(0, self.width(), step):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), step):
                painter.drawLine(0, y, self.width(), y)

            # Draw text
            painter.setPen(QColor(120, 128, 142))
            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._placeholder_text,
            )


class PreviewPanel(QGroupBox):
    """Panel displaying the 2D livery preview and AI generation controls."""

    generateRequested = Signal()
    rerollRequested = Signal()
    finishChanged = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("3. PREVIEW & GENERATION", parent)
        self.generated_image: Optional[Image.Image] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        # 1. 2D Texture Viewport
        self.viewport = TextureViewport(self)
        layout.addWidget(self.viewport, 1)

        # 2. Controls Row: Finish Selector + Resolution Indicator
        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)

        finish_label = QLabel("Finish:")
        finish_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        controls_row.addWidget(finish_label)

        self.combo_finish = QComboBox()
        for finish in FINISH_ALPHA_MAP.keys():
            self.combo_finish.addItem(finish)
        self.combo_finish.setCurrentText(DEFAULT_FINISH)
        self.combo_finish.currentTextChanged.connect(self._on_finish_changed)
        self.combo_finish.setStyleSheet(
            """
            QComboBox {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 10px;
                color: #ffffff;
                font-size: 11px;
                min-width: 100px;
            }
            """
        )
        controls_row.addWidget(self.combo_finish)

        controls_row.addSpacing(12)

        res_label = QLabel("Resolution:")
        res_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        controls_row.addWidget(res_label)

        res_badge = QLabel("4096px (4K)")
        res_badge.setStyleSheet(
            """
            QLabel {
                background-color: #1a2736;
                color: #64b5f6;
                border: 1px solid #234b73;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            """
        )
        controls_row.addWidget(res_badge)

        controls_row.addSpacing(12)

        self.check_overlay = QCheckBox("Overlay Wireframe")
        self.check_overlay.setToolTip("Overlay semi-transparent wireframe guide lines over the generated livery texture")
        self.check_overlay.setStyleSheet(
            """
            QCheckBox {
                color: #b0b8c6;
                font-size: 11px;
                font-weight: 500;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3c404b;
                border-radius: 3px;
                background-color: #1e2026;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            """
        )
        self.check_overlay.toggled.connect(self.viewport.set_show_wireframe)
        controls_row.addWidget(self.check_overlay)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        # 3. Action Buttons Row: Generate Livery + Re-roll
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_generate = QPushButton("⚡ Generate Livery")
        self.btn_generate.setFixedHeight(38)
        self.btn_generate.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_generate.setStyleSheet(
            """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d4, stop:1 #0099ff);
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 16px;
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

        self.btn_reroll = QPushButton("⟳ Re-roll")
        self.btn_reroll.setFixedHeight(38)
        self.btn_reroll.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reroll.setStyleSheet(
            """
            QPushButton {
                background-color: #2b303c;
                color: #ffffff;
                font-size: 13px;
                font-weight: 500;
                border: 1px solid #3c404b;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #0078d4;
            }
            QPushButton:disabled {
                background: #252830;
                color: #5a5f6c;
                border-color: #2d3038;
            }
            """
        )
        self.btn_reroll.clicked.connect(self.rerollRequested.emit)
        btn_row.addWidget(self.btn_reroll, 1)

        layout.addLayout(btn_row)

    def _on_finish_changed(self, finish: str):
        self.finishChanged.emit(finish)

    def set_wireframe(self, wireframe: Optional[Image.Image]):
        """Sets the active wireframe template for optional overlay."""
        self.viewport.set_wireframe(wireframe)

    def set_generated_image(self, image: Image.Image):
        self.generated_image = image
        self.viewport.set_texture(image)

    def get_generated_image(self) -> Optional[Image.Image]:
        return self.generated_image

    def get_finish(self) -> str:
        return self.combo_finish.currentText()

    def set_busy(self, is_busy: bool):
        self.btn_generate.setEnabled(not is_busy)
        self.btn_reroll.setEnabled(not is_busy)
        if is_busy:
            self.btn_generate.setText("⚡ Generating...")
        else:
            self.btn_generate.setText("⚡ Generate Livery")
