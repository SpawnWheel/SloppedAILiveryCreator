"""
In-Game UI Menu Preview Thumbnail Widget (252x64 px).
Displays either auto-generated flatpack thumbnail or user-uploaded custom preview image.
"""

from pathlib import Path
from typing import Optional
from PIL import Image

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.dds_converter import DDSConverter
from src.core.image_processor import ImageProcessor
from src.utils.constants import SUPPORTED_IMAGE_EXTS


class PreviewThumbnailBox(QFrame):
    """
    Widget allowing user to preview and upload a custom 252x64 UI menu thumbnail,
    or default to auto-generating it from the livery texture.
    """

    customPreviewChanged = Signal(object)  # Optional[Image.Image]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setStyleSheet(
            """
            PreviewThumbnailBox {
                background-color: #1a1c22;
                border: 1px solid #2e323b;
                border-radius: 6px;
                padding: 4px;
            }
            """
        )

        self._current_texture: Optional[Image.Image] = None
        self._custom_preview: Optional[Image.Image] = None
        self._thumb_cache: Optional[Image.Image] = None

        self._build_ui()
        self._update_display()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        # 1. 252x64 Thumbnail Display Label (scaled smoothly, e.g. 150x38 or 180x46 for compact UI)
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(168, 43)  # Exact 252x64 aspect ratio (4:1 approx)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet(
            """
            QLabel {
                background-color: #121418;
                border: 1px solid #3c404b;
                border-radius: 4px;
            }
            """
        )
        layout.addWidget(self.thumb_label)

        # 2. Status Badge & Description
        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        info_col.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("In-Game Menu Preview (252×64):")
        self.title_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        info_col.addWidget(self.title_label)

        self.badge_label = QLabel("Auto (Flatpack)")
        self.badge_label.setStyleSheet(
            """
            QLabel {
                color: #8c96a5;
                font-size: 10px;
                font-weight: 500;
            }
            """
        )
        info_col.addWidget(self.badge_label)
        layout.addLayout(info_col, 1)

        # 3. Action Buttons
        btn_col = QHBoxLayout()
        btn_col.setSpacing(6)

        self.btn_upload = QPushButton("📁 Upload...")
        self.btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upload.setToolTip("Upload custom 3D showroom screenshot, photo, or banner for the in-game menu")
        self.btn_upload.setStyleSheet(
            """
            QPushButton {
                background-color: #2b3547;
                border: 1px solid #3d5070;
                border-radius: 4px;
                color: #82b1ff;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #374763;
                color: #ffffff;
            }
            """
        )
        self.btn_upload.clicked.connect(self._choose_custom_file)
        btn_col.addWidget(self.btn_upload)

        self.btn_reset = QPushButton("✕ Reset")
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.setToolTip("Revert back to auto-generated flatpack thumbnail")
        self.btn_reset.setStyleSheet(
            """
            QPushButton {
                background-color: #2e2626;
                border: 1px solid #5a3535;
                border-radius: 4px;
                color: #ff8a80;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3d2b2b;
                color: #ffffff;
            }
            """
        )
        self.btn_reset.clicked.connect(self.clear_custom_preview)
        self.btn_reset.hide()
        btn_col.addWidget(self.btn_reset)

        layout.addLayout(btn_col)

    def set_auto_texture(self, texture_img: Optional[Image.Image]) -> None:
        """Called when a new livery texture is generated or loaded."""
        self._current_texture = texture_img
        self._update_display()

    def set_custom_preview(self, preview_img: Optional[Image.Image]) -> None:
        """Sets a user-provided custom preview image."""
        self._custom_preview = preview_img
        self._update_display()
        self.customPreviewChanged.emit(self._custom_preview)

    def clear_custom_preview(self) -> None:
        """Resets to auto flatpack mode."""
        self._custom_preview = None
        self._update_display()
        self.customPreviewChanged.emit(None)

    def get_custom_preview(self) -> Optional[Image.Image]:
        return self._custom_preview

    def get_effective_preview_thumbnail(self) -> Optional[Image.Image]:
        """Returns the 252x64 PIL thumbnail image ready for DDS compression."""
        if self._custom_preview is not None:
            return ImageProcessor.generate_preview_thumbnail(self._custom_preview)
        elif self._current_texture is not None:
            return ImageProcessor.generate_preview_thumbnail(self._current_texture)
        return None

    def _choose_custom_file(self) -> None:
        filter_str = "Image Files (*.png *.jpg *.jpeg *.bmp *.dds *.webp);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Livery Preview Image", "", filter_str
        )
        if file_path:
            self.load_custom_file(Path(file_path))

    def load_custom_file(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".dds":
                img = DDSConverter.load_dds_image(path)
            else:
                img = Image.open(path).copy()

            self.set_custom_preview(img)
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Error", f"Failed to load preview image:\n{e}")

    def _update_display(self) -> None:
        thumb = self.get_effective_preview_thumbnail()
        if thumb is not None:
            rgb = thumb.convert("RGB")
            data = rgb.tobytes("raw", "RGB")
            qimg = QImage(data, rgb.width, rgb.height, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                self.thumb_label.width(),
                self.thumb_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumb_label.setPixmap(pix)
        else:
            self.thumb_label.setText("No Preview")
            self.thumb_label.setStyleSheet(
                "background-color: #121418; color: #555c68; font-size: 10px; border: 1px dashed #2d313b;"
            )

        if self._custom_preview is not None:
            self.badge_label.setText("🌟 Custom Preview Active")
            self.badge_label.setStyleSheet("color: #81c784; font-size: 10px; font-weight: bold;")
            self.btn_reset.show()
        else:
            self.badge_label.setText("Auto (Flatpack Crop)")
            self.badge_label.setStyleSheet("color: #8c96a5; font-size: 10px; font-style: italic;")
            self.btn_reset.hide()

    # ==========================================================================
    # Drag & Drop Support
    # ==========================================================================

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS or ext == ".dds":
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            fpath = Path(url.toLocalFile())
            if fpath.is_file():
                self.load_custom_file(fpath)
                event.acceptProposedAction()
                return
