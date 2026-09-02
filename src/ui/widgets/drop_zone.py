"""
Drag-and-Drop file and URL ingestion widget.
"""

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog

from src.utils.constants import SUPPORTED_ARCHIVE_EXTS, SUPPORTED_IMAGE_EXTS


class DropZone(QWidget):
    """Interactive drag-and-drop zone for wireframe templates (files or zip archives)."""

    fileDropped = Signal(Path)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(110)
        self._is_hovered = False
        self._loaded_filename: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(12, 12, 12, 12)

        self.label_icon = QLabel("📂", self)
        self.label_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.label_icon.font()
        font.setPointSize(24)
        self.label_icon.setFont(font)

        self.label_text = QLabel("Drag & Drop Wireframe (.png, .psd, .zip)\nor Click to Browse", self)
        self.label_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_text.setStyleSheet("color: #b0b0b0; font-size: 12px;")

        layout.addWidget(self.label_icon)
        layout.addWidget(self.label_text)

        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_loaded_file(self, filename: str) -> None:
        self._loaded_filename = filename
        self.label_icon.setText("✅")
        self.label_text.setText(f"Loaded: {filename}")
        self.label_text.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 12px;")
        self.update()

    def clear(self) -> None:
        self._loaded_filename = None
        self.label_icon.setText("📂")
        self.label_text.setText("Drag & Drop Wireframe (.png, .psd, .zip)\nor Click to Browse")
        self.label_text.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._browse_file()

    def _browse_file(self):
        filter_str = (
            "All Supported Files (*.png *.jpg *.jpeg *.psd *.zip *.bmp *.tga);;"
            "Image Files (*.png *.jpg *.jpeg *.psd *.bmp *.tga);;"
            "ZIP Archives (*.zip);;All Files (*.*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Vehicle UV Wireframe / Template", "", filter_str
        )
        if file_path:
            p = Path(file_path)
            self.set_loaded_file(p.name)
            self.fileDropped.emit(p)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                suffix = Path(urls[0].toLocalFile()).suffix.lower()
                if suffix in SUPPORTED_IMAGE_EXTS or suffix in SUPPORTED_ARCHIVE_EXTS:
                    event.acceptProposedAction()
                    self._is_hovered = True
                    self.update()
                    return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._is_hovered = False
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:
        self._is_hovered = False
        urls = event.mimeData().urls()
        if urls:
            local_path = urls[0].toLocalFile()
            if local_path:
                p = Path(local_path)
                self.set_loaded_file(p.name)
                self.fileDropped.emit(p)
                event.acceptProposedAction()
                self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background color
        if self._is_hovered:
            bg_color = QColor(40, 60, 80, 180)
            border_color = QColor(0, 150, 255)
            pen_style = Qt.PenStyle.SolidLine
        elif self._loaded_filename:
            bg_color = QColor(25, 45, 30, 140)
            border_color = QColor(76, 175, 80)
            pen_style = Qt.PenStyle.SolidLine
        else:
            bg_color = QColor(32, 35, 42, 160)
            border_color = QColor(70, 75, 88)
            pen_style = Qt.PenStyle.DashLine

        painter.setBrush(bg_color)
        pen = QPen(border_color, 2, pen_style)
        painter.setPen(pen)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 8, 8)
