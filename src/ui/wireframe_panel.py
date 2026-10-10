"""
Panel for Section 1: Wireframe / Template Ingestion.
"""

from pathlib import Path
from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from src.core.template_loader import TemplateLoader
from src.ui.widgets.drop_zone import DropZone


class URLLoaderThread(QThread):
    """Worker thread for downloading templates via URL without freezing GUI."""

    success = Signal(object, str)  # (PIL.Image, filename)
    error = Signal(str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        try:
            img, name = TemplateLoader.load_from_url(self.url)
            self.success.emit(img, name)
        except Exception as e:
            self.error.emit(str(e))


class WireframePanel(QGroupBox):
    """Panel for vehicle UV wireframe / template ingestion (file, ZIP, or URL)."""

    wireframeLoaded = Signal(object, str)  # (PIL.Image, display_name)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("TEMPLATE / WIREFRAME", parent)
        self.current_wireframe: Optional[Image.Image] = None
        self.current_name: str = ""
        self._url_thread: Optional[URLLoaderThread] = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        # 1. Drag & Drop File Zone
        self.drop_zone = DropZone(self)
        self.drop_zone.fileDropped.connect(self._on_file_dropped)
        layout.addWidget(self.drop_zone)

        # 2. URL Input Bar
        url_layout = QHBoxLayout()
        url_layout.setSpacing(6)

        url_label = QLabel("or URL:")
        url_label.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        url_layout.addWidget(url_label)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://.../template.zip or .png")
        self.url_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
            }
            """
        )
        self.url_input.returnPressed.connect(self._on_fetch_url)
        url_layout.addWidget(self.url_input, 1)

        self.btn_fetch = QPushButton("Fetch")
        self.btn_fetch.setFixedWidth(60)
        self.btn_fetch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fetch.setStyleSheet(
            """
            QPushButton {
                background-color: #2b303c;
                border: 1px solid #3c404b;
                border-radius: 4px;
                color: #ffffff;
                padding: 4px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #0078d4;
            }
            """
        )
        self.btn_fetch.clicked.connect(self._on_fetch_url)
        url_layout.addWidget(self.btn_fetch)

        layout.addLayout(url_layout)

        # 3. Mini wireframe status / preview summary
        self.info_label = QLabel("No template loaded")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #707580; font-size: 11px; font-style: italic;")
        layout.addWidget(self.info_label)

    def _on_file_dropped(self, file_path: Path):
        try:
            img, name = TemplateLoader.load_from_path(file_path)
            self._set_wireframe(img, name)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load wireframe:\n{e}")
            self.drop_zone.clear()

    def _on_fetch_url(self):
        url = self.url_input.text().strip()
        if not url:
            return

        self.btn_fetch.setEnabled(False)
        self.btn_fetch.setText("Fetching...")
        self.info_label.setText("Downloading and analyzing template...")

        self._url_thread = URLLoaderThread(url)
        self._url_thread.success.connect(self._on_url_success)
        self._url_thread.error.connect(self._on_url_error)
        self._url_thread.start()

    def _on_url_success(self, img: Image.Image, name: str):
        self.btn_fetch.setEnabled(True)
        self.btn_fetch.setText("Fetch")
        self.drop_zone.set_loaded_file(name)
        self._set_wireframe(img, name)

    def _on_url_error(self, err_msg: str):
        self.btn_fetch.setEnabled(True)
        self.btn_fetch.setText("Fetch")
        self.info_label.setText("Fetch failed")
        QMessageBox.warning(self, "Download Error", f"Failed to fetch template from URL:\n{err_msg}")

    def _set_wireframe(self, img: Image.Image, name: str):
        self.current_wireframe = img
        self.current_name = name
        w, h = img.size
        self.info_label.setText(f"📐 {name} ({w} × {h} px)")
        self.info_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
        self.wireframeLoaded.emit(img, name)

    def get_wireframe(self) -> Optional[Image.Image]:
        return self.current_wireframe
