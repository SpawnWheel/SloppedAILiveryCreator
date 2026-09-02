"""
Reference Image Thumbnail Widget with role tagging and removal controls.
"""

from typing import Optional
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.constants import REFERENCE_ROLES


class ImageThumbnail(QFrame):
    """A card widget displaying a reference image with role assignment and removal."""

    removeRequested = Signal(object)  # Emits self

    def __init__(
        self,
        pil_image: Image.Image,
        initial_role: str = REFERENCE_ROLES[0],
        filename: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.pil_image = pil_image
        self.filename = filename

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            ImageThumbnail {
                background-color: #262930;
                border: 1px solid #3d424d;
                border-radius: 6px;
            }
            """
        )
        self.setFixedWidth(130)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(4)

        # Header row: small filename label + remove button
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self.label_name = QLabel(filename[:12] + "..." if len(filename) > 12 else filename)
        self.label_name.setStyleSheet("color: #8e94a0; font-size: 10px;")

        self.btn_remove = QPushButton("×")
        self.btn_remove.setFixedSize(18, 18)
        self.btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_remove.setStyleSheet(
            """
            QPushButton {
                background-color: #442222;
                color: #ff6666;
                border: none;
                border-radius: 9px;
                font-weight: bold;
                font-size: 12px;
                padding-bottom: 2px;
            }
            QPushButton:hover {
                background-color: #aa2222;
                color: #ffffff;
            }
            """
        )
        self.btn_remove.clicked.connect(lambda: self.removeRequested.emit(self))

        header.addWidget(self.label_name, 1)
        header.addWidget(self.btn_remove, 0)
        main_layout.addLayout(header)

        # Thumbnail Image Display
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(118, 75)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #1a1c22; border-radius: 4px;")
        self._set_pixmap(pil_image)
        main_layout.addWidget(self.preview_label)

        # Role Selector Dropdown
        self.combo_role = QComboBox()
        for role in REFERENCE_ROLES:
            self.combo_role.addItem(role)
        if initial_role in REFERENCE_ROLES:
            self.combo_role.setCurrentText(initial_role)
        self.combo_role.setStyleSheet(
            """
            QComboBox {
                font-size: 10px;
                padding: 2px 4px;
                background-color: #1e2026;
                color: #e0e0e0;
                border: 1px solid #3c404b;
                border-radius: 4px;
            }
            """
        )
        main_layout.addWidget(self.combo_role)

    def _set_pixmap(self, pil_img: Image.Image) -> None:
        """Converts PIL image to QPixmap scaled for the preview."""
        rgb_img = pil_img.convert("RGBA")
        data = rgb_img.tobytes("raw", "RGBA")
        qimg = QImage(data, rgb_img.width, rgb_img.height, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        scaled_pix = pix.scaled(
            118, 75, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pix)

    @property
    def role(self) -> str:
        return self.combo_role.currentText()
