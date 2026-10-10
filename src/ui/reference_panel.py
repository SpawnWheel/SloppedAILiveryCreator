"""
Panel for Section 2: Multi-Reference Image Board and AI Prompt Studio.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.image_thumbnail import ImageThumbnail
from src.utils.constants import REFERENCE_ROLES, SUPPORTED_IMAGE_EXTS

MAX_REFERENCES = 5


class ReferencePanel(QGroupBox):
    """Panel for uploading 1-5 reference images with role tags and entering prompt instructions."""

    openPromptEditorRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("REFERENCE BOARD && PROMPT", parent)
        self.thumbnails: List[ImageThumbnail] = []
        self._custom_user_prompt: Optional[str] = None
        self._custom_system_instruction: Optional[str] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        # Header for Reference Board
        tray_header = QHBoxLayout()
        self.count_label = QLabel("Reference Images (0/5):")
        self.count_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        tray_header.addWidget(self.count_label)
        tray_header.addStretch()

        self.btn_add_ref = QPushButton("+ Add Reference")
        self.btn_add_ref.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_ref.setStyleSheet(
            """
            QPushButton {
                background-color: #2a3547;
                border: 1px solid #3c5270;
                border-radius: 4px;
                color: #82b1ff;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #354763;
                color: #ffffff;
            }
            """
        )
        self.btn_add_ref.clicked.connect(self._add_reference_dialog)
        tray_header.addWidget(self.btn_add_ref)
        layout.addLayout(tray_header)

        # Horizontal Scroll Area for Reference Cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setFixedHeight(140)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet(
            """
            QScrollArea {
                background-color: #1a1c22;
                border: 1px solid #2e323b;
                border-radius: 6px;
            }
            """
        )

        self.tray_widget = QWidget()
        self.tray_layout = QHBoxLayout(self.tray_widget)
        self.tray_layout.setContentsMargins(6, 6, 6, 6)
        self.tray_layout.setSpacing(8)
        self.tray_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Placeholder label when no references are added
        self.empty_label = QLabel("Add reference images for Colors, Patterns, or Sponsor Decals.")
        self.empty_label.setStyleSheet("color: #606674; font-size: 11px; font-style: italic;")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tray_layout.addWidget(self.empty_label)

        self.scroll_area.setWidget(self.tray_widget)
        layout.addWidget(self.scroll_area)

        # Prompt Input Section Header
        prompt_header = QHBoxLayout()
        prompt_header.setSpacing(8)

        prompt_label = QLabel("Design Guidance & Style Prompt:")
        prompt_label.setStyleSheet("color: #b0b0b0; font-size: 11px; font-weight: bold;")
        prompt_header.addWidget(prompt_label)

        self.custom_badge = QLabel("✏️ Custom Full Prompt Active")
        self.custom_badge.setStyleSheet("color: #64b5f6; font-size: 10px; font-weight: bold;")
        self.custom_badge.hide()
        prompt_header.addWidget(self.custom_badge)

        prompt_header.addStretch()

        self.btn_edit_prompt = QPushButton("📝 View / Edit Full Prompt")
        self.btn_edit_prompt.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_prompt.setToolTip("Inspect or edit the exact text prompt and system instructions sent to Gemini")
        self.btn_edit_prompt.setStyleSheet(
            """
            QPushButton {
                background-color: #2b3547;
                border: 1px solid #3d5070;
                border-radius: 4px;
                color: #82b1ff;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #374763;
                color: #ffffff;
                border-color: #64b5f6;
            }
            """
        )
        self.btn_edit_prompt.clicked.connect(self.openPromptEditorRequested.emit)
        prompt_header.addWidget(self.btn_edit_prompt)

        layout.addLayout(prompt_header)

        self.prompt_text = QTextEdit()
        self.prompt_text.setPlaceholderText(
            "e.g. 1970s retro Gulf theme, racing number #42, vibrant light blue body with bright orange center racing stripe, vintage sponsor decals (Gulf, Heuer, Firestone) on hood and side panels, driver name 'D. Born' on side windows..."
        )
        self.prompt_text.setMinimumHeight(80)
        self.prompt_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 6px;
                padding: 6px;
                color: #ffffff;
                font-size: 12px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border: 1px solid #0078d4;
            }
            """
        )
        self.prompt_text.textChanged.connect(self._on_style_text_changed)
        layout.addWidget(self.prompt_text)

    def _add_reference_dialog(self):
        if len(self.thumbnails) >= MAX_REFERENCES:
            QMessageBox.information(
                self, "Reference Limit", f"You can add a maximum of {MAX_REFERENCES} reference images."
            )
            return

        filter_str = "Image Files (*.png *.jpg *.jpeg *.bmp *.tga *.webp);;All Files (*.*)"
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Reference Images (1-5)", "", filter_str
        )
        for f in files:
            if len(self.thumbnails) >= MAX_REFERENCES:
                break
            self.add_reference_file(Path(f))

    def add_reference_file(self, file_path: Path):
        if len(self.thumbnails) >= MAX_REFERENCES:
            return

        try:
            pil_img = Image.open(file_path)
            # Assign intelligent default role based on index
            role_idx = len(self.thumbnails) % len(REFERENCE_ROLES)
            role = REFERENCE_ROLES[role_idx]

            thumb = ImageThumbnail(pil_img, initial_role=role, filename=file_path.name)
            thumb.removeRequested.connect(self._remove_thumbnail)

            # Hide empty label on first item
            if len(self.thumbnails) == 0:
                self.empty_label.hide()

            self.thumbnails.append(thumb)
            self.tray_layout.addWidget(thumb)
            self._update_counter()
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Could not load reference image:\n{e}")

    def _remove_thumbnail(self, thumb: ImageThumbnail):
        if thumb in self.thumbnails:
            self.thumbnails.remove(thumb)
            self.tray_layout.removeWidget(thumb)
            thumb.deleteLater()
            self._update_counter()

            if len(self.thumbnails) == 0:
                self.empty_label.show()

    def _update_counter(self):
        count = len(self.thumbnails)
        self.count_label.setText(f"Reference Images ({count}/{MAX_REFERENCES}):")
        self.btn_add_ref.setEnabled(count < MAX_REFERENCES)

    def get_references(self) -> List[Tuple[Image.Image, str]]:
        """Returns list of (PIL.Image, role_str)."""
        return [(t.pil_image, t.role) for t in self.thumbnails]

    def get_prompt(self) -> str:
        return self.prompt_text.toPlainText().strip()

    def _on_style_text_changed(self):
        # If user edits the base style text, clear custom prompt override so auto-prompt incorporates changes
        if self._custom_user_prompt is not None:
            self.clear_custom_prompt()

    def set_custom_prompt(self, user_prompt: str, sys_instruction: str) -> None:
        self._custom_user_prompt = user_prompt
        self._custom_system_instruction = sys_instruction
        self.custom_badge.show()

    def get_custom_prompt(self) -> Optional[Tuple[str, str]]:
        if self._custom_user_prompt is not None:
            return (self._custom_user_prompt, self._custom_system_instruction or "")
        return None

    def has_custom_prompt(self) -> bool:
        return self._custom_user_prompt is not None

    def clear_custom_prompt(self) -> None:
        self._custom_user_prompt = None
        self._custom_system_instruction = None
        self.custom_badge.hide()

