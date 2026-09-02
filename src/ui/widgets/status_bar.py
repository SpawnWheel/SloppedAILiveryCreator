"""
Custom animated status bar widget with progress indicator and status codes.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)


class CustomStatusBar(QFrame):
    """Bottom status bar with status message, progress bar, and status indicator."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            """
            CustomStatusBar {
                background-color: #1a1c22;
                border-top: 1px solid #2e323b;
                min-height: 28px;
                max-height: 28px;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        layout.setSpacing(8)

        # Status Indicator Dot
        self.indicator = QLabel("●")
        self.indicator.setStyleSheet("color: #4CAF50; font-size: 14px;")
        layout.addWidget(self.indicator)

        # Status Text
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        layout.addWidget(self.status_label, 1)

        # Progress Bar (hidden when idle)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate by default
        self.progress_bar.setFixedWidth(160)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #3d424d;
                border-radius: 4px;
                background-color: #262930;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            """
        )
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

    def show_info(self, message: str) -> None:
        self.indicator.setStyleSheet("color: #4CAF50; font-size: 14px;")
        self.status_label.setText(message)
        self.progress_bar.hide()

    def show_busy(self, message: str) -> None:
        self.indicator.setStyleSheet("color: #FFA500; font-size: 14px;")
        self.status_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

    def show_error(self, message: str) -> None:
        self.indicator.setStyleSheet("color: #FF5252; font-size: 14px;")
        self.status_label.setText(f"Error: {message}")
        self.progress_bar.hide()
