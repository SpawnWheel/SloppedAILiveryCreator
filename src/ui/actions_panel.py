"""
Panel for Section 5: Actions (Install to AMS2, Shareable ZIP Export, Rollback Backups).
"""

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.installer import LiveryInstaller


class BackupRestoreDialog(QDialog):
    """Dialog allowing user to view and restore previous XML backups for a car model."""

    def __init__(self, car_folder: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.car_folder = car_folder
        self.selected_backup: Optional[Path] = None

        self.setWindowTitle(f"Restore Backup — {car_folder}")
        self.setMinimumSize(450, 300)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1e2026;
                color: #ffffff;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        header = QLabel(f"Available Backups for '{car_folder}':")
        header.setStyleSheet("font-weight: bold; font-size: 12px; color: #ffffff;")
        layout.addWidget(header)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            """
            QListWidget {
                background-color: #14161a;
                border: 1px solid #2e323b;
                border-radius: 6px;
                color: #e0e0e0;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #232730;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
                border-radius: 4px;
            }
            """
        )
        layout.addWidget(self.list_widget)

        # Populate backups
        backups = LiveryInstaller.list_backups(car_folder)
        if not backups:
            item = QListWidgetItem("No backups found for this car.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_widget.addItem(item)
        else:
            for b_path, b_label in backups:
                item = QListWidgetItem(b_label)
                item.setData(Qt.ItemDataRole.UserRole, b_path)
                self.list_widget.addItem(item)
            self.list_widget.setCurrentRow(0)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(
            "background-color: #2b303c; border: 1px solid #3c404b; padding: 6px 14px; border-radius: 4px;"
        )
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_restore = QPushButton("↺ Restore Selected")
        self.btn_restore.setEnabled(len(backups) > 0)
        self.btn_restore.setStyleSheet(
            "background-color: #d83b01; color: white; font-weight: bold; border: none; padding: 6px 16px; border-radius: 4px;"
        )
        self.btn_restore.clicked.connect(self._on_restore_clicked)
        btn_layout.addWidget(self.btn_restore)

        layout.addLayout(btn_layout)

    def _on_restore_clicked(self):
        curr_item = self.list_widget.currentItem()
        if curr_item:
            b_path = curr_item.data(Qt.ItemDataRole.UserRole)
            if b_path:
                self.selected_backup = b_path
                self.accept()


class ActionsPanel(QGroupBox):
    """Panel for installation, export, and rollback actions."""

    installRequested = Signal()
    exportRequested = Signal()
    restoreRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("5. ACTIONS", parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 14, 10, 10)

        # 1. Install Button (Primary)
        self.btn_install = QPushButton("💾 Install to AMS2 (Auto-Backup)")
        self.btn_install.setFixedHeight(40)
        self.btn_install.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_install.setStyleSheet(
            """
            QPushButton {
                background-color: #2e7d32;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:disabled {
                background-color: #253326;
                color: #5c6e5e;
            }
            """
        )
        self.btn_install.clicked.connect(self.installRequested.emit)
        layout.addWidget(self.btn_install)

        # 2. Export Button
        self.btn_export = QPushButton("📦 Export as Shareable ZIP")
        self.btn_export.setFixedHeight(36)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export.setStyleSheet(
            """
            QPushButton {
                background-color: #1565c0;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:disabled {
                background-color: #1a2736;
                color: #4a5c70;
            }
            """
        )
        self.btn_export.clicked.connect(self.exportRequested.emit)
        layout.addWidget(self.btn_export)

        # 3. Restore Button
        self.btn_restore = QPushButton("↺ Restore Previous Backup")
        self.btn_restore.setFixedHeight(32)
        self.btn_restore.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_restore.setStyleSheet(
            """
            QPushButton {
                background-color: #2b303c;
                color: #d0d4dc;
                font-size: 11px;
                font-weight: 500;
                border: 1px solid #3c404b;
                border-radius: 6px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #3b4252;
                color: #ffffff;
                border-color: #d83b01;
            }
            """
        )
        self.btn_restore.clicked.connect(self.restoreRequested.emit)
        layout.addWidget(self.btn_restore)

        layout.addStretch()

    def set_busy(self, is_busy: bool):
        self.btn_install.setEnabled(not is_busy)
        self.btn_export.setEnabled(not is_busy)
        self.btn_restore.setEnabled(not is_busy)
