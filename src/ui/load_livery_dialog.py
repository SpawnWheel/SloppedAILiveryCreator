"""
Dialog for loading existing custom liveries from AMS2 Overrides or disk files.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.dds_converter import DDSConverter
from src.core.image_io import load_any_image
from src.core.installer import LiveryInstaller
from src.core.xml_manager import XMLManager
from src.utils.constants import SUPPORTED_IMAGE_EXTS


class LoadLiveryDialog(QDialog):
    """
    Modal dialog allowing users to browse and load an existing livery into the studio,
    either from AMS2 Overrides car XMLs or directly from DDS/PNG/JPG files.
    """

    # (texture, preview, car, id, team_name, meta) - meta: {"base_livery": str, "extra_images": {NAME: Image}}
    liverySelected = Signal(object, object, str, int, str, object)

    def __init__(self, initial_car: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Load Existing Livery to Edit")
        self.resize(600, 450)
        self.setStyleSheet("background-color: #1e2026; color: #ffffff;")

        self.initial_car = initial_car
        self.selected_texture: Optional[Image.Image] = None
        self.selected_preview: Optional[Image.Image] = None
        self.selected_car: str = initial_car
        self.selected_id: int = 51
        self.selected_team: str = "Loaded Livery"

        self._build_ui()
        self._populate_cars()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
                border: 1px solid #3c404b;
                border-radius: 6px;
                background-color: #14161a;
            }
            QTabBar::tab {
                background-color: #232730;
                color: #b0b8c6;
                padding: 6px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #14161a;
                color: #ffffff;
                font-weight: bold;
            }
            """
        )

        # Tab 1: AMS2 Overrides
        tab_ams2 = QWidget()
        ams2_layout = QVBoxLayout(tab_ams2)
        ams2_layout.setContentsMargins(10, 10, 10, 10)
        ams2_layout.setSpacing(8)

        car_row = QHBoxLayout()
        car_row.addWidget(QLabel("Select Car Model:"))
        self.combo_cars = QComboBox()
        self.combo_cars.setStyleSheet(
            "background-color: #1e2026; border: 1px solid #3c404b; padding: 4px; border-radius: 4px; color: #fff;"
        )
        self.combo_cars.currentTextChanged.connect(self._on_car_selected)
        car_row.addWidget(self.combo_cars, 1)
        ams2_layout.addLayout(car_row)

        ams2_layout.addWidget(QLabel("Configured Liveries for this Vehicle:"))
        self.list_liveries = QListWidget()
        self.list_liveries.setStyleSheet(
            """
            QListWidget {
                background-color: #181a20;
                border: 1px solid #2e323b;
                border-radius: 6px;
                padding: 4px;
                color: #e0e0e0;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #232730;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
                border-radius: 4px;
            }
            """
        )
        ams2_layout.addWidget(self.list_liveries)

        self.tabs.addTab(tab_ams2, "Installed in AMS2")

        # Tab 2: Open File
        tab_file = QWidget()
        file_layout = QVBoxLayout(tab_file)
        file_layout.setContentsMargins(10, 10, 10, 10)
        file_layout.setSpacing(10)

        file_prompt = QLabel("Load any raw 2D livery file (.dds, .png, .jpg, .tga):")
        file_prompt.setStyleSheet("color: #b0b8c6; font-size: 11px;")
        file_layout.addWidget(file_prompt)

        pick_row = QHBoxLayout()
        self.edit_file_path = QLineEdit()
        self.edit_file_path.setPlaceholderText("Path to livery image or DDS file...")
        self.edit_file_path.setStyleSheet(
            "background-color: #181a20; border: 1px solid #3c404b; padding: 6px; border-radius: 4px;"
        )
        pick_row.addWidget(self.edit_file_path, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet(
            "background-color: #2b303c; border: 1px solid #3c404b; padding: 6px 14px; border-radius: 4px;"
        )
        btn_browse.clicked.connect(self._browse_file)
        pick_row.addWidget(btn_browse)
        file_layout.addLayout(pick_row)

        # File preview label
        self.lbl_file_info = QLabel("No file selected")
        self.lbl_file_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_file_info.setStyleSheet("color: #606674; font-style: italic; padding: 20px;")
        file_layout.addWidget(self.lbl_file_info, 1)

        self.tabs.addTab(tab_file, "From Image / DDS File")
        layout.addWidget(self.tabs, 1)

        # Dialog Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(
            "background-color: #2b303c; border: 1px solid #3c404b; padding: 6px 14px; border-radius: 4px;"
        )
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        self.btn_load = QPushButton("📂 Load into Livery Studio")
        self.btn_load.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #1084e0;
            }
            """
        )
        self.btn_load.clicked.connect(self._on_load_clicked)
        btn_box.addWidget(self.btn_load)

        layout.addLayout(btn_box)

    def _populate_cars(self):
        cars = LiveryInstaller.list_car_folders()
        self.combo_cars.clear()
        for c in cars:
            self.combo_cars.addItem(c)

        if self.initial_car and self.initial_car in cars:
            self.combo_cars.setCurrentText(self.initial_car)
        elif cars:
            self.combo_cars.setCurrentIndex(0)

    def _on_car_selected(self, car_folder: str):
        self.list_liveries.clear()
        if not car_folder:
            return

        entries = LiveryInstaller.list_installed_liveries(car_folder)
        if not entries:
            item = QListWidgetItem("No custom liveries configured for this car yet.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list_liveries.addItem(item)
            return

        car_dir = LiveryInstaller.get_overrides_path() / car_folder

        for entry in entries:
            lid = entry.get("livery_id", "?")
            name = entry.get("name", "Unnamed")
            tex_path = entry.get("texture_path", "")
            prev_path = entry.get("preview_path", "")

            label = f"#{lid} — {name} ({tex_path})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry)

            # Try loading preview thumbnail for list icon
            if prev_path:
                full_prev = car_dir / prev_path.replace("/", "\\")
                if full_prev.exists():
                    try:
                        prev_img = DDSConverter.load_dds_image(full_prev)
                        rgb = prev_img.convert("RGB")
                        qimg = QImage(rgb.tobytes("raw", "RGB"), rgb.width, rgb.height, QImage.Format.Format_RGB888)
                        item.setIcon(QIcon(QPixmap.fromImage(qimg)))
                    except Exception:
                        pass

            self.list_liveries.addItem(item)

        if self.list_liveries.count() > 0:
            self.list_liveries.setCurrentRow(0)

    def _browse_file(self):
        filter_str = "Livery Files (*.dds *.png *.jpg *.jpeg *.tga *.bmp *.webp *.psd);;All Files (*.*)"
        fpath, _ = QFileDialog.getOpenFileName(self, "Open Existing Livery Texture", "", filter_str)
        if fpath:
            self.edit_file_path.setText(fpath)
            try:
                p = Path(fpath)
                img = load_any_image(p)
                self.selected_texture = img
                w, h = img.size
                self.lbl_file_info.setText(f"✅ Loaded {p.name} ({w} × {h} px)")
                self.lbl_file_info.setStyleSheet("color: #81c784; font-weight: bold; padding: 20px;")
            except Exception as e:
                self.lbl_file_info.setText(f"❌ Error loading file:\n{e}")
                self.lbl_file_info.setStyleSheet("color: #ff8a80; padding: 20px;")

    def _on_load_clicked(self):
        current_tab = self.tabs.currentIndex()

        if current_tab == 0:
            # Installed livery from AMS2
            item = self.list_liveries.currentItem()
            if not item:
                QMessageBox.warning(self, "Select Livery", "Please select a livery from the list.")
                return

            entry = item.data(Qt.ItemDataRole.UserRole)
            if not entry:
                return

            car = self.combo_cars.currentText()
            try:
                lid = int(entry.get("livery_id", 51))
            except ValueError:
                lid = 51
            team = entry.get("name", "Custom Team")

            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    tex_img, prev_img, meta = LiveryInstaller.load_installed_livery(car, lid)
                finally:
                    QApplication.restoreOverrideCursor()
                self.selected_texture = tex_img
                self.selected_preview = prev_img
                self.selected_car = car
                self.selected_id = lid
                self.selected_team = team
                extras = meta.get("extra_images") or {}
                self.liverySelected.emit(
                    tex_img, prev_img, car, lid, team,
                    {"base_livery": meta.get("base_livery", ""), "extra_images": extras},
                )
                self.accept()
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load installed livery:\n{e}")
        else:
            # From file
            if self.selected_texture is None:
                path_str = self.edit_file_path.text().strip()
                if not path_str:
                    QMessageBox.warning(self, "No File", "Please select an image or DDS file.")
                    return
                p = Path(path_str)
                try:
                    self.selected_texture = load_any_image(p)
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Could not open file:\n{e}")
                    return

            self.liverySelected.emit(
                self.selected_texture, None, "", 0, Path(self.edit_file_path.text()).stem, {}
            )
            self.accept()
