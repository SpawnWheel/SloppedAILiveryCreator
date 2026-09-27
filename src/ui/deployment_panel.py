"""
Panel for Section 4: AMS2 Target Car Selection, Metadata, and Real-Time Collision Warning.
"""

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.config import config
from src.core.installer import LiveryInstaller
from src.core.xml_manager import XMLManager


class DeploymentPanel(QGroupBox):
    """Panel for configuring the target AMS2 car model, livery ID, base livery and team metadata."""

    configChanged = Signal()
    profileChanged = Signal(object)  # CarProfile

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__("4. AMS2 TARGET & DEPLOYMENT", parent)
        self.profile = None
        self._build_ui()
        self._populate_car_folders()
        self._reload_profile()
        self.check_id_collision()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 14, 10, 10)

        form_layout = QFormLayout()
        form_layout.setSpacing(8)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # 1. Car Folder Combo Box with Browse button & Autocompletion
        car_layout = QHBoxLayout()
        car_layout.setSpacing(6)

        self.combo_car = QComboBox()
        self.combo_car.setEditable(True)
        self.combo_car.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo_car.setStyleSheet(
            """
            QComboBox {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
            }
            """
        )
        self.combo_car.currentTextChanged.connect(self._on_car_changed)
        car_layout.addWidget(self.combo_car, 1)

        self.btn_browse_car = QPushButton("📁 Browse...")
        self.btn_browse_car.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_car.setStyleSheet(
            """
            QPushButton {
                background-color: #2b303c;
                border: 1px solid #3c404b;
                border-radius: 4px;
                color: #d0d4dc;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3b4252;
                color: #ffffff;
                border-color: #0078d4;
            }
            """
        )
        self.btn_browse_car.clicked.connect(self._browse_car_folder)
        car_layout.addWidget(self.btn_browse_car)

        form_layout.addRow("Car Folder:", car_layout)

        # 2. Livery ID Spinner
        id_layout = QHBoxLayout()
        self.spin_id = QSpinBox()
        self.spin_id.setRange(1, 999)
        self.spin_id.setValue(config.get("last_livery_id", 51))
        self.spin_id.setFixedWidth(80)
        self.spin_id.setStyleSheet(
            """
            QSpinBox {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
            }
            """
        )
        self.spin_id.valueChanged.connect(self._on_id_changed)
        id_layout.addWidget(self.spin_id)
        id_layout.addStretch()

        form_layout.addRow("Livery ID:", id_layout)

        # 2b. Base livery (read from the car's _dist.xml). Anything you don't override - e.g. wheels,
        #     tyres, windows - comes from this base livery.
        self.combo_base = QComboBox()
        self.combo_base.setEditable(True)
        self.combo_base.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.combo_base.setToolTip(
            "The stock AMS2 livery your design is built on (listed in the car's _dist.xml).\n"
            "  • Every texture you don't replace - rims, tyres, windows, some carbon parts - comes from it.\n"
            "  • It sets the paint SHADER: a 'Matte', 'Chrome' or 'Metallic' base gives that paint type.\n\n"
            "The Gloss setting next to the preview is separate: it controls how strongly this base\n"
            "paint's shine shows on your design (alpha channel). 'Default' is the car's normal gloss paint."
        )
        self.combo_base.setStyleSheet(self.combo_car.styleSheet())
        self.combo_base.currentTextChanged.connect(lambda _: self.configChanged.emit())
        form_layout.addRow("Base Livery:", self.combo_base)

        # 3. Team Name
        self.edit_team = QLineEdit(config.get("last_team_name", "Gulf Sim Racing"))
        self.edit_team.setStyleSheet(
            """
            QLineEdit {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
            }
            """
        )
        self.edit_team.textChanged.connect(lambda: self.configChanged.emit())
        form_layout.addRow("Team Name:", self.edit_team)

        # 4. Driver Name
        self.edit_driver = QLineEdit(config.get("last_driver", "D. Born"))
        self.edit_driver.setStyleSheet(
            """
            QLineEdit {
                background-color: #1e2026;
                border: 1px solid #3c404b;
                border-radius: 4px;
                padding: 4px 8px;
                color: #ffffff;
                font-size: 11px;
            }
            """
        )
        self.edit_driver.textChanged.connect(lambda: self.configChanged.emit())
        form_layout.addRow("Driver:", self.edit_driver)

        layout.addLayout(form_layout)

        # 5. Collision Status Warning Badge
        self.collision_badge = QLabel()
        self.collision_badge.setWordWrap(True)
        self.collision_badge.setStyleSheet(
            """
            QLabel {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: 500;
            }
            """
        )
        layout.addWidget(self.collision_badge)

    def _populate_car_folders(self):
        self.combo_car.blockSignals(True)
        self.combo_car.clear()
        cars = LiveryInstaller.list_car_folders()
        for c in cars:
            self.combo_car.addItem(c)

        # Set up searchable completer
        completer = QCompleter(cars, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.combo_car.setCompleter(completer)

        last_car = config.get("last_car_folder", "bmw_m4_gt3")
        if last_car in cars:
            self.combo_car.setCurrentText(last_car)
        elif cars:
            self.combo_car.setCurrentIndex(0)
        self.combo_car.blockSignals(False)

    def _browse_car_folder(self):
        """Allows user to point directly to any car folder or Overrides root on their drive."""
        start_dir = str(LiveryInstaller.get_overrides_path())
        chosen_dir = QFileDialog.getExistingDirectory(
            self, "Select Car Model Folder or Overrides Directory", start_dir
        )
        if not chosen_dir:
            return

        chosen_path = Path(chosen_dir)

        # Case 1: User pointed directly to a car subfolder (e.g. .../Overrides/alpine_a110_gt4_evo)
        parent_dir = chosen_path.parent
        if parent_dir.name.lower() == "overrides" or (parent_dir.exists() and sum(1 for _ in parent_dir.iterdir()) > 10):
            # Update app overrides root to parent
            config.ams2_overrides_path = parent_dir
            self._populate_car_folders()
            self.combo_car.setCurrentText(chosen_path.name.lower())
            self._reload_profile()
            config.set("last_car_folder", chosen_path.name.lower())
        # Case 2: User pointed to the Overrides folder itself
        elif chosen_path.name.lower() == "overrides" or sum(1 for _ in chosen_path.iterdir()) > 5:
            config.ams2_overrides_path = chosen_path
            self._populate_car_folders()
            self._reload_profile()
        # Case 3: Custom external car folder
        else:
            car_name = chosen_path.name.lower()
            if self.combo_car.findText(car_name) == -1:
                self.combo_car.addItem(car_name)
            self.combo_car.setCurrentText(car_name)
            config.set("last_car_folder", car_name)

        self.check_id_collision()
        self.configChanged.emit()

    def _on_car_changed(self, text: str):
        config.set("last_car_folder", text)
        self._reload_profile()
        self.check_id_collision()
        self.configChanged.emit()

    def _reload_profile(self):
        """Reads the car's _dist.xml to list its base liveries and supported textures."""
        car = self.get_car_folder()
        if not car:
            return
        try:
            self.profile = LiveryInstaller.get_car_profile(car)
        except Exception as e:
            print(f"[DeploymentPanel] profile error: {e}")
            return
        previous = self.combo_base.currentText()
        self.combo_base.blockSignals(True)
        self.combo_base.clear()
        for b in self.profile.base_liveries:
            self.combo_base.addItem(b)
        if previous and previous in self.profile.base_liveries:
            self.combo_base.setCurrentText(previous)
        else:
            self.combo_base.setCurrentIndex(0)
        self.combo_base.blockSignals(False)
        self.profileChanged.emit(self.profile)

    def get_base_livery(self) -> str:
        return self.combo_base.currentText().strip() or (
            self.profile.default_base_livery if self.profile else "Default"
        )

    def set_base_livery(self, name: str) -> None:
        if not name:
            return
        if self.combo_base.findText(name) == -1:
            self.combo_base.addItem(name)
        self.combo_base.setCurrentText(name)

    def _on_id_changed(self, val: int):
        config.set("last_livery_id", val)
        self.check_id_collision()
        self.configChanged.emit()

    def check_id_collision(self):
        """Scans the car's XML override and updates the collision warning badge."""
        car_folder = self.get_car_folder()
        livery_id = self.get_livery_id()

        if not car_folder:
            self.collision_badge.hide()
            return

        xml_path = LiveryInstaller.get_car_xml_path(car_folder)
        collision = XMLManager.check_collision(xml_path, livery_id)

        self.collision_badge.show()
        if collision:
            occ_name = collision.get("name", "Unknown Team")
            self.collision_badge.setText(
                f"⚠️ Slot #{livery_id} is currently used by '{occ_name}' — will be safely replaced"
            )
            self.collision_badge.setStyleSheet(
                """
                QLabel {
                    background-color: #3b2a1a;
                    color: #ffb74d;
                    border: 1px solid #7c4d1e;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 10px;
                    font-weight: bold;
                }
                """
            )
        else:
            self.collision_badge.setText(f"✅ Slot #{livery_id} is available for '{car_folder}'")
            self.collision_badge.setStyleSheet(
                """
                QLabel {
                    background-color: #1a2e1e;
                    color: #81c784;
                    border: 1px solid #2e5934;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 10px;
                }
                """
            )

    def get_car_folder(self) -> str:
        return self.combo_car.currentText().strip()

    def get_livery_id(self) -> int:
        return self.spin_id.value()

    def get_team_name(self) -> str:
        return self.edit_team.text().strip() or "Custom Sim Racing"

    def get_driver(self) -> str:
        return self.edit_driver.text().strip() or "Sim Driver"
