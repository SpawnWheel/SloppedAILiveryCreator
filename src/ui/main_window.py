"""
Main Application Window for AMS2 AI Livery Studio.
Integrates all panels, async worker threads, settings, and menus.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.ai_engine import AIEngine
from src.core.config import config
from src.core.exporter import LiveryExporter
from src.core.installer import LiveryInstaller
from src.ui.actions_panel import ActionsPanel, BackupRestoreDialog
from src.ui.deployment_panel import DeploymentPanel
from src.ui.editor.livery_editor import LiveryPaintEditor, PaintEditorWidget
from src.ui.editor.theme import MAIN_TABS_QSS
from src.ui.load_livery_dialog import LoadLiveryDialog
from src.ui.preview_panel import LiveryRevision, PreviewPanel
from src.ui.qt_image import pil_to_qpixmap
from src.ui.prompt_dialog import PromptEditorDialog
from src.ui.prompt_pack_dialog import PromptPackDialog, copy_pack_to_clipboard
from src.ui.reference_panel import ReferencePanel
from src.ui.widgets.status_bar import CustomStatusBar
from src.ui.wheels_panel import WheelsPanel
from src.ui.wireframe_panel import WireframePanel
from src.utils.constants import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_AI_MODELS,
    DEFAULT_AI_MODEL,
    FALLBACK_AI_MODEL,
    SUPPORTED_LIVERY_INSTALL_EXTS,
    TEXTURE_SPECS,
)
from src.utils.file_helpers import cleanup_dir, get_cache_dir, slugify


class SettingsDialog(QDialog):
    """Configuration dialog for API Keys, AI Models, and AMS2 installation paths."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Settings — AMS2 AI Livery Studio")
        self.setMinimumWidth(520)
        self.setStyleSheet("background-color: #1e2026; color: #ffffff;")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        form = QFormLayout()
        form.setSpacing(10)

        # Gemini API Key Input
        self.key_edit = QLineEdit(config.api_key)
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("Enter Google AI Studio API Key (AIzaSy...)")
        self.key_edit.setStyleSheet(
            "background-color: #14161a; border: 1px solid #3c404b; padding: 6px; border-radius: 4px;"
        )
        form.addRow("Google AI API Key:", self.key_edit)

        # AI Generation Model
        self.model_combo = QComboBox()
        for m in AVAILABLE_AI_MODELS:
            self.model_combo.addItem(m)
        current_model = config.get("ai_model", DEFAULT_AI_MODEL)
        self.model_combo.setCurrentText(current_model)
        self.model_combo.setStyleSheet(
            "background-color: #14161a; border: 1px solid #3c404b; padding: 6px; border-radius: 4px; color: #ffffff;"
        )
        form.addRow("AI Generation Model:", self.model_combo)

        # AMS2 Overrides Path
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit(str(config.ams2_overrides_path))
        self.path_edit.setStyleSheet(
            "background-color: #14161a; border: 1px solid #3c404b; padding: 6px; border-radius: 4px;"
        )
        path_layout.addWidget(self.path_edit, 1)

        btn_browse = QPushButton("Browse...")
        btn_browse.setStyleSheet(
            "background-color: #2b303c; border: 1px solid #3c404b; padding: 6px 12px; border-radius: 4px;"
        )
        btn_browse.clicked.connect(self._browse_path)
        path_layout.addWidget(btn_browse)

        form.addRow("AMS2 Overrides Path:", path_layout)
        layout.addLayout(form)

        # Action Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(
            "background-color: #2b303c; border: 1px solid #3c404b; padding: 6px 14px; border-radius: 4px;"
        )
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QPushButton("Save Settings")
        btn_save.setStyleSheet(
            "background-color: #0078d4; color: white; font-weight: bold; border: none; padding: 6px 16px; border-radius: 4px;"
        )
        btn_save.clicked.connect(self._save)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

    def _browse_path(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Automobilista 2 CustomLiveries Overrides Folder", self.path_edit.text()
        )
        if folder:
            self.path_edit.setText(folder)

    def _save(self):
        config.api_key = self.key_edit.text().strip()
        config.set("ai_model", self.model_combo.currentText())
        config.ams2_overrides_path = Path(self.path_edit.text().strip())
        self.accept()


# ==============================================================================
# Background Worker Threads
# ==============================================================================


class GenerationWorker(QThread):
    """Background worker for AI generation to prevent UI freezes."""

    finished = Signal(object, str)  # (PIL.Image, template-fit report)
    error = Signal(str)

    def __init__(
        self,
        ai_engine: AIEngine,
        wireframe: Optional[Image.Image],
        references: List[Tuple[Image.Image, str]],
        prompt: str = "",
        system_instruction: Optional[str] = None,
        target_resolution: int = 4096,
    ):
        super().__init__()
        self.ai_engine = ai_engine
        self.wireframe = wireframe
        self.references = references
        self.prompt = prompt
        self.system_instruction = system_instruction
        self.target_resolution = target_resolution

    def run(self):
        try:
            img = self.ai_engine.generate_livery(
                wireframe=self.wireframe,
                references=self.references,
                prompt=self.prompt,
                system_instruction=self.system_instruction,
                target_resolution=self.target_resolution,
            )
            img, report = fit_to_template(img, self.wireframe)
            notes = list(getattr(self.ai_engine, "last_notes", []) or [])
            if notes:
                report = "; ".join([report] + notes) if report else "; ".join(notes)
            self.finished.emit(img, report)
        except Exception as e:
            self.error.emit(str(e))


def fit_to_template(img: Image.Image, wireframe: Optional[Image.Image],
                    previous: Optional[Image.Image] = None):
    """Applies the 'Lock to template' post-processing (auto-align + clip to UV islands) if enabled."""
    if not config.get("lock_to_template", True) or (wireframe is None and previous is None):
        return img, ""
    try:
        from src.core.template_fit import lock_to_template

        out, report = lock_to_template(img, wireframe, previous=previous,
                                       do_clip=config.get("clip_to_islands", True))
        if out.size != img.size:
            out = out.resize(img.size, Image.Resampling.LANCZOS)
        return out, report.summary()
    except Exception as e:  # never lose a paid-for generation over post-processing
        print(f"[TemplateFit] skipped: {e}")
        return img, ""


class RefinementWorker(QThread):
    """Background worker for iterative livery refinement with Gemini multimodal generation."""

    finished = Signal(object, str)  # (PIL.Image, template-fit report)
    error = Signal(str)

    def __init__(
        self,
        ai_engine: AIEngine,
        current_livery: Image.Image,
        annotated_livery: Optional[Image.Image],
        notes: str,
        wireframe: Optional[Image.Image],
        references: List[Tuple[Image.Image, str]],
        finish: str,
        iteration: int = 2,
        target_resolution: int = 4096,
    ):
        super().__init__()
        self.ai_engine = ai_engine
        self.current_livery = current_livery
        self.annotated_livery = annotated_livery
        self.notes = notes
        self.wireframe = wireframe
        self.references = references
        self.finish = finish
        self.iteration = iteration
        self.target_resolution = target_resolution

    def run(self):
        try:
            if self.ai_engine.is_configured():
                img = self.ai_engine.refine_livery(
                    current_livery=self.current_livery,
                    annotated_livery=self.annotated_livery,
                    notes=self.notes,
                    wireframe=self.wireframe,
                    references=self.references,
                    finish=self.finish,
                    target_resolution=self.target_resolution,
                )
            else:
                img = self.ai_engine.refine_mock_livery(
                    current_livery=self.current_livery,
                    annotated_livery=self.annotated_livery,
                    notes=self.notes,
                    finish=self.finish,
                    iteration=self.iteration,
                    target_resolution=self.target_resolution,
                )
            img, report = fit_to_template(img, self.wireframe, previous=self.current_livery)
            self.finished.emit(img, report)
        except Exception as e:
            self.error.emit(str(e))


class InstallWorker(QThread):
    """Background worker for livery processing, DDS compression, and AMS2 install."""

    success = Signal(object, object)  # (Path, Optional[Path])
    error = Signal(str)

    def __init__(
        self,
        car_folder: str,
        livery_id: int,
        team_name: str,
        driver: str,
        texture_image: Image.Image,
        finish_type: str,
        custom_preview_image: Optional[Image.Image] = None,
        base_livery: str = "Default",
        extra_textures: Optional[Dict[str, Image.Image]] = None,
    ):
        super().__init__()
        self.car_folder = car_folder
        self.livery_id = livery_id
        self.team_name = team_name
        self.driver = driver
        self.texture_image = texture_image
        self.finish_type = finish_type
        self.custom_preview_image = custom_preview_image
        self.base_livery = base_livery
        self.extra_textures = extra_textures or {}

    def run(self):
        try:
            target_dir, backup = LiveryInstaller.install_livery(
                car_folder_name=self.car_folder,
                livery_id=self.livery_id,
                team_name=self.team_name,
                driver=self.driver,
                texture_image=self.texture_image,
                finish_type=self.finish_type,
                custom_preview_image=self.custom_preview_image,
                base_livery=self.base_livery,
                extra_textures=self.extra_textures,
            )
            self.success.emit(target_dir, backup)
        except Exception as e:
            self.error.emit(str(e))


class ExportWorker(QThread):
    """Background worker for ZIP package generation."""

    success = Signal(object)  # Path
    error = Signal(str)

    def __init__(
        self,
        car_folder: str,
        livery_id: int,
        team_name: str,
        driver: str,
        texture_image: Image.Image,
        output_zip_path: Path,
        finish_type: str,
        custom_preview_image: Optional[Image.Image] = None,
        base_livery: str = "Default",
        extra_textures: Optional[Dict[str, Image.Image]] = None,
    ):
        super().__init__()
        self.base_livery = base_livery
        self.extra_textures = extra_textures or {}
        self.car_folder = car_folder
        self.livery_id = livery_id
        self.team_name = team_name
        self.driver = driver
        self.texture_image = texture_image
        self.output_zip_path = output_zip_path
        self.finish_type = finish_type
        self.custom_preview_image = custom_preview_image

    def run(self):
        try:
            out_path = LiveryExporter.export_to_zip(
                car_folder_name=self.car_folder,
                livery_id=self.livery_id,
                team_name=self.team_name,
                driver=self.driver,
                texture_image=self.texture_image,
                output_zip_path=self.output_zip_path,
                finish_type=self.finish_type,
                custom_preview_image=self.custom_preview_image,
                base_livery=self.base_livery,
                extra_textures=self.extra_textures,
            )
            self.success.emit(out_path)
        except Exception as e:
            self.error.emit(str(e))


class QuickInstallWorker(QThread):
    """Background worker for installing pre-made livery files directly without AI."""

    success = Signal(object, object)  # (Path, Optional[Path])
    error = Signal(str)

    def __init__(
        self,
        file_path: Path,
        car_folder: str,
        livery_id: int,
        team_name: str,
        driver: str,
        finish_type: str,
        custom_preview_image: Optional[Image.Image] = None,
        base_livery: str = "Default",
    ):
        super().__init__()
        self.base_livery = base_livery
        self.file_path = file_path
        self.car_folder = car_folder
        self.livery_id = livery_id
        self.team_name = team_name
        self.driver = driver
        self.finish_type = finish_type
        self.custom_preview_image = custom_preview_image

    def run(self):
        try:
            target_dir, backup = LiveryInstaller.install_livery_from_file(
                file_path=self.file_path,
                car_folder_name=self.car_folder,
                livery_id=self.livery_id,
                team_name=self.team_name,
                driver=self.driver,
                finish_type=self.finish_type,
                custom_preview_image=self.custom_preview_image,
                base_livery=self.base_livery,
            )
            self.success.emit(target_dir, backup)
        except Exception as e:
            self.error.emit(str(e))


# ==============================================================================
# Main Window Class
# ==============================================================================


class MainWindow(QMainWindow):
    """Main window for AMS2 AI Livery Studio."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1200, 820)
        self.setMinimumSize(950, 700)

        self.ai_engine = AIEngine()
        self._gen_worker: Optional[GenerationWorker] = None
        self._refine_worker: Optional[RefinementWorker] = None
        self._install_worker: Optional[InstallWorker] = None
        self._quick_install_worker: Optional[QuickInstallWorker] = None
        self._export_worker: Optional[ExportWorker] = None

        self._build_menu()
        self._build_ui()
        self._check_initial_setup()

    def _build_menu(self):
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        action_open_livery = QAction("&Open Existing Livery...", self)
        action_open_livery.setToolTip("Load an existing livery from AMS2 Overrides or disk to edit and refine")
        action_open_livery.triggered.connect(self._open_load_livery_dialog)
        file_menu.addAction(action_open_livery)

        action_pack = QAction("&Copy Prompt + Images for Other AI...", self)
        action_pack.setShortcut("Ctrl+Shift+C")
        action_pack.triggered.connect(self._on_create_prompt_pack)
        file_menu.addAction(action_pack)

        action_import_ext = QAction("&Import Result from Other AI...", self)
        action_import_ext.triggered.connect(lambda: self._import_external_result())
        file_menu.addAction(action_import_ext)
        file_menu.addSeparator()

        action_settings = QAction("&Settings / API Key...", self)
        action_settings.triggered.connect(self._open_settings)
        file_menu.addAction(action_settings)

        file_menu.addSeparator()

        action_exit = QAction("E&xit", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")

        action_open_ams2 = QAction("Open &AMS2 Overrides Folder", self)
        action_open_ams2.triggered.connect(self._open_overrides_folder)
        tools_menu.addAction(action_open_ams2)

        action_paint = QAction("Open Current Version in &Paint Editor", self)
        action_paint.triggered.connect(self._open_paint_editor)
        tools_menu.addAction(action_paint)
        tools_menu.addSeparator()

        action_align = QAction("Re-&align Current Version to Template", self)
        action_align.setToolTip("Snap a drifted AI result back onto the template's panels")
        action_align.triggered.connect(self._realign_current)
        tools_menu.addAction(action_align)

        self.action_lock = QAction("&Lock AI Results to Template (auto-align)", self, checkable=True)
        self.action_lock.setChecked(config.get("lock_to_template", True))
        self.action_lock.toggled.connect(lambda v: config.set("lock_to_template", v))
        tools_menu.addAction(self.action_lock)

        self.action_clip = QAction("&Clip AI Paint to UV Islands", self, checkable=True)
        self.action_clip.setChecked(config.get("clip_to_islands", True))
        self.action_clip.toggled.connect(lambda v: config.set("clip_to_islands", v))
        tools_menu.addAction(self.action_clip)

        self.action_orient = QAction("&Fix Rotated Panels (extra AI call per rotated panel)", self, checkable=True)
        self.action_orient.setToolTip(
            "For templates whose placeholder numbers show a panel is stored sideways/upside-down (PSD), repaint that "
            "panel upright in a second pass and rotate it back into place")
        self.action_orient.setChecked(config.get("orientation_pass", True))
        self.action_orient.toggled.connect(lambda v: config.set("orientation_pass", v))
        tools_menu.addAction(self.action_orient)
        tools_menu.addSeparator()

        action_clear_cache = QAction("&Clear Template Cache", self)
        action_clear_cache.triggered.connect(self._clear_cache)
        tools_menu.addAction(action_clear_cache)

        # Go Menu: the main tabs
        go_menu = menu_bar.addMenu("&Go")
        self._tab_actions = []
        for i, name in enumerate(["&AI Generation", "&Paint Editor", "&Wheels / Rims", "&Install && Export"]):
            act = QAction(name, self)
            act.setShortcut(QKeySequence(f"Alt+{i + 1}"))
            act.triggered.connect(lambda _=False, idx=i: self.tabs.setCurrentIndex(idx))
            go_menu.addAction(act)
            self._tab_actions.append(act)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        self._help_menu = help_menu
        action_wheels = QAction("How do &wheels / rims work?", self)
        action_wheels.triggered.connect(self._show_wheels_help)
        help_menu.addAction(action_wheels)
        action_about = QAction("&About", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)

    def _build_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 4)
        main_layout.setSpacing(0)

        # Document-style tabs across the top, like Photoshop / GIMP
        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(MAIN_TABS_QSS)
        main_layout.addWidget(self.tabs, 1)

        # ---------------------------------------------------------------- tab 1: AI Generation
        ai_tab = QWidget()
        ai_layout = QHBoxLayout(ai_tab)
        ai_layout.setContentsMargins(10, 10, 10, 6)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(8)

        self.wireframe_panel = WireframePanel(self)
        self.wireframe_panel.wireframeLoaded.connect(self._on_wireframe_loaded)
        left_layout.addWidget(self.wireframe_panel)

        self.reference_panel = ReferencePanel(self)
        self.reference_panel.openPromptEditorRequested.connect(self._open_prompt_editor)
        left_layout.addWidget(self.reference_panel, 1)
        splitter.addWidget(left_container)

        self.preview_panel = PreviewPanel(self)
        self.preview_panel.generateRequested.connect(self._on_generate_livery)
        self.preview_panel.promptPackRequested.connect(self._on_create_prompt_pack)
        self.preview_panel.refineRequested.connect(self._on_refine_livery)
        self.preview_panel.commitRequested.connect(self._on_commit_livery)
        self.preview_panel.loadLiveryRequested.connect(self._open_load_livery_dialog)
        self.preview_panel.paintEditorRequested.connect(self._open_paint_editor)
        self.preview_panel.revisionChanged.connect(self._on_revision_changed)
        splitter.addWidget(self.preview_panel)
        splitter.setSizes([480, 720])
        ai_layout.addWidget(splitter)
        self.tabs.addTab(ai_tab, "✦  AI Generation")

        # ---------------------------------------------------------------- tab 2: Paint Editor
        self.paint_editor = PaintEditorWidget(self, embedded=True)
        self.paint_editor.applied.connect(self._on_editor_applied)
        self.paint_editor.openLiveryRequested.connect(self._open_load_livery_dialog)
        self.paint_editor.useCurrentVersionRequested.connect(lambda: self._sync_editor(force=True))
        self.paint_editor.guide_provider = lambda: self.wireframe_panel.get_wireframe()
        self.tabs.addTab(self.paint_editor, "🖌  Paint Editor")
        self._editor_menu_actions = [self.menuBar().insertMenu(self._help_menu.menuAction(), m)
                                     for m in self.paint_editor.menus()]
        self._applying_edit = False

        # ---------------------------------------------------------------- tab 3: Wheels
        self.wheels_panel = WheelsPanel(
            ai_engine=self.ai_engine,
            body_provider=self.preview_panel.get_generated_image,
            parent=self,
        )
        self.wheels_panel.statusMessage.connect(self.status_bar_message)
        self.wheels_panel.busyChanged.connect(self._on_wheels_busy)
        self.wheels_panel.paintEditorRequested.connect(self._open_wheel_editor)
        wheels_tab = QWidget()
        wl = QVBoxLayout(wheels_tab)
        wl.setContentsMargins(10, 10, 10, 6)
        wl.addWidget(self.wheels_panel)
        self.tabs.addTab(wheels_tab, "🛞  Wheels / Rims")

        # ---------------------------------------------------------------- tab 4: Install & Export
        install_tab = QWidget()
        il = QHBoxLayout(install_tab)
        il.setContentsMargins(10, 10, 10, 6)
        il.setSpacing(10)
        left = QVBoxLayout()
        self.deployment_panel = DeploymentPanel(self)
        left.addWidget(self.deployment_panel)
        left.addStretch()
        il.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(self._build_install_summary())
        self.actions_panel = ActionsPanel(self)
        self.actions_panel.installRequested.connect(self._on_install_livery)
        self.actions_panel.quickInstallRequested.connect(self._on_quick_install)
        self.actions_panel.exportRequested.connect(self._on_export_zip)
        self.actions_panel.restoreRequested.connect(self._on_restore_backup)
        right.addWidget(self.actions_panel)
        right.addStretch()
        il.addLayout(right, 1)
        self.tabs.addTab(install_tab, "📦  Install && Export")

        # Keep the Gloss control's "on '<base>' base paint" hint in sync with the Base Livery
        self.deployment_panel.configChanged.connect(
            lambda: self.preview_panel.set_base_livery_hint(self.deployment_panel.get_base_livery())
        )
        self.deployment_panel.configChanged.connect(self._refresh_install_summary)
        self.deployment_panel.profileChanged.connect(
            lambda _p: self.preview_panel.set_base_livery_hint(self.deployment_panel.get_base_livery())
        )
        self.preview_panel.set_base_livery_hint(self.deployment_panel.get_base_livery())

        # Corner of the tab bar: current version + one-click install, visible from every tab
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 2, 8, 2)
        cl.setSpacing(8)
        self.lbl_corner_version = QLabel("No livery yet")
        self.lbl_corner_version.setStyleSheet("color: #8c9099; font-size: 11px;")
        cl.addWidget(self.lbl_corner_version)
        self.btn_corner_install = QPushButton("💾 Install to AMS2")
        self.btn_corner_install.setToolTip("Install the current version (car, slot and team are set in the "
                                           "Install & Export tab)")
        self.btn_corner_install.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border: none;"
            " border-radius: 3px; padding: 4px 12px; font-size: 11px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #2a2c30; color: #5d6168; }")
        self.btn_corner_install.clicked.connect(self._on_install_livery)
        self.btn_corner_install.setEnabled(False)
        cl.addWidget(self.btn_corner_install)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed(0)

        # Bottom Custom Status Bar
        self.status_bar = CustomStatusBar(self)
        main_layout.addWidget(self.status_bar)

        self.deployment_panel.profileChanged.connect(self.wheels_panel.set_profile)
        self.wheels_panel.set_profile(self.deployment_panel.profile)

    def _build_install_summary(self) -> QGroupBox:
        box = QGroupBox("VERSION TO INSTALL")
        v = QHBoxLayout(box)
        v.setContentsMargins(12, 14, 12, 12)
        v.setSpacing(12)
        self.lbl_install_thumb = QLabel()
        self.lbl_install_thumb.setFixedSize(180, 180)
        self.lbl_install_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_install_thumb.setStyleSheet("background: #14161a; border: 1px solid #2e323b; border-radius: 4px;"
                                             " color: #555c68;")
        v.addWidget(self.lbl_install_thumb)
        self.lbl_install_info = QLabel()
        self.lbl_install_info.setWordWrap(True)
        self.lbl_install_info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_install_info.setStyleSheet("color: #b8beca; font-size: 11px;")
        v.addWidget(self.lbl_install_info, 1)
        return box

    def _refresh_install_summary(self) -> None:
        if not hasattr(self, "lbl_install_thumb"):
            return
        rev = self.preview_panel.current_revision()
        if rev is None:
            self.lbl_install_thumb.setPixmap(QPixmap())
            self.lbl_install_thumb.setText("No livery")
            self.lbl_install_info.setText("Generate a livery (AI Generation tab), paint one (Paint Editor tab) or "
                                          "load an existing one first.")
            return
        thumb = rev.image.copy()
        thumb.thumbnail((178, 178))
        self.lbl_install_thumb.setPixmap(pil_to_qpixmap(thumb))
        dp = self.deployment_panel
        state = "✅ committed" if rev.committed else "draft (not committed yet)"
        edits = ""
        if getattr(self, "paint_editor", None) is not None and self.paint_editor.is_dirty():
            edits = "<br><span style='color:#ffb74d'>⚠ The Paint Editor has edits that aren't applied yet - " \
                    "you'll be asked to apply them before installing.</span>"
        wheels = " + custom rims" if self.wheels_panel.get_extra_textures() else ""
        self.lbl_install_info.setText(
            f"<b style='color:white; font-size:13px'>{rev.label}</b> &nbsp; <span>{state}</span><br><br>"
            f"Car: <b>{dp.get_car_folder() or '—'}</b><br>"
            f"Slot: <b>#{dp.get_livery_id()}</b> &nbsp; Team: <b>{dp.get_team_name() or '—'}</b><br>"
            f"Base livery: <b>{dp.get_base_livery()}</b> &nbsp; Gloss: <b>{self.preview_panel.get_finish()}</b>"
            f"<br>Texture: {rev.image.size[0]} × {rev.image.size[1]} px{wheels}{edits}"
        )

    # ==========================================================================
    # Tabs + Paint Editor sync
    # ==========================================================================

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        on_paint = widget is self.paint_editor
        for act in self._editor_menu_actions:
            act.setVisible(on_paint)
        if on_paint:
            self._sync_editor()
        elif index == 2:
            self.wheels_panel.refresh_palette()
        if index == 3:
            self._refresh_install_summary()

    def _on_revision_changed(self) -> None:
        rev = self.preview_panel.current_revision()
        self.lbl_corner_version.setText(f"Current: {rev.label}" if rev else "No livery yet")
        self.btn_corner_install.setEnabled(rev is not None)
        self.paint_editor.set_current_version_available(rev.label if rev else None)
        self._refresh_install_summary()
        if not self._applying_edit and self.tabs.currentWidget() is self.paint_editor:
            self._sync_editor()

    def _sync_editor(self, force: bool = False) -> None:
        """Opens the current version in the Paint tab unless it's already there (asks before dropping edits)."""
        pe = self.paint_editor
        rev = self.preview_panel.current_revision()
        pe.set_current_version_available(rev.label if rev else None)
        if rev is None:
            return
        if pe.source_key is rev and pe.has_content():
            return
        if pe.has_content() and not force and pe.source_key in ("blank", "project") and pe.is_dirty():
            return  # the user is painting something of their own - don't replace it
        if pe.is_dirty():
            current = pe.source_key.label if isinstance(pe.source_key, LiveryRevision) else "your canvas"
            res = QMessageBox.question(
                self, "Unapplied paint edits",
                f"The Paint Editor still has unapplied edits on {current}.\n\n"
                f"Open {rev.label} instead and discard those edits?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Discard:
                return
        guide = self.wireframe_panel.get_wireframe()
        pe.load(rev.image, guide_image=guide, project=rev.project, source_key=rev)
        if guide is None and rev.project is None:
            pe._flash("No template loaded - load the car's template in the AI Generation tab (or the Template "
                      "dock) to see and select its panels.")
        else:
            pe._flash(f"Editing {rev.label}. Press W and click panels to paint inside them.")

    def _on_editor_applied(self, img: Image.Image) -> None:
        pe = self.paint_editor
        pp = self.preview_panel
        src = pe.source_key
        if isinstance(src, LiveryRevision) and img.size != src.image.size:
            img = img.resize(src.image.size, Image.Resampling.LANCZOS)
        n = len(pp.revisions) + 1
        label = f"v{n} (Hand-painted)"
        self._applying_edit = True
        try:
            if pp.revisions:
                pp.add_revision(img, notes="Hand-painted in Paint Editor", label=label, project=pe.project_data())
            else:
                pp.reset_revisions(img, label=label)
                pp.revisions[0].project = pe.project_data()
        finally:
            self._applying_edit = False
        pe.mark_clean(source_key=pp.current_revision())
        self.wheels_panel.refresh_palette()
        self._refresh_install_summary()
        pe._flash(f"Applied as {label} - it's now the current version. Keep painting, or install it from the "
                  "Install & Export tab.")
        self.status_bar.show_info(f"Hand-painted version {label} added.")

    def _confirm_editor_applied(self) -> bool:
        """Before install / export: offer to apply unapplied Paint Editor edits first."""
        pe = self.paint_editor
        if not pe.is_dirty():
            return True
        res = QMessageBox.question(
            self, "Unapplied paint edits",
            "The Paint Editor has edits that haven't been applied to a version yet.\n\n"
            "Apply them as a new version first? (No = use the current version without them)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if res == QMessageBox.StandardButton.Cancel:
            return False
        if res == QMessageBox.StandardButton.Yes:
            pe._apply()
        return True

    def status_bar_message(self, msg: str):
        self.status_bar.show_info(msg)

    def _on_wheels_busy(self, busy: bool, msg: str):
        self.actions_panel.set_busy(busy)
        if busy:
            self.status_bar.show_busy(msg)
        else:
            self.status_bar.show_info("Ready")

    def _check_initial_setup(self):
        """Notifies user if API key is needed."""
        if not self.ai_engine.is_configured():
            self.status_bar.show_info(
                "Gemini API key is not configured. Click Settings to set up your Google AI Studio key."
            )

    def _open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.ai_engine.set_api_key(config.api_key)
            self.deployment_panel._populate_car_folders()
            self.status_bar.show_info("Settings updated successfully.")

    def _open_overrides_folder(self):
        path = config.ams2_overrides_path
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))
        else:
            subprocess.run(["xdg-open", str(path)])

    def _clear_cache(self):
        cache_dir = get_cache_dir()
        cleanup_dir(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.status_bar.show_info("Template cache cleared.")
        QMessageBox.information(self, "Cache Cleared", "Offline template cache has been cleaned.")

    def _show_wheels_help(self):
        from src.ui.wheels_panel import HOW_IT_WORKS

        self.tabs.setCurrentIndex(2)
        QMessageBox.information(self, "Wheels & rims in AMS2", HOW_IT_WORKS)

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "AI-Powered Livery Studio for Automobilista 2.<br>"
            "Generates 4K game-ready DDS textures (BC3/DXT5 + Mipmaps) with Alpha-channel reflectivity shaders "
            "and injects them into AMS2's XML override system with automatic backup and rollback.<br><br>"
            "Built with PySide6, Pillow, DirectXTex texconv, and Google Gemini image models.",
        )

    # ==========================================================================
    # Generation Workflow
    # ==========================================================================

    def _on_wireframe_loaded(self, img: Image.Image, name: str):
        """Passes loaded wireframe into the preview viewport (overlay) and the Paint Editor (panel guide)."""
        self.preview_panel.set_wireframe(img)
        self.paint_editor.set_guide_image(img)

    def _assemble_current_prompts(self) -> Tuple[str, str]:
        """Synthesizes structured user prompt and system instruction strictly from references and style text."""
        ref_roles = [role for _, role in self.reference_panel.get_references()]
        user_prompt = AIEngine.build_default_user_prompt(
            car_name=self.deployment_panel.get_car_folder(),
            template_name=self.wireframe_panel.current_name,
            user_guidance=self.reference_panel.get_prompt(),
            reference_roles=ref_roles,
            finish=self.preview_panel.get_finish(),
        )
        sys_instruction = AIEngine.build_default_system_instruction()
        return user_prompt, sys_instruction

    def _open_prompt_editor(self):
        """Opens interactive dialog allowing the user to view and edit the model prompt and instructions."""
        custom = self.reference_panel.get_custom_prompt()
        if custom:
            user_prompt, sys_inst = custom
        else:
            user_prompt, sys_inst = self._assemble_current_prompts()

        dialog = PromptEditorDialog(
            current_user_prompt=user_prompt,
            current_system_instruction=sys_inst,
            template_name=self.wireframe_panel.current_name,
            wireframe=self.wireframe_panel.get_wireframe(),
            references=self.reference_panel.get_references(),
            regenerate_callback=self._assemble_current_prompts,
            parent=self,
        )
        dialog.promptApplied.connect(self._on_custom_prompt_applied)
        dialog.exec()

    def _on_custom_prompt_applied(self, user_prompt: str, sys_inst: str):
        self.reference_panel.set_custom_prompt(user_prompt, sys_inst)
        self.status_bar.show_info("Custom prompt applied and saved for generation.")

    def _on_generate_livery(self):
        wireframe = self.wireframe_panel.get_wireframe()
        references = self.reference_panel.get_references()
        guidance = self.reference_panel.get_prompt()

        if not wireframe and not guidance and not references:
            QMessageBox.warning(
                self,
                "Input Required",
                "Please load a wireframe template, enter a design prompt, or upload reference images before generating.",
            )
            return

        # Use customized prompt if the user edited it via the dialog, otherwise assemble automatically
        custom = self.reference_panel.get_custom_prompt()
        if custom:
            effective_user_prompt, effective_sys_instruction = custom
        else:
            effective_user_prompt, effective_sys_instruction = self._assemble_current_prompts()

        if not self.ai_engine.is_configured():
            res = QMessageBox.question(
                self,
                "API Key Required",
                "No Gemini API key is configured.\nWould you like to enter your Google AI Studio API key now?\n\n"
                "(Click 'No' to generate a mock test livery for preview testing)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if res == QMessageBox.StandardButton.Yes:
                self._open_settings()
                if not self.ai_engine.is_configured():
                    return
            elif res == QMessageBox.StandardButton.No:
                # Generate Mock Texture for offline test
                self.status_bar.show_busy("Generating test livery pattern...")
                mock_img = AIEngine.generate_mock_livery(
                    wireframe=wireframe,
                    prompt=guidance,
                    finish=self.preview_panel.get_finish(),
                    team_name=self.deployment_panel.get_team_name(),
                    driver=self.deployment_panel.get_driver(),
                    livery_id=self.deployment_panel.get_livery_id(),
                    target_resolution=4096,
                )
                self._on_generation_finished(mock_img)
                return
            else:
                return

        active_model = config.get("ai_model", DEFAULT_AI_MODEL)
        self._set_busy(True, f"Generating 4K Livery Texture with {active_model}...")

        self._gen_worker = GenerationWorker(
            ai_engine=self.ai_engine,
            wireframe=wireframe,
            references=references,
            prompt=effective_user_prompt,
            system_instruction=effective_sys_instruction,
            target_resolution=4096,
        )
        self._gen_worker.finished.connect(self._on_generation_finished)
        self._gen_worker.error.connect(self._on_generation_error)
        self._gen_worker.start()

    # ==========================================================================
    # Prompt Pack: use another AI website instead of the built-in Gemini call
    # ==========================================================================

    def _on_create_prompt_pack(self):
        """Saves prompt + template + references to a folder and copies them to the clipboard."""
        from src.core.prompt_pack import create_prompt_pack

        wireframe = self.wireframe_panel.get_wireframe()
        references = self.reference_panel.get_references()
        guidance = self.reference_panel.get_prompt()
        if not wireframe and not guidance and not references:
            QMessageBox.warning(
                self,
                "Input Required",
                "Please load a wireframe template, enter a design prompt, or upload reference images first.",
            )
            return

        custom = self.reference_panel.get_custom_prompt()
        user_prompt, sys_inst = custom if custom else self._assemble_current_prompts()
        root = config.get("prompt_pack_dir", "") or None
        name = self.deployment_panel.get_team_name() or self.deployment_panel.get_car_folder() or "livery"
        try:
            pack = create_prompt_pack(wireframe, references, user_prompt, sys_inst, root=root, name=name)
        except Exception as e:
            QMessageBox.critical(self, "Prompt Pack Failed", f"Could not create the prompt pack:\n\n{e}")
            return

        copy_pack_to_clipboard(pack, "all")
        self._last_pack_dir = pack.folder
        self.status_bar.show_info(
            f"Prompt + {len(pack.image_paths)} image(s) copied to clipboard — saved in {pack.folder}"
        )
        dialog = PromptPackDialog(pack, parent=self)
        dialog.importResultRequested.connect(lambda d: self._import_external_result(d, dialog))
        dialog.exec()

    def _import_external_result(self, start_dir: str = "", dialog: Optional[QDialog] = None):
        """Loads an image made by another AI, undoes the pack padding, aligns it to the template."""
        from src.core.image_io import load_any_image
        from src.core.prompt_pack import find_pack_info, normalize_external_result

        start = start_dir or str(getattr(self, "_last_pack_dir", "") or Path.home() / "Downloads")
        path, _ = QFileDialog.getOpenFileName(
            dialog or self, "Import Image from Other AI", start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tga *.dds)",
        )
        if not path:
            return
        try:
            raw = load_any_image(path)
            info = find_pack_info(Path(path))
            if info is None and start_dir:
                info = find_pack_info(Path(start_dir) / "_")
            img = normalize_external_result(raw, info)
            img, report = fit_to_template(img, self.wireframe_panel.get_wireframe())
        except Exception as e:
            QMessageBox.critical(dialog or self, "Import Failed", f"Could not import the image:\n\n{e}")
            return

        pp = self.preview_panel
        if pp.revisions:  # keep earlier versions; the import becomes the next one
            pp.add_revision(img, notes="Imported from other AI", label=f"v{len(pp.revisions) + 1} (Other AI)")
        else:
            pp.reset_revisions(img, label="v1 (Other AI)")
        self.wheels_panel.refresh_palette()
        extra = f" — {report}" if report else ""
        self.status_bar.show_info(f"Imported {Path(path).name} ({img.size[0]} × {img.size[1]} px){extra}")
        if dialog is not None:
            dialog.accept()

    def _on_generation_finished(self, image: Image.Image, report: str = ""):
        self._set_busy(False)
        self.preview_panel.set_generated_image(image, is_refinement=False)
        self.wheels_panel.refresh_palette()
        w, h = image.size
        extra = f" — {report}" if report else ""
        self.status_bar.show_info(f"Livery generated ({w} × {h} px){extra}")

    def _on_generation_error(self, err_msg: str):
        self._set_busy(False)
        self.status_bar.show_error(err_msg)
        QMessageBox.critical(
            self,
            "Generation Failed",
            f"Failed to generate livery texture from AI:\n\n{err_msg}\n\n"
            "Tip: Ensure your API key is valid and has access to Gemini image models.",
        )

    # ==========================================================================
    # Refinement & Multi-Turn Iteration Workflow
    # ==========================================================================

    def _on_refine_livery(self, notes: str):
        curr_livery = self.preview_panel.get_generated_image()
        if curr_livery is None:
            QMessageBox.warning(
                self, "No Livery Loaded", "Please generate or load a livery first before refining."
            )
            return

        annotated_img = self.preview_panel.get_annotated_image()
        wireframe = self.wireframe_panel.get_wireframe()
        references = self.reference_panel.get_references()
        finish = self.preview_panel.get_finish()

        if not self.ai_engine.is_configured():
            res = QMessageBox.question(
                self,
                "API Key Required",
                "No Gemini API key is configured.\nWould you like to enter your Google AI Studio API key now?\n\n"
                "(Click 'No' to run offline mock refinement for preview testing)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if res == QMessageBox.StandardButton.Yes:
                self._open_settings()
                if not self.ai_engine.is_configured():
                    return
            elif res == QMessageBox.StandardButton.No:
                # Mock refinement offline
                self.status_bar.show_busy("Running offline refinement pattern...")
                iteration = len(self.preview_panel.revisions) + 1
                mock_refined = AIEngine.refine_mock_livery(
                    current_livery=curr_livery,
                    annotated_livery=annotated_img,
                    notes=notes,
                    finish=finish,
                    iteration=iteration,
                )
                self._on_refinement_finished(mock_refined, notes)
                return
            else:
                return

        active_model = config.get("ai_model", DEFAULT_AI_MODEL)
        self._set_busy(True, f"Refining Livery with {active_model}...")

        iteration = len(self.preview_panel.revisions) + 1
        self._refine_worker = RefinementWorker(
            ai_engine=self.ai_engine,
            current_livery=curr_livery,
            annotated_livery=annotated_img,
            notes=notes,
            wireframe=wireframe,
            references=references,
            finish=finish,
            iteration=iteration,
            target_resolution=4096,
        )
        self._refine_worker.finished.connect(lambda img, rep, n=notes: self._on_refinement_finished(img, n, rep))
        self._refine_worker.error.connect(self._on_refinement_error)
        self._refine_worker.start()

    def _on_refinement_finished(self, image: Image.Image, notes: str, report: str = ""):
        self._set_busy(False)
        self.preview_panel.set_generated_image(image, is_refinement=True, notes=notes)
        rev_count = len(self.preview_panel.revisions)
        extra = f" — {report}" if report else ""
        self.status_bar.show_info(f"Livery refined (Revision {rev_count}){extra}")

    def _realign_current(self):
        """Tools menu: snap the current version onto the template (fixes older, drifted results)."""
        rev = self.preview_panel.current_revision()
        wf = self.wireframe_panel.get_wireframe()
        if rev is None or wf is None:
            QMessageBox.information(self, "Re-align", "Load a template in the AI Generation tab and have a livery version open first.")
            return
        from src.core.template_fit import lock_to_template

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            out, report = lock_to_template(rev.image, wf, do_clip=config.get("clip_to_islands", True))
            if out.size != rev.image.size:
                out = out.resize(rev.image.size, Image.Resampling.LANCZOS)
        finally:
            QApplication.restoreOverrideCursor()
        if not report.aligned and not report.clipped:
            QMessageBox.information(self, "Re-align", "Nothing to fix: " + (report.summary() or "no change needed."))
            return
        n = len(self.preview_panel.revisions) + 1
        self.preview_panel.add_revision(out, notes="Re-aligned to template", label=f"v{n} (Aligned)")
        self.preview_panel.check_overlay.setChecked(True)
        self.status_bar.show_info(f"v{n}: {report.summary()}")

    def _on_refinement_error(self, err_msg: str):
        self._set_busy(False)
        self.status_bar.show_error(err_msg)
        QMessageBox.critical(
            self,
            "Refinement Failed",
            f"Failed to refine livery texture from AI:\n\n{err_msg}\n\n"
            "Tip: Ensure your API key is valid and has quota for Gemini image models.",
        )

    def _on_commit_livery(self):
        curr_idx = self.preview_panel.current_rev_idx + 1
        self.status_bar.show_info(f"✅ Livery version {curr_idx} committed! Ready to install or export.")
        QMessageBox.information(
            self,
            "Livery Committed",
            f"Livery Revision {curr_idx} has been committed as approved!\n\n"
            "You can now install it to Automobilista 2 or export it as a shareable ZIP package.",
        )

    # ==========================================================================
    # Existing Livery Editing Workflow
    # ==========================================================================

    def _open_load_livery_dialog(self):
        current_car = self.deployment_panel.get_car_folder()
        dialog = LoadLiveryDialog(initial_car=current_car, parent=self)
        dialog.liverySelected.connect(self._on_existing_livery_loaded)
        dialog.exec()

    def _on_existing_livery_loaded(
        self,
        texture_img: Image.Image,
        preview_img: Optional[Image.Image],
        car_folder: str,
        livery_id: int,
        team_name: str,
        meta: Optional[dict] = None,
    ):
        meta = meta or {}
        label = f"v1 (Loaded: {team_name[:18]})" if team_name else "v1 (Loaded)"
        self.preview_panel.reset_revisions(texture_img, label=label)
        self.preview_panel.set_custom_preview_image(preview_img)

        if car_folder:
            if self.deployment_panel.combo_car.findText(car_folder) == -1:
                self.deployment_panel.combo_car.addItem(car_folder)
            self.deployment_panel.combo_car.setCurrentText(car_folder)

        if livery_id:
            self.deployment_panel.spin_id.setValue(livery_id)

        if team_name:
            # strip the " #51" suffix the installer adds, so re-installing doesn't produce "Team #51 #51"
            clean = team_name
            suffix = f" #{livery_id}" if livery_id else ""
            if suffix and clean.endswith(suffix):
                clean = clean[: -len(suffix)]
            self.deployment_panel.edit_team.setText(clean)

        if meta.get("base_livery"):
            self.deployment_panel.set_base_livery(meta["base_livery"])

        extras = meta.get("extra_images") or {}
        self.wheels_panel.load_existing(extras)
        self.wheels_panel.refresh_palette()

        self.deployment_panel.check_id_collision()
        wheel_note = " (incl. custom rims)" if extras else ""
        self.status_bar.show_info(f"Loaded existing livery '{team_name}'{wheel_note} into editor")

    # ==========================================================================
    # Paint Editor
    # ==========================================================================

    def _open_paint_editor(self):
        """The Paint Editor lives in its own tab now; switching to it opens the current version."""
        self.tabs.setCurrentWidget(self.paint_editor)

    def _open_wheel_editor(self, slot: str):
        img = self.wheels_panel.get_slot_image(slot)
        if img is None:
            QMessageBox.information(
                self, "Wheels",
                "Load the rim texture from your car's template pack first (📂 Load texture…).",
            )
            return
        guide = self.wheels_panel.get_slot_template(slot)
        editor = LiveryPaintEditor(
            base_image=img,
            guide_image=guide,
            title=f"Wheel Paint Editor — {TEXTURE_SPECS[slot]['label']}",
            parent=self,
        )
        editor.doc.guide_opacity = 0.0 if guide is None else 0.25
        editor.sl_guide.setValue(int(editor.doc.guide_opacity * 100))
        editor.showMaximized()
        if editor.exec() == QDialog.DialogCode.Accepted and editor.result_image() is not None:
            self.wheels_panel.set_slot_result(slot, editor.result_image())
            self.status_bar.show_info(f"{TEXTURE_SPECS[slot]['label']} updated from the Paint Editor.")

    # ==========================================================================
    # Installation Workflow
    # ==========================================================================

    def _on_install_livery(self):
        if not self._confirm_editor_applied():
            return
        texture = self.preview_panel.get_generated_image()
        if texture is None:
            QMessageBox.warning(
                self, "No Livery Available", "Please generate or load a livery first before installing."
            )
            return

        car_folder = self.deployment_panel.get_car_folder()
        livery_id = self.deployment_panel.get_livery_id()
        team_name = self.deployment_panel.get_team_name()
        driver = self.deployment_panel.get_driver()
        finish = self.preview_panel.get_finish()
        custom_preview = self.preview_panel.get_custom_preview_image()

        if not car_folder:
            QMessageBox.warning(self, "Invalid Car Folder",
                                "Please select or enter a target car folder in the Install & Export tab.")
            self.tabs.setCurrentIndex(3)
            return

        wheels = self.wheels_panel.get_extra_textures()
        wheel_txt = " + custom rims" if wheels else ""
        self._set_busy(True, f"Installing '{team_name}' #{livery_id}{wheel_txt} to AMS2 ({finish} finish)...")

        self._install_worker = InstallWorker(
            car_folder=car_folder,
            livery_id=livery_id,
            team_name=team_name,
            driver=driver,
            texture_image=texture,
            finish_type=finish,
            custom_preview_image=custom_preview,
            base_livery=self.deployment_panel.get_base_livery(),
            extra_textures=self.wheels_panel.get_extra_textures(),
        )
        self._install_worker.success.connect(self._on_install_success)
        self._install_worker.error.connect(self._on_install_error)
        self._install_worker.start()

    def _on_install_success(self, target_dir: Path, backup_path: Optional[Path]):
        self._set_busy(False)
        self.deployment_panel.check_id_collision()
        msg = f"Livery successfully installed to:\n{target_dir}\n\n"
        if any(target_dir.glob("wheel_*.dds")):
            msg += "Custom wheel rims installed too.\n\n"
        if backup_path:
            msg += f"Backup created:\n{backup_path.name}"
        else:
            msg += "Created new car XML configuration."

        self.status_bar.show_info(f"Livery installed to AMS2 successfully ({target_dir.name})")
        QMessageBox.information(self, "Installation Complete", msg)

    def _on_install_error(self, err: str):
        self._set_busy(False)
        self.status_bar.show_error(err)
        QMessageBox.critical(self, "Installation Error", f"Failed to install livery:\n{err}")

    # ==========================================================================
    # Quick Livery File Installation Workflow (Direct File Install, No AI)
    # ==========================================================================

    def _on_quick_install(self):
        car_folder = self.deployment_panel.get_car_folder()
        if not car_folder:
            QMessageBox.warning(self, "Invalid Car Folder", "Please select or enter a target car folder.")
            return

        livery_id = self.deployment_panel.get_livery_id()
        team_name = self.deployment_panel.get_team_name()
        driver = self.deployment_panel.get_driver()
        finish = self.preview_panel.get_finish()

        # 1. File picker for livery texture file
        file_filter = "Livery Images (*.dds *.png *.jpg *.jpeg *.bmp *.tga *.webp);;DDS Textures (*.dds);;All Files (*.*)"
        chosen_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Livery Texture File to Install",
            "",
            file_filter,
        )
        if not chosen_file:
            return

        texture_path = Path(chosen_file)
        if not texture_path.exists():
            QMessageBox.warning(self, "File Not Found", f"The selected file does not exist:\n{texture_path}")
            return

        # 2. Optional custom preview thumbnail
        custom_preview = None
        reply = QMessageBox.question(
            self,
            "Preview Thumbnail",
            "Would you like to select a custom in-game preview image (2048x600, e.g. a showroom screenshot)?\n\n"
            "• Click 'Yes' to choose an image file\n"
            "• Click 'No' to auto-generate thumbnail from the livery texture",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        elif reply == QMessageBox.StandardButton.Yes:
            prev_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.tga *.webp);;All Files (*.*)"
            prev_file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Preview Image (2048x600)",
                "",
                prev_filter,
            )
            if prev_file:
                try:
                    with Image.open(prev_file) as pimg:
                        custom_preview = pimg.convert("RGBA").copy()
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Preview Load Failed",
                        f"Could not load custom preview image ({e}). Auto-generating fallback thumbnail instead.",
                    )
                    custom_preview = None

        ext_label = texture_path.suffix.upper()
        finish_desc = "preserving original alpha" if texture_path.suffix.lower() == ".dds" else f"{finish} finish"
        self._set_busy(True, f"Quick-installing {ext_label} livery '{team_name}' #{livery_id} ({finish_desc})...")

        self._quick_install_worker = QuickInstallWorker(
            file_path=texture_path,
            car_folder=car_folder,
            livery_id=livery_id,
            team_name=team_name,
            driver=driver,
            finish_type=finish,
            custom_preview_image=custom_preview,
            base_livery=self.deployment_panel.get_base_livery(),
        )
        self._quick_install_worker.success.connect(self._on_install_success)
        self._quick_install_worker.error.connect(self._on_install_error)
        self._quick_install_worker.start()

    # ==========================================================================
    # ZIP Export Workflow
    # ==========================================================================


    def _on_export_zip(self):
        if not self._confirm_editor_applied():
            return
        texture = self.preview_panel.get_generated_image()
        if texture is None:
            QMessageBox.warning(
                self, "No Livery Available", "Please generate or load a livery first before exporting."
            )
            return

        car_folder = self.deployment_panel.get_car_folder()
        livery_id = self.deployment_panel.get_livery_id()
        team_name = self.deployment_panel.get_team_name()
        driver = self.deployment_panel.get_driver()
        finish = self.preview_panel.get_finish()
        custom_preview = self.preview_panel.get_custom_preview_image()

        default_zip_name = f"{slugify(team_name)}_{car_folder}_{livery_id}_livery.zip"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Shareable Livery ZIP", default_zip_name, "ZIP Archives (*.zip)"
        )
        if not save_path:
            return

        self._set_busy(True, "Packaging 4K DDS, preview thumbnail, and XML into ZIP...")

        self._export_worker = ExportWorker(
            car_folder=car_folder,
            livery_id=livery_id,
            team_name=team_name,
            driver=driver,
            texture_image=texture,
            output_zip_path=Path(save_path),
            finish_type=finish,
            custom_preview_image=custom_preview,
            base_livery=self.deployment_panel.get_base_livery(),
            extra_textures=self.wheels_panel.get_extra_textures(),
        )
        self._export_worker.success.connect(self._on_export_success)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_success(self, zip_path: Path):
        self._set_busy(False)
        self.status_bar.show_info(f"Exported shareable ZIP: {zip_path.name}")
        QMessageBox.information(
            self,
            "Export Complete",
            f"Shareable ZIP package successfully created at:\n{zip_path}\n\n"
            "Ready to upload to Overtake.gg or share with friends!",
        )

    def _on_export_error(self, err: str):
        self._set_busy(False)
        self.status_bar.show_error(err)
        QMessageBox.critical(self, "Export Error", f"Failed to export ZIP:\n{err}")

    # ==========================================================================
    # Rollback / Restore Backup Workflow
    # ==========================================================================

    def _on_restore_backup(self):
        car_folder = self.deployment_panel.get_car_folder()
        if not car_folder:
            QMessageBox.warning(self, "Select Car", "Please select a car folder to restore backups.")
            return

        dialog = BackupRestoreDialog(car_folder, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_backup:
            try:
                restored = LiveryInstaller.restore_backup(dialog.selected_backup, car_folder)
                self.deployment_panel.check_id_collision()
                self.status_bar.show_info(f"Restored backup: {dialog.selected_backup.name}")
                QMessageBox.information(
                    self,
                    "Restore Successful",
                    f"Successfully restored XML configuration from:\n{dialog.selected_backup.name}",
                )
            except Exception as e:
                self.status_bar.show_error(str(e))
                QMessageBox.critical(self, "Restore Failed", f"Failed to restore backup:\n{e}")

    def closeEvent(self, event):
        if self.paint_editor.is_dirty():
            res = QMessageBox.question(
                self, "Unapplied paint edits",
                "The Paint Editor has edits that haven't been applied to a version. Quit and discard them?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Discard:
                event.ignore()
                return
        self.paint_editor.save_workspace()
        workers = [self._gen_worker, self._refine_worker, self._install_worker,
                   self._quick_install_worker, self._export_worker, getattr(self.wheels_panel, "_worker", None)]
        running = [w for w in workers if w is not None and w.isRunning()]
        if running:
            res = QMessageBox.question(
                self, "Work in progress",
                "A generation / install is still running. Quit anyway?\n"
                "(An install interrupted half-way can leave files incomplete.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if res != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            for w in running:
                w.wait(3000)
        super().closeEvent(event)

    def _set_busy(self, is_busy: bool, message: str = ""):
        self.preview_panel.set_busy(is_busy, message)
        self.actions_panel.set_busy(is_busy)
        if is_busy:
            self.status_bar.show_busy(message)
        else:
            self.status_bar.show_info("Ready")
