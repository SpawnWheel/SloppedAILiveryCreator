"""
Main Application Window for AMS2 AI Livery Studio.
Integrates all panels, async worker threads, settings, and menus.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.core.ai_engine import AIEngine
from src.core.config import config
from src.core.exporter import LiveryExporter
from src.core.installer import LiveryInstaller
from src.ui.actions_panel import ActionsPanel, BackupRestoreDialog
from src.ui.deployment_panel import DeploymentPanel
from src.ui.load_livery_dialog import LoadLiveryDialog
from src.ui.preview_panel import PreviewPanel
from src.ui.prompt_dialog import PromptEditorDialog
from src.ui.reference_panel import ReferencePanel
from src.ui.widgets.status_bar import CustomStatusBar
from src.ui.wireframe_panel import WireframePanel
from src.utils.constants import (
    APP_NAME,
    APP_VERSION,
    AVAILABLE_AI_MODELS,
    DEFAULT_AI_MODEL,
    FALLBACK_AI_MODEL,
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

    finished = Signal(object)  # PIL.Image
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
            self.finished.emit(img)
        except Exception as e:
            self.error.emit(str(e))


class RefinementWorker(QThread):
    """Background worker for iterative livery refinement with Gemini multimodal generation."""

    finished = Signal(object)  # PIL.Image
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
            self.finished.emit(img)
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
    ):
        super().__init__()
        self.car_folder = car_folder
        self.livery_id = livery_id
        self.team_name = team_name
        self.driver = driver
        self.texture_image = texture_image
        self.finish_type = finish_type
        self.custom_preview_image = custom_preview_image

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
    ):
        super().__init__()
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
            )
            self.success.emit(out_path)
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

        action_clear_cache = QAction("&Clear Template Cache", self)
        action_clear_cache.triggered.connect(self._clear_cache)
        tools_menu.addAction(action_clear_cache)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        action_about = QAction("&About", self)
        action_about.triggered.connect(self._show_about)
        help_menu.addAction(action_about)

    def _build_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 4)
        main_layout.setSpacing(8)

        # Splitter dividing Left Panels and Right Panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # LEFT COLUMN: Section 1 (Wireframe), Section 2 (References & Prompt), Section 4 (AMS2 Deployment)
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

        self.deployment_panel = DeploymentPanel(self)
        left_layout.addWidget(self.deployment_panel)

        splitter.addWidget(left_container)

        # RIGHT COLUMN: Section 3 (Preview & Generation), Section 5 (Actions)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(8)

        self.preview_panel = PreviewPanel(self)
        self.preview_panel.generateRequested.connect(self._on_generate_livery)
        self.preview_panel.refineRequested.connect(self._on_refine_livery)
        self.preview_panel.rerollRequested.connect(self._on_reroll_livery)
        self.preview_panel.commitRequested.connect(self._on_commit_livery)
        self.preview_panel.loadLiveryRequested.connect(self._open_load_livery_dialog)
        right_layout.addWidget(self.preview_panel, 1)

        self.actions_panel = ActionsPanel(self)
        self.actions_panel.installRequested.connect(self._on_install_livery)
        self.actions_panel.exportRequested.connect(self._on_export_zip)
        self.actions_panel.restoreRequested.connect(self._on_restore_backup)
        right_layout.addWidget(self.actions_panel)

        splitter.addWidget(right_container)
        splitter.setSizes([520, 680])
        main_layout.addWidget(splitter, 1)

        # Bottom Custom Status Bar
        self.status_bar = CustomStatusBar(self)
        main_layout.addWidget(self.status_bar)

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

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "AI-Powered Livery Studio for Automobilista 2.<br>"
            "Generates 4K game-ready DDS textures (BC3/DXT5 + Mipmaps) with Alpha-channel reflectivity shaders "
            "and injects them into AMS2's XML override system with automatic backup and rollback.<br><br>"
            "Built with PySide6, Pillow, DirectXTex texconv, and Google Gemini Pro Image.",
        )

    # ==========================================================================
    # Generation Workflow
    # ==========================================================================

    def _on_wireframe_loaded(self, img: Image.Image, name: str):
        """Passes loaded wireframe into the preview viewport for optional overlay."""
        self.preview_panel.set_wireframe(img)

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

    def _on_generation_finished(self, image: Image.Image):
        self._set_busy(False)
        self.preview_panel.set_generated_image(image, is_refinement=False)
        self.status_bar.show_info("Livery generated successfully! (4096 × 4096 px)")

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
        self._refine_worker.finished.connect(lambda img, n=notes: self._on_refinement_finished(img, n))
        self._refine_worker.error.connect(self._on_refinement_error)
        self._refine_worker.start()

    def _on_refinement_finished(self, image: Image.Image, notes: str):
        self._set_busy(False)
        self.preview_panel.set_generated_image(image, is_refinement=True, notes=notes)
        rev_count = len(self.preview_panel.revisions)
        self.status_bar.show_info(f"Livery refined successfully! (Revision {rev_count})")

    def _on_refinement_error(self, err_msg: str):
        self._set_busy(False)
        self.status_bar.show_error(err_msg)
        QMessageBox.critical(
            self,
            "Refinement Failed",
            f"Failed to refine livery texture from AI:\n\n{err_msg}\n\n"
            "Tip: Ensure your API key is valid and has quota for Gemini image models.",
        )

    def _on_reroll_livery(self):
        """Re-rolls either the current refinement iteration or the initial generation."""
        if len(self.preview_panel.revisions) > 1 and self.preview_panel.current_rev_idx > 0:
            notes = self.preview_panel.get_refinement_notes()
            self._on_refine_livery(notes)
        else:
            self._on_generate_livery()

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
    ):
        label = f"v1 (Loaded: {team_name[:18]})" if team_name else "v1 (Loaded)"
        self.preview_panel.reset_revisions(texture_img, label=label)
        if preview_img:
            self.preview_panel.set_custom_preview_image(preview_img)

        if car_folder:
            if self.deployment_panel.combo_car.findText(car_folder) == -1:
                self.deployment_panel.combo_car.addItem(car_folder)
            self.deployment_panel.combo_car.setCurrentText(car_folder)

        if livery_id:
            self.deployment_panel.spin_id.setValue(livery_id)

        if team_name:
            self.deployment_panel.edit_team.setText(team_name)

        self.deployment_panel.check_id_collision()
        self.status_bar.show_info(f"Loaded existing livery '{team_name}' into editor")

    # ==========================================================================
    # Installation Workflow
    # ==========================================================================

    def _on_install_livery(self):
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
            QMessageBox.warning(self, "Invalid Car Folder", "Please select or enter a target car folder.")
            return

        self._set_busy(True, f"Installing '{team_name}' #{livery_id} to AMS2 ({finish} finish)...")

        self._install_worker = InstallWorker(
            car_folder=car_folder,
            livery_id=livery_id,
            team_name=team_name,
            driver=driver,
            texture_image=texture,
            finish_type=finish,
            custom_preview_image=custom_preview,
        )
        self._install_worker.success.connect(self._on_install_success)
        self._install_worker.error.connect(self._on_install_error)
        self._install_worker.start()

    def _on_install_success(self, target_dir: Path, backup_path: Optional[Path]):
        self._set_busy(False)
        self.deployment_panel.check_id_collision()
        msg = f"Livery successfully installed to:\n{target_dir}\n\n"
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
    # ZIP Export Workflow
    # ==========================================================================

    def _on_export_zip(self):
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

    def _set_busy(self, is_busy: bool, message: str = ""):
        self.preview_panel.set_busy(is_busy)
        self.actions_panel.set_busy(is_busy)
        if is_busy:
            self.status_bar.show_busy(message)
        else:
            self.status_bar.show_info("Ready")
