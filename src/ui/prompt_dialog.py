"""
Interactive Prompt & System Instructions Editor Dialog.
Allows viewing, inspecting, and editing the exact text prompt and system instructions sent to Gemini.
"""

from typing import Callable, List, Optional, Tuple
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PromptEditorDialog(QDialog):
    """Dialog to view and edit the text prompts and system instructions sent to the AI model."""

    promptApplied = Signal(str, str)  # (user_prompt, system_instruction)

    def __init__(
        self,
        current_user_prompt: str,
        current_system_instruction: str,
        template_name: str = "",
        wireframe: Optional[Image.Image] = None,
        references: Optional[List[Tuple[Image.Image, str]]] = None,
        regenerate_callback: Optional[Callable[[], Tuple[str, str]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("AI Model Prompt & Instructions Inspector")
        self.resize(780, 620)
        self.setMinimumSize(640, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1c22;
                color: #ffffff;
            }
        """)

        self.template_name = template_name
        self.wireframe = wireframe
        self.references = references or []
        self.regenerate_callback = regenerate_callback

        self.initial_user_prompt = current_user_prompt
        self.initial_system_instruction = current_system_instruction

        self._build_ui(current_user_prompt, current_system_instruction)

    def _build_ui(self, user_prompt: str, system_instruction: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # 1. Header Banner & Asset Summary
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #21252e;
                border: 1px solid #2f3442;
                border-radius: 6px;
                padding: 8px 12px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setSpacing(4)
        header_layout.setContentsMargins(4, 4, 4, 4)

        title_label = QLabel("<b>Gemini Multimodal Payload Inspection</b>")
        title_label.setStyleSheet("color: #64b5f6; font-size: 13px;")
        header_layout.addWidget(title_label)

        # Asset summaries
        wf_info = f"📐 <b>Image 1 (Wireframe Template):</b> {self.template_name}" if self.template_name else "📐 <b>Image 1:</b> No wireframe loaded"
        if self.wireframe:
            wf_info += f" ({self.wireframe.width} × {self.wireframe.height} px)"

        ref_count = len(self.references)
        ref_roles = [role for _, role in self.references]
        roles_str = ", ".join(ref_roles) if ref_roles else "None"
        ref_info = f"🖼 <b>Reference Images ({ref_count}):</b> {roles_str}"

        model_info = "⚡ <b>Target Model:</b> gemini-3.1-flash-image (Output: 4096×4096 px 4K, 1:1 Aspect Ratio)"

        for line in [wf_info, ref_info, model_info]:
            lbl = QLabel(line)
            lbl.setStyleSheet("color: #b0b8c6; font-size: 11px;")
            header_layout.addWidget(lbl)

        layout.addWidget(header_frame)

        # 2. Tabs for Prompt vs System Instruction
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2f3442;
                background: #1e2026;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #252933;
                color: #8c93a4;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 500;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1e2026;
                color: #ffffff;
                border-top: 2px solid #0078d4;
            }
        """)

        # Tab 1: User Specification Prompt
        tab_user = QWidget()
        tab_user_layout = QVBoxLayout(tab_user)
        tab_user_layout.setContentsMargins(10, 10, 10, 10)
        tab_user_layout.setSpacing(6)

        user_hint = QLabel("The text prompt passed to the model alongside the wireframe and reference images:")
        user_hint.setStyleSheet("color: #8c93a4; font-size: 11px;")
        tab_user_layout.addWidget(user_hint)

        self.user_prompt_edit = QTextEdit()
        self.user_prompt_edit.setPlainText(user_prompt)
        self.user_prompt_edit.setStyleSheet("""
            QTextEdit {
                background-color: #14161a;
                border: 1px solid #333846;
                border-radius: 4px;
                padding: 8px;
                color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border: 1px solid #0078d4;
            }
        """)
        tab_user_layout.addWidget(self.user_prompt_edit, 1)

        self.tabs.addTab(tab_user, "📝 Livery Design Specification (Prompt)")

        # Tab 2: System Instruction
        tab_sys = QWidget()
        tab_sys_layout = QVBoxLayout(tab_sys)
        tab_sys_layout.setContentsMargins(10, 10, 10, 10)
        tab_sys_layout.setSpacing(6)

        sys_hint = QLabel("System instruction conditioning the model to output a strict flat 2D texture map (no 3D car perspective):")
        sys_hint.setStyleSheet("color: #8c93a4; font-size: 11px;")
        tab_sys_layout.addWidget(sys_hint)

        self.sys_prompt_edit = QTextEdit()
        self.sys_prompt_edit.setPlainText(system_instruction)
        self.sys_prompt_edit.setStyleSheet("""
            QTextEdit {
                background-color: #14161a;
                border: 1px solid #333846;
                border-radius: 4px;
                padding: 8px;
                color: #e2e8f0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border: 1px solid #0078d4;
            }
        """)
        tab_sys_layout.addWidget(self.sys_prompt_edit, 1)

        self.tabs.addTab(tab_sys, "⚙️ System Instructions (Rules & UV Constraints)")

        layout.addWidget(self.tabs, 1)

        # 3. Action Buttons Row
        btn_box = QHBoxLayout()
        btn_box.setSpacing(8)

        btn_auto = QPushButton("⟳ Auto-Generate from Inputs")
        btn_auto.setToolTip("Re-synthesize prompt using currently loaded template, car model, team, and style notes")
        btn_auto.setStyleSheet("""
            QPushButton {
                background-color: #2b303c;
                color: #82b1ff;
                border: 1px solid #3c404b;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #82b1ff;
            }
        """)
        btn_auto.clicked.connect(self._on_auto_generate)
        btn_box.addWidget(btn_auto)

        btn_reset = QPushButton("Reset to Default")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #252830;
                color: #9aa0a6;
                border: 1px solid #333846;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #30343f;
                color: #ffffff;
            }
        """)
        btn_reset.clicked.connect(self._on_reset)
        btn_box.addWidget(btn_reset)

        btn_box.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #2b303c;
                color: #ffffff;
                border: 1px solid #3c404b;
                padding: 6px 14px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3b4252;
            }
        """)
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QPushButton("Apply & Use Custom Prompt")
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                font-weight: bold;
                border: none;
                padding: 6px 18px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #1084e0;
            }
        """)
        btn_save.clicked.connect(self._on_apply)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

    def _on_auto_generate(self):
        if self.regenerate_callback:
            new_user, new_sys = self.regenerate_callback()
            self.user_prompt_edit.setPlainText(new_user)
            self.sys_prompt_edit.setPlainText(new_sys)

    def _on_reset(self):
        self.user_prompt_edit.setPlainText(self.initial_user_prompt)
        self.sys_prompt_edit.setPlainText(self.initial_system_instruction)

    def _on_apply(self):
        self.promptApplied.emit(
            self.user_prompt_edit.toPlainText().strip(),
            self.sys_prompt_edit.toPlainText().strip(),
        )
        self.accept()

    def get_prompts(self) -> Tuple[str, str]:
        return (
            self.user_prompt_edit.toPlainText().strip(),
            self.sys_prompt_edit.toPlainText().strip(),
        )
