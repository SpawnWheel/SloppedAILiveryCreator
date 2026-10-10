"""
Wheels / Rims tab.

How wheels work in AMS2
-----------------------
A livery override only replaces the textures you list in the car XML. Anything you *don't* list
comes from the BASELIVERY you picked - that's why, until now, your cars always had the stock rims.

Most cars expose these wheel textures in their <car>_dist.xml:
    WHEELDIFF      rim_diff       the rim colour texture            1024 x 1024
    WHEELBLURDIFF  rim_blur_diff  the rim when it spins fast (blur)  1024 x 1024
To repaint rims you need the rim texture(s) from the car's template pack, recolour / repaint them,
and install them next to body.dds. This panel does exactly that.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.core.car_profile import CarProfile
from src.core.image_io import load_any_image
from src.core.image_processor import ImageProcessor
from src.utils.constants import IMAGE_OPEN_FILTER, TEXTURE_SPECS, WHEEL_TEXTURE_NAMES
from src.ui.qt_image import pil_to_qpixmap

HOW_IT_WORKS = (
    "<b>How wheels work in AMS2</b><br>"
    "Your livery only replaces the textures listed in the car's XML. Everything else - including the rims - "
    "comes from the <b>Base Livery</b> you choose in the <b>Install &amp; Export</b> tab. So you have two options:<br>"
    "&nbsp;&nbsp;<b>1. Easy:</b> pick a Base Livery whose stock rims you like (e.g. Chrome / Matte variants).<br>"
    "&nbsp;&nbsp;<b>2. Custom rims:</b> load the rim texture from your car's template pack "
    "(files named like <i>rim_diff</i> and <i>rim_blur_diff</i>), then recolour it, let the AI match your livery, "
    "or paint it by hand. It is installed as <code>WHEELDIFF</code> / <code>WHEELBLURDIFF</code> next to body.dds."
)


class WheelWorker(QThread):
    finished_ok = Signal(str, object)  # slot, image
    failed = Signal(str)

    def __init__(self, ai_engine, slot: str, template: Image.Image, body: Optional[Image.Image], prompt: str):
        super().__init__()
        self.ai_engine, self.slot, self.template, self.body, self.prompt = ai_engine, slot, template, body, prompt

    def run(self):
        try:
            if self.ai_engine is not None and self.ai_engine.is_configured():
                img = self.ai_engine.generate_wheel(
                    rim_template=self.template, body_livery=self.body, prompt=self.prompt,
                    is_blur=self.slot == "WHEELBLURDIFF", target_resolution=TEXTURE_SPECS[self.slot]["size"],
                )
            else:
                from src.core.ai_engine import AIEngine

                img = AIEngine.generate_mock_wheel(self.template, self.body, TEXTURE_SPECS[self.slot]["size"])
            self.finished_ok.emit(self.slot, img)
        except Exception as e:
            self.failed.emit(str(e))


class WheelSlotCard(QFrame):
    selected = Signal(str)

    def __init__(self, slot: str, parent=None):
        super().__init__(parent)
        self.slot = slot
        self.template: Optional[Image.Image] = None
        self.result: Optional[Image.Image] = None
        self.setStyleSheet(
            "WheelSlotCard { background-color: #1a1c22; border: 1px solid #2e323b; border-radius: 8px; }"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        head = QHBoxLayout()
        self.radio = QRadioButton(TEXTURE_SPECS[slot]["label"])
        self.radio.setStyleSheet("color: #e0e4ec; font-weight: bold; font-size: 11px;")
        self.radio.toggled.connect(lambda on: on and self.selected.emit(self.slot))
        head.addWidget(self.radio)
        head.addStretch()
        code = QLabel(slot)
        code.setStyleSheet("color: #64b5f6; font-size: 10px; font-family: Consolas, monospace;")
        head.addWidget(code)
        v.addLayout(head)

        self.preview = QLabel("No rim texture loaded")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(200, 200)
        self.preview.setStyleSheet(
            "QLabel { background-color: #121418; border: 1px dashed #3c404b; border-radius: 6px;"
            " color: #606674; font-size: 11px; }"
        )
        v.addWidget(self.preview, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #8c96a5; font-size: 10px;")
        v.addWidget(self.status)

        row = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load texture…")
        self.btn_load.setToolTip(
            "Load the rim texture from your car's template pack (PNG / PSD / DDS).\n"
            "Look for names like rim_diff / wheel / rims." if slot == "WHEELDIFF" else
            "Load the blurred (spinning) rim texture from the template pack - usually named rim_blur_diff."
        )
        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setToolTip("Throw away edits and go back to the loaded texture")
        self.btn_clear = QPushButton("✕")
        self.btn_clear.setToolTip("Don't install this texture (use the base livery's rims)")
        self.btn_clear.setFixedWidth(30)
        row.addWidget(self.btn_load, 1)
        row.addWidget(self.btn_reset)
        row.addWidget(self.btn_clear)
        v.addLayout(row)
        self.refresh()

    def current(self) -> Optional[Image.Image]:
        return self.result if self.result is not None else self.template

    def refresh(self) -> None:
        img = self.current()
        if img is None:
            self.preview.clear()
            self.preview.setText("No rim texture loaded")
            self.status.setText("Not installed - the Base Livery's stock rim is used.")
            self.btn_reset.setEnabled(False)
            self.btn_clear.setEnabled(False)
            return
        thumb = img.copy()
        thumb.thumbnail((260, 260))
        self.preview.setPixmap(pil_to_qpixmap(thumb))
        w, h = img.size
        if self.result is not None:
            self.status.setText(f"✅ Custom rim ready ({w}×{h}) - will be installed as {self.slot}.")
            self.status.setStyleSheet("color: #81c784; font-size: 10px;")
        else:
            self.status.setText(f"Template loaded ({w}×{h}). Recolour it, use AI, or paint it - then it gets installed.")
            self.status.setStyleSheet("color: #ffb74d; font-size: 10px;")
        self.btn_reset.setEnabled(self.result is not None and self.template is not None)
        self.btn_clear.setEnabled(True)


class WheelsPanel(QWidget):
    statusMessage = Signal(str)
    busyChanged = Signal(bool, str)
    paintEditorRequested = Signal(str)  # slot

    def __init__(self, ai_engine=None, body_provider: Optional[Callable[[], Optional[Image.Image]]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.ai_engine = ai_engine
        self.body_provider = body_provider or (lambda: None)
        self._worker: Optional[WheelWorker] = None
        self._queue = []
        self._tint_color = QColor(20, 20, 22)
        self.profile: Optional[CarProfile] = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        info = QLabel(HOW_IT_WORKS)
        info.setWordWrap(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setStyleSheet(
            "QLabel { background-color: #17212e; border: 1px solid #234b73; border-radius: 6px;"
            " padding: 8px; color: #b9cbe0; font-size: 11px; }"
        )
        v.addWidget(info)

        self.lbl_car = QLabel("")
        self.lbl_car.setWordWrap(True)
        self.lbl_car.setStyleSheet("font-size: 11px;")
        v.addWidget(self.lbl_car)

        cards = QHBoxLayout()
        self.cards: Dict[str, WheelSlotCard] = {}
        self.slot_group = QButtonGroup(self)
        for slot in WHEEL_TEXTURE_NAMES:
            card = WheelSlotCard(slot)
            card.btn_load.clicked.connect(lambda _=False, s=slot: self._load_template(s))
            card.btn_reset.clicked.connect(lambda _=False, s=slot: self._reset(s))
            card.btn_clear.clicked.connect(lambda _=False, s=slot: self._clear(s))
            self.slot_group.addButton(card.radio)
            self.cards[slot] = card
            cards.addWidget(card, 1)
        self.cards["WHEELDIFF"].radio.setChecked(True)
        v.addLayout(cards, 1)

        self.chk_both = QCheckBox("Apply recolour / AI to both textures at once (recommended - keeps them matching)")
        self.chk_both.setChecked(True)
        self.chk_both.setStyleSheet("color: #b0b8c6; font-size: 11px;")
        v.addWidget(self.chk_both)

        # Quick recolour
        tint = QFrame()
        tint.setStyleSheet("QFrame { background-color: #1f2229; border: 1px solid #2e323b; border-radius: 6px; }"
                           "QLabel { border: none; }")
        tl = QVBoxLayout(tint)
        tl.setContentsMargins(8, 6, 8, 8)
        t_head = QLabel("<b>🎨 Quick recolour</b> &nbsp;<span style='color:#8c96a5'>(no AI - keeps all the rim's "
                        "shading & detail)</span>")
        t_head.setStyleSheet("color: #d0d4dc; font-size: 11px;")
        tl.addWidget(t_head)
        row = QHBoxLayout()
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(34, 26)
        self.btn_color.clicked.connect(self._pick_tint)
        row.addWidget(self.btn_color)
        self.palette_row = QHBoxLayout()
        self.palette_row.setSpacing(3)
        row.addLayout(self.palette_row)
        row.addStretch()
        tl.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Strength"))
        self.sl_strength = QSlider(Qt.Orientation.Horizontal)
        self.sl_strength.setRange(10, 100)
        self.sl_strength.setValue(90)
        row2.addWidget(self.sl_strength, 1)
        self.chk_highlights = QCheckBox("Keep highlights")
        self.chk_highlights.setChecked(True)
        self.chk_highlights.setStyleSheet("color: #b0b8c6; font-size: 11px;")
        row2.addWidget(self.chk_highlights)
        self.btn_tint = QPushButton("Apply recolour")
        self.btn_tint.clicked.connect(self._apply_tint)
        row2.addWidget(self.btn_tint)
        tl.addLayout(row2)
        v.addWidget(tint)
        self._set_tint_color(self._tint_color)
        self.refresh_palette()

        # AI + editor
        ai = QHBoxLayout()
        self.ed_prompt = QLineEdit()
        self.ed_prompt.setPlaceholderText("Optional style for AI, e.g. 'gloss black with a red lip stripe', "
                                          "'gold like the Rothmans cars', 'match the livery's main blue'")
        self.ed_prompt.setStyleSheet("background-color: #181a20; border: 1px solid #3c404b; border-radius: 4px;"
                                     " padding: 5px; color: #fff; font-size: 11px;")
        ai.addWidget(self.ed_prompt, 1)
        self.btn_ai = QPushButton("✨ Match Livery with AI")
        self.btn_ai.setStyleSheet(
            "QPushButton { background-color: #7b1fa2; color: white; font-weight: bold; border: none;"
            " border-radius: 5px; padding: 7px 12px; } QPushButton:hover { background-color: #8e24aa; }"
            " QPushButton:disabled { background: #252830; color: #555c68; }"
        )
        self.btn_ai.setToolTip("Sends the rim texture + your current body livery to Gemini and asks for matching rims.\n"
                               "Without an API key it falls back to a simple colour match.")
        self.btn_ai.clicked.connect(self._run_ai)
        ai.addWidget(self.btn_ai)
        self.btn_paint = QPushButton("🖌 Paint by hand…")
        self.btn_paint.setStyleSheet(
            "QPushButton { background-color: #2b3547; border: 1px solid #3d5070; color: #82b1ff; border-radius: 5px;"
            " padding: 7px 12px; font-weight: bold; } QPushButton:hover { background-color: #374763; color: white; }"
        )
        self.btn_paint.clicked.connect(lambda: self.paintEditorRequested.emit(self.selected_slot()))
        ai.addWidget(self.btn_paint)
        v.addLayout(ai)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.set_profile(None)

    # ------------------------------------------------------------------ state
    def selected_slot(self) -> str:
        for slot, card in self.cards.items():
            if card.radio.isChecked():
                return slot
        return "WHEELDIFF"

    def _target_slots(self):
        if self.chk_both.isChecked():
            return [s for s, c in self.cards.items() if c.current() is not None]
        return [self.selected_slot()]

    def set_profile(self, profile: Optional[CarProfile]) -> None:
        self.profile = profile
        if profile is None:
            self.lbl_car.setText("")
            return
        if profile.supports_wheels:
            names = ", ".join(n for n in WHEEL_TEXTURE_NAMES if profile.supports(n))
            self.lbl_car.setText(f"✅ <b>{profile.car_folder}</b> supports custom rims ({names}).")
            self.lbl_car.setStyleSheet("color: #81c784; font-size: 11px;")
        elif profile.dist_xml is not None:
            self.lbl_car.setText(f"⚠️ <b>{profile.car_folder}</b>'s _dist.xml doesn't list wheel textures. Custom rims "
                                 "may not show in game - choosing a different Base Livery is the safe option here.")
            self.lbl_car.setStyleSheet("color: #ffb74d; font-size: 11px;")
        else:
            self.lbl_car.setText(f"ℹ️ No _dist.xml found for <b>{profile.car_folder}</b>, so wheel support can't be "
                                 "checked. Most AMS2 cars support WHEELDIFF.")
            self.lbl_car.setStyleSheet("color: #8c96a5; font-size: 11px;")

    def refresh_palette(self) -> None:
        while self.palette_row.count():
            w = self.palette_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        body = self.body_provider()
        colours = ImageProcessor.dominant_colors(body, 6) if body is not None else []
        colours = colours + [(18, 18, 20), (60, 62, 66), (192, 192, 192), (212, 175, 55)]
        tips = {(18, 18, 20): "Gloss black", (60, 62, 66): "Gunmetal", (192, 192, 192): "Silver", (212, 175, 55): "Gold"}
        for c in colours:
            b = QPushButton()
            b.setFixedSize(22, 22)
            b.setToolTip(tips.get(c, "From your livery"))
            b.setStyleSheet(f"background-color: rgb{tuple(c)}; border: 1px solid #555; border-radius: 3px;")
            b.clicked.connect(lambda _=False, col=QColor(*c): self._set_tint_color(col))
            self.palette_row.addWidget(b)

    def _set_tint_color(self, c: QColor) -> None:
        self._tint_color = QColor(c)
        self.btn_color.setStyleSheet(f"background-color: {c.name()}; border: 2px solid #9aa3b2; border-radius: 4px;")

    def _pick_tint(self) -> None:
        c = QColorDialog.getColor(self._tint_color, self, "Rim colour")
        if c.isValid():
            self._set_tint_color(c)

    # ------------------------------------------------------------------ actions
    def _load_template(self, slot: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Load {TEXTURE_SPECS[slot]['label']} texture", "", IMAGE_OPEN_FILTER)
        if not path:
            return
        try:
            img = load_any_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Wheel texture", f"Could not open file:\n{e}")
            return
        card = self.cards[slot]
        card.template, card.result = img, None
        card.radio.setChecked(True)
        card.refresh()
        self.statusMessage.emit(f"Loaded {TEXTURE_SPECS[slot]['label']} texture ({img.width}×{img.height}).")

    def _reset(self, slot: str) -> None:
        card = self.cards[slot]
        card.result = None
        card.refresh()

    def _clear(self, slot: str) -> None:
        card = self.cards[slot]
        card.template = card.result = None
        card.refresh()

    def _apply_tint(self) -> None:
        slots = [s for s in self._target_slots() if self.cards[s].current() is not None]
        if not slots:
            QMessageBox.information(self, "Wheels", "Load a rim texture from your template pack first.")
            return
        rgb = (self._tint_color.red(), self._tint_color.green(), self._tint_color.blue())
        for s in slots:
            card = self.cards[s]
            src = card.template if card.template is not None else card.result
            card.result = ImageProcessor.tint_texture(src, rgb, self.sl_strength.value() / 100.0,
                                                      self.chk_highlights.isChecked())
            card.refresh()
        self.statusMessage.emit(f"Rims recoloured to {self._tint_color.name().upper()}.")

    def _run_ai(self) -> None:
        slots = [s for s in self._target_slots() if self.cards[s].current() is not None]
        if not slots:
            QMessageBox.information(self, "Wheels", "Load a rim texture from your template pack first.")
            return
        body = self.body_provider()
        self._queue = list(slots)
        self._run_next_ai(body)

    def _run_next_ai(self, body) -> None:
        if not self._queue:
            self.busyChanged.emit(False, "")
            self.btn_ai.setEnabled(True)
            return
        slot = self._queue.pop(0)
        card = self.cards[slot]
        tpl = card.template if card.template is not None else card.result
        self.btn_ai.setEnabled(False)
        self.busyChanged.emit(True, f"Painting {TEXTURE_SPECS[slot]['label']} with AI...")
        self._worker = WheelWorker(self.ai_engine, slot, tpl, body, self.ed_prompt.text())
        self._worker.finished_ok.connect(lambda s, img, b=body: self._on_ai_done(s, img, b))
        self._worker.failed.connect(self._on_ai_failed)
        self._worker.start()

    def _on_ai_done(self, slot: str, img: Image.Image, body) -> None:
        card = self.cards[slot]
        card.result = img
        card.refresh()
        self.statusMessage.emit(f"{TEXTURE_SPECS[slot]['label']} painted.")
        self._run_next_ai(body)

    def _on_ai_failed(self, msg: str) -> None:
        self._queue = []
        self.btn_ai.setEnabled(True)
        self.busyChanged.emit(False, "")
        QMessageBox.critical(self, "Wheel AI failed", msg)

    # ------------------------------------------------------------------ public API
    def get_slot_image(self, slot: str) -> Optional[Image.Image]:
        return self.cards[slot].current()

    def get_slot_template(self, slot: str) -> Optional[Image.Image]:
        return self.cards[slot].template

    def set_slot_result(self, slot: str, img: Optional[Image.Image]) -> None:
        card = self.cards[slot]
        card.result = img
        card.refresh()

    def load_existing(self, textures: Dict[str, Image.Image]) -> None:
        """Called when an installed livery is opened: its current wheel textures become the starting point."""
        for slot, card in self.cards.items():
            img = textures.get(slot)
            card.template = img
            card.result = img
            card.refresh()

    def get_extra_textures(self) -> Dict[str, Image.Image]:
        """Textures to install. Only edited ones - an untouched template would just re-install the stock rim."""
        return {s: c.result for s, c in self.cards.items() if c.result is not None}

    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()
