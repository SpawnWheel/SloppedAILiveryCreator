"""
Dialog shown after "Copy Prompt for Other AI": shows the prompt pack folder, lets the user copy the prompt
and/or the images to the clipboard, drag images straight into a browser, and import the finished result.
"""

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QMimeData, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.prompt_pack import PromptPack

# ------------------------------------------------------------------------------
# Clipboard helpers
# ------------------------------------------------------------------------------


def _file_urls(paths: List[Path]) -> List[QUrl]:
    return [QUrl.fromLocalFile(str(Path(p).resolve())) for p in paths]


def copy_pack_to_clipboard(pack: PromptPack, mode: str = "all") -> None:
    """
    mode = "all"    -> prompt text + image files (paste into ChatGPT / Gemini / Claude web in Chrome or Edge)
           "text"   -> prompt text only
           "images" -> image files only (like Ctrl+C on the files in Explorer)
    """
    mime = QMimeData()
    if mode in ("all", "images") and pack.image_paths:
        mime.setUrls(_file_urls(pack.image_paths))
    if mode in ("all", "text"):
        mime.setText(pack.prompt_text)
    QApplication.clipboard().setMimeData(mime)


def copy_single_image(path: Path) -> None:
    """Copies one image as bitmap *and* file - for sites that only accept pasted pictures."""
    mime = QMimeData()
    img = QImage(str(path))
    if not img.isNull():
        mime.setImageData(img)
    mime.setUrls(_file_urls([path]))
    QApplication.clipboard().setMimeData(mime)


class _DraggableImageList(QListWidget):
    """Thumbnail strip whose items can be dragged into a browser / Explorer as real files."""

    def __init__(self, paths: List[Path], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._paths = paths
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(112, 112))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setFixedHeight(160)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        for p in paths:
            pix = QPixmap(str(p))
            item = QListWidgetItem(QIcon(pix.scaled(112, 112, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation)), p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            item.setToolTip(f"{p.name}\nDrag into the AI website, or right-click to copy just this image")
            self.addItem(item)

    def mimeData(self, items):  # noqa: N802 (Qt override)
        mime = QMimeData()
        mime.setUrls(_file_urls([Path(i.data(Qt.ItemDataRole.UserRole)) for i in items]))
        return mime

    def supportedDragActions(self):  # noqa: N802
        return Qt.DropAction.CopyAction

    def _context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        act = menu.addAction("Copy this image")
        if menu.exec(self.mapToGlobal(pos)) == act:
            copy_single_image(Path(item.data(Qt.ItemDataRole.UserRole)))


# ------------------------------------------------------------------------------
# Dialog
# ------------------------------------------------------------------------------

_BTN = """
QPushButton {{ background-color: {bg}; color: #ffffff; font-weight: bold; font-size: 11px;
               border: none; border-radius: 5px; padding: 6px 12px; }}
QPushButton:hover {{ background-color: {hover}; }}
"""


class PromptPackDialog(QDialog):
    importResultRequested = Signal(str)  # pack folder to start the file picker in

    def __init__(self, pack: PromptPack, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.pack = pack
        self.setWindowTitle("Prompt Pack — use with another AI")
        self.setMinimumSize(640, 600)
        self.setStyleSheet("QDialog { background-color: #1e2026; color: #ffffff; } QLabel { color: #d0d4dc; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        title = QLabel("📋 Prompt + images copied to the clipboard")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        lay.addWidget(title)

        steps = QLabel(
            "1. Open your AI site (ChatGPT, Gemini, …) and press <b>Ctrl+V</b> in the message box — "
            "the images and the prompt are pasted together.<br>"
            "2. If only one of them arrives, use <b>Copy Images</b> and <b>Copy Prompt</b> below one after the "
            "other, or drag the thumbnails straight into the browser.<br>"
            "3. Download the AI's image and click <b>Import AI Result…</b> to load it as a new version "
            "(it's auto-aligned to the template)."
        )
        steps.setWordWrap(True)
        steps.setTextFormat(Qt.TextFormat.RichText)
        steps.setStyleSheet("font-size: 11px; color: #b0b8c6;")
        lay.addWidget(steps)

        folder_lbl = QLabel(f"Saved to: <a href='{QUrl.fromLocalFile(str(pack.folder)).toString()}' "
                            f"style='color:#4fc3f7'>{pack.folder}</a>")
        folder_lbl.setTextFormat(Qt.TextFormat.RichText)
        folder_lbl.setOpenExternalLinks(True)
        folder_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        folder_lbl.setStyleSheet("font-size: 10px;")
        lay.addWidget(folder_lbl)

        self.image_list = _DraggableImageList(pack.image_paths, self)
        self.image_list.setStyleSheet(
            "QListWidget { background-color: #14161a; border: 1px solid #2e323b; border-radius: 6px; color: #c0c6d0; }"
        )
        if pack.image_paths:
            lay.addWidget(self.image_list)
        else:
            lay.addWidget(QLabel("(No images — load a template or add reference images to include them.)"))

        self.prompt_view = QPlainTextEdit(pack.prompt_text)
        self.prompt_view.setReadOnly(True)
        self.prompt_view.setStyleSheet(
            "QPlainTextEdit { background-color: #14161a; color: #e0e0e0; border: 1px solid #2e323b; "
            "border-radius: 6px; font-family: Consolas, monospace; font-size: 11px; }"
        )
        lay.addWidget(self.prompt_view, 1)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_all = self._btn("📋 Copy All", "#0078d4", "#1084e0", lambda: self._copy("all"))
        self.btn_images = self._btn("🖼 Copy Images", "#455a64", "#546e7a", lambda: self._copy("images"))
        self.btn_images.setEnabled(bool(pack.image_paths))
        self.btn_text = self._btn("📝 Copy Prompt", "#455a64", "#546e7a", lambda: self._copy("text"))
        self.btn_folder = self._btn("📂 Open Folder", "#2b303c", "#3b4252", self._open_folder)
        for b in (self.btn_all, self.btn_images, self.btn_text, self.btn_folder):
            row.addWidget(b)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        self.status = QLabel("")
        self.status.setStyleSheet("color: #81c784; font-size: 11px;")
        row2.addWidget(self.status, 1)
        self.btn_import = self._btn("📥 Import AI Result…", "#2e7d32", "#388e3c", self._import)
        row2.addWidget(self.btn_import)
        btn_close = self._btn("Close", "#2b303c", "#3b4252", self.accept)
        row2.addWidget(btn_close)
        lay.addLayout(row2)

    def _btn(self, text, bg, hover, slot) -> QPushButton:
        b = QPushButton(text)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet(_BTN.format(bg=bg, hover=hover))
        b.clicked.connect(slot)
        return b

    def _copy(self, mode: str):
        copy_pack_to_clipboard(self.pack, mode)
        self.status.setText({"all": "Prompt + images copied.", "images": "Images copied.",
                             "text": "Prompt copied."}[mode])

    def _open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.pack.folder)))

    def _import(self):
        self.importResultRequested.emit(str(self.pack.folder))
