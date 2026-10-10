"""
Selection + template panels for the Livery Paint Editor.

  Selection  - the current selection mask (None = nothing selected = the whole texture is editable)
  PanelMap   - the template's UV panels (from template_fit.analyse_template), mapped to document pixels so
               the Panel Select tool can pick them and the Panels dock can list them
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRect, QThread, Signal
from PySide6.QtGui import QImage, QPainterPath

from src.core import template_fit  # noqa: F401  (import in the main thread, not inside the worker)
from src.ui.editor.masks import mask_bbox, mask_to_alpha_qimage, mask_to_path


class Selection:
    """Selection mask with cached Qt representations (outline path, alpha image, dim overlay)."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.mask: Optional[np.ndarray] = None
        self.label = ""
        self._path: Optional[QPainterPath] = None
        self._overlay: Optional[QImage] = None
        self._bbox: Optional[QRect] = None
        self.version = 0

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self.mask is not None

    def set(self, mask: Optional[np.ndarray], label: str = "") -> None:
        if mask is not None:
            if mask.shape != (self.h, self.w):
                raise ValueError(f"selection mask {mask.shape} != document {(self.h, self.w)}")
            if not mask.any():
                mask = None
        self.mask = mask
        self.label = label if mask is not None else ""
        self._path = None
        self._overlay = None
        self._bbox = None
        self.version += 1

    def resize(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.set(None)

    # ------------------------------------------------------------------ caches
    def bbox(self) -> Optional[QRect]:
        if self.mask is None:
            return None
        if self._bbox is None:
            self._bbox = mask_bbox(self.mask)
        return self._bbox

    def path(self) -> QPainterPath:
        if self._path is None:
            self._path = mask_to_path(self.mask)
        return self._path

    def overlay(self) -> Optional[QImage]:
        """Dark veil over everything NOT selected (makes the selected panels pop)."""
        if self.mask is None:
            return None
        if self._overlay is None:
            veil = (255 - self.mask)
            self._overlay = mask_to_alpha_qimage((veil.astype(np.uint16) * 85 // 255).astype(np.uint8),
                                                 rgb=(8, 10, 14))
        return self._overlay

    def contains(self, x: int, y: int) -> bool:
        if self.mask is None:
            return True
        return 0 <= x < self.w and 0 <= y < self.h and self.mask[y, x] > 127

    def copy_mask(self) -> Optional[np.ndarray]:
        return None if self.mask is None else self.mask.copy()


# ==============================================================================================
# Template panels
# ==============================================================================================


@dataclass
class PanelInfo:
    index: int              # 0-based; shown to the user as P{index + 1} (same numbering as the AI prompt)
    bbox: QRect             # document px
    area: float             # fraction of the texture
    centroid: QPointF       # document px
    where: str              # "top-left", "centre", ...
    name: str = ""

    @property
    def code(self) -> str:
        return f"P{self.index + 1}"

    @property
    def title(self) -> str:
        return f"{self.code} · {self.name}" if self.name else self.code


class PanelMap:
    """Per-pixel panel index map (uint8, 0 = background, k = panel k-1) at document resolution."""

    def __init__(self, labels: np.ndarray, infos: List[PanelInfo]):
        self.labels = labels
        self.infos = infos
        h, w = labels.shape
        step = max(1, max(w, h) // 1024)
        self._small = labels[::step, ::step]
        self._small_counts = np.bincount(self._small.ravel(), minlength=len(infos) + 1)
        self._paths: Dict[int, QPainterPath] = {}

    def __len__(self) -> int:
        return len(self.infos)

    @property
    def size(self):
        return self.labels.shape[1], self.labels.shape[0]

    def index_at(self, x: int, y: int) -> int:
        h, w = self.labels.shape
        if not (0 <= x < w and 0 <= y < h):
            return -1
        return int(self.labels[y, x]) - 1

    def mask(self, indices: Sequence[int]) -> np.ndarray:
        ids = [i + 1 for i in indices if 0 <= i < len(self.infos)]
        if not ids:
            return np.zeros(self.labels.shape, np.uint8)
        if len(ids) == 1:
            return (self.labels == ids[0]).astype(np.uint8) * 255
        lut = np.zeros(256, np.uint8)
        lut[ids] = 255
        return lut[self.labels]

    def all_mask(self) -> np.ndarray:
        return (self.labels > 0).astype(np.uint8) * 255

    def path(self, index: int) -> QPainterPath:
        if index not in self._paths:
            self._paths[index] = mask_to_path(self.mask([index]))
        return self._paths[index]

    def coverage(self, sel: Optional[np.ndarray]) -> List[float]:
        """Fraction of each panel covered by the selection (fast, measured on a ~1024 px copy)."""
        n = len(self.infos)
        if sel is None:
            return [0.0] * n
        h, w = self.labels.shape
        step = max(1, max(w, h) // 1024)
        s = sel[::step, ::step] > 127
        hit = np.bincount(self._small.ravel(), weights=s.ravel(), minlength=n + 1)
        return [float(hit[i + 1] / self._small_counts[i + 1]) if self._small_counts[i + 1] else 0.0
                for i in range(n)]

    def names(self) -> Dict[str, str]:
        return {str(p.index): p.name for p in self.infos if p.name}

    def set_names(self, names: Optional[Dict]) -> None:
        for k, v in (names or {}).items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(self.infos):
                self.infos[i].name = str(v)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_guide(cls, guide: QImage, doc_w: int, doc_h: int) -> Optional["PanelMap"]:
        """Finds the template's panels. Returns None when the template has no recognisable panel layout."""
        from src.core.template_fit import _where, analyse_template
        from src.ui.qt_image import qimage_to_pil

        if guide is None or guide.isNull():
            return None
        pil = qimage_to_pil(guide)
        try:
            analysis = analyse_template(pil)
        except Exception as e:  # never break the editor over analysis
            print(f"[PanelMap] analysis failed: {e}")
            return None
        if analysis is None or analysis.labels is None or not analysis.panel_ids:
            return None
        labels = analysis.labels
        lut = np.zeros(int(labels.max()) + 1, np.uint8)
        for k, pid in enumerate(analysis.panel_ids[:255]):
            lut[pid] = k + 1
        small = lut[labels]
        wh, ww = small.shape
        if (ww, wh) != (doc_w, doc_h):
            ys = (np.arange(doc_h) * wh // doc_h).clip(0, wh - 1)
            xs = (np.arange(doc_w) * ww // doc_w).clip(0, ww - 1)
            small = small[ys][:, xs]
        small = np.ascontiguousarray(small)
        infos: List[PanelInfo] = []
        total = float(doc_w * doc_h)
        for k in range(len(analysis.panel_ids[:255])):
            m = small == (k + 1)
            ys, xs = np.nonzero(m)
            if xs.size == 0:
                continue
            bbox = QRect(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            cx, cy = float(xs.mean()), float(ys.mean())
            infos.append(PanelInfo(len(infos), bbox, xs.size / total, QPointF(cx, cy),
                                   _where(cx / doc_w, cy / doc_h)))
        if len(infos) != len(analysis.panel_ids[:255]):  # renumber if a panel vanished in resampling
            remap = np.zeros(256, np.uint8)
            keep = [k + 1 for k in range(len(analysis.panel_ids[:255])) if (small == k + 1).any()]
            for new, old in enumerate(keep, 1):
                remap[old] = new
            small = remap[small]
        return cls(small, infos)


_live_workers: set = set()


def wait_for_workers(timeout_ms: int = 20000) -> None:
    """Called on app exit: a QThread destroyed while running aborts the process."""
    for w in list(_live_workers):
        w.wait(timeout_ms)


class PanelWorker(QThread):
    """Runs the (1-3 s) template analysis off the UI thread. Kept alive by _live_workers until it finishes."""

    done = Signal(object, int)  # PanelMap | None, request id

    def __init__(self, guide: QImage, w: int, h: int, request_id: int, parent: Optional[QObject] = None):
        super().__init__(None)
        _live_workers.add(self)
        self.finished.connect(lambda: _live_workers.discard(self))
        self.guide = guide.copy()
        self.w, self.h = w, h
        self.request_id = request_id

    def run(self) -> None:
        pm = None
        try:
            pm = PanelMap.from_guide(self.guide, self.w, self.h)
        except Exception as e:  # pragma: no cover
            print(f"[PanelWorker] {e}")
        self.done.emit(pm, self.request_id)


def _install_exit_hook() -> None:
    import atexit

    atexit.register(wait_for_workers)
    try:
        from PySide6.QtCore import QCoreApplication

        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(wait_for_workers)
    except Exception:  # pragma: no cover
        pass


_install_exit_hook()
