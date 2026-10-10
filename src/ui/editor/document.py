"""
EditorDocument: the layer stack, the template guide + its panels, the selection, cached composites and
the undo history.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter

from src.ui.editor.layers import (
    ItemLayer,
    Layer,
    LayerClip,
    RasterLayer,
    bake_layer,
    clone_layer,
    composite,
    new_image,
    paint_exact,
    rasterize_item,
)
from src.ui.editor.masks import blend_masked, combine, rect_slices
from src.ui.editor.selection import PanelMap, PanelWorker, Selection
from src.ui.editor.undo import UndoCommand, UndoStack


def _img_bytes(img: Optional[QImage]) -> int:
    return 0 if img is None else img.sizeInBytes()


class EditorDocument(QObject):
    changed = Signal()          # something visible changed -> repaint
    layersChanged = Signal()    # structure / names / visibility changed -> refresh layer list
    activeChanged = Signal(int)
    selectionChanged = Signal()
    panelsChanged = Signal()    # template panels (re)analysed - PanelMap or None
    guideChanged = Signal()

    def __init__(self, width: int, height: int, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.layers: List[Layer] = []
        self.active_index = -1
        self.undo = UndoStack()

        # template guide
        self.guide: Optional[QImage] = None
        self.guide_visible = True
        self.guide_opacity = 0.45
        self.guide_blend = "Multiply"

        # selection + panels
        self.selection = Selection(width, height)
        self.panels: Optional[PanelMap] = None
        self.panels_state = "none"   # none | analysing | ready | failed
        self._panel_names: Dict[str, str] = {}
        self._panel_req = 0
        self._panel_worker: Optional[PanelWorker] = None

        self._below: Optional[QImage] = None
        self._above: Optional[QImage] = None
        self._cache_valid = False
        self.dirty = False

    # ------------------------------------------------------------------ basics
    @property
    def active(self) -> Optional[Layer]:
        if 0 <= self.active_index < len(self.layers):
            return self.layers[self.active_index]
        return None

    def set_active(self, idx: int) -> None:
        idx = max(-1, min(idx, len(self.layers) - 1))
        if idx != self.active_index:
            self.active_index = idx
            self.invalidate()
            self.activeChanged.emit(idx)

    def index_of(self, layer: Layer) -> int:
        for i, l in enumerate(self.layers):
            if l is layer:
                return i
        return -1

    def invalidate(self) -> None:
        self._cache_valid = False
        self.changed.emit()

    def mark_dirty(self) -> None:
        self.dirty = True

    def rect(self) -> QRect:
        return QRect(0, 0, self.width, self.height)

    def reset(self, width: int, height: int, layers: List[Layer], active: int = -1) -> None:
        """Starts over with a new texture (used when the Paint tab loads another version)."""
        size_changed = (width, height) != (self.width, self.height)
        self.width, self.height = width, height
        self.selection.resize(width, height)
        if size_changed:
            self.guide = None
            self._set_panels(None, "none")
        self.replace_all(layers, active)
        self.selectionChanged.emit()

    # ------------------------------------------------------------------ guide + panels
    def set_guide(self, img: Optional[QImage], analyse: bool = True) -> None:
        if img is not None and not img.isNull():
            if img.width() != self.width or img.height() != self.height:
                img = img.scaled(self.width, self.height, Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        else:
            img = None
        same = (img is not None and self.guide is not None and img.cacheKey() == self.guide.cacheKey())
        self.guide = img
        self.guideChanged.emit()
        self.changed.emit()
        if same and self.panels is not None:
            return
        if img is None:
            self._set_panels(None, "none")
        elif analyse:
            self.request_panels()

    def request_panels(self) -> None:
        """Analyses the guide's panels in the background; panelsChanged fires when done."""
        if self.guide is None:
            self._set_panels(None, "none")
            return
        self._panel_req += 1
        self._set_panels(None, "analysing")
        worker = PanelWorker(self.guide, self.width, self.height, self._panel_req)
        worker.done.connect(self._on_panels_done)
        self._panel_worker = worker
        worker.start()

    def ensure_panels(self) -> Optional[PanelMap]:
        """Synchronous analysis (tests / first click while the worker hasn't finished)."""
        if self.panels is None and self.guide is not None and self.panels_state != "failed":
            if self._panel_worker is not None and self._panel_worker.isRunning():
                self._panel_worker.wait(15000)
                from PySide6.QtCore import QCoreApplication

                QCoreApplication.processEvents()
            if self.panels is None and self.panels_state != "failed":
                pm = PanelMap.from_guide(self.guide, self.width, self.height)
                self._set_panels(pm, "ready" if pm else "failed")
        return self.panels

    def _on_panels_done(self, pm: Optional[PanelMap], req: int) -> None:
        if req != self._panel_req:
            return  # an older request finished after a newer guide was set
        if self.panels is not None and self.panels_state == "ready":
            return  # ensure_panels() already computed this one synchronously
        if pm is not None and pm.size != (self.width, self.height):
            pm = None
        self._set_panels(pm, "ready" if pm is not None else "failed")

    def _set_panels(self, pm: Optional[PanelMap], state: str) -> None:
        self.panels = pm
        self.panels_state = state
        if pm is not None:
            pm.set_names(self._panel_names)
        self.panelsChanged.emit()
        self.changed.emit()

    def panel_names(self) -> Dict[str, str]:
        if self.panels is not None:
            self._panel_names = self.panels.names()
        return dict(self._panel_names)

    def set_panel_names(self, names: Optional[Dict]) -> None:
        self._panel_names = {str(k): str(v) for k, v in (names or {}).items()}
        if self.panels is not None:
            self.panels.set_names(self._panel_names)
            self.panelsChanged.emit()

    def rename_panel(self, index: int, name: str) -> None:
        if self.panels is not None and 0 <= index < len(self.panels.infos):
            self.panels.infos[index].name = name.strip()
            self._panel_names = self.panels.names()
            self.mark_dirty()
            self.panelsChanged.emit()
            self.changed.emit()

    # ------------------------------------------------------------------ selection (undoable)
    def set_selection(self, mask: Optional[np.ndarray], label: str = "Select", record: bool = True,
                      before: Optional[np.ndarray] = None, before_label: Optional[str] = None) -> None:
        """Sets the selection mask; `before` lets drag-selects record one undo step for the whole drag."""
        old = self.selection.copy_mask() if before is None else before
        old_label = self.selection.label if before_label is None else before_label
        self.selection.set(mask, label)
        new = self.selection.copy_mask()
        new_label = self.selection.label
        self.selectionChanged.emit()
        self.changed.emit()
        if not record:
            return
        if old is None and new is None:
            return

        def put(m, lab):
            self.selection.set(None if m is None else m.copy(), lab)
            self.selectionChanged.emit()
            self.changed.emit()

        cost = (0 if old is None else old.nbytes) + (0 if new is None else new.nbytes)
        self.undo.push(UndoCommand(label, lambda: put(old, old_label), lambda: put(new, new_label), cost))

    def apply_selection(self, area: np.ndarray, op: str = "replace", label: str = "Select") -> None:
        result = combine(self.selection.mask, area, op)
        if op == "replace" or not self.selection.label or self.selection.mask is None:
            new_label = label
        else:
            new_label = "Custom selection"
        self.set_selection(result, new_label if result is not None else "", record=True)

    def selection_label(self) -> str:
        if not self.selection.active:
            return ""
        if self.panels is not None:
            cov = self.panels.coverage(self.selection.mask)
            full = [self.panels.infos[i].code for i, c in enumerate(cov) if c >= 0.9]
            part = [i for i, c in enumerate(cov) if 0.02 < c < 0.9]
            if full and not part:
                return ", ".join(full)
        return self.selection.label or "Custom selection"

    # ------------------------------------------------------------------ caches
    def _simple(self, layers: List[Layer]) -> bool:
        return all(l.blend == "Normal" for l in layers if l.visible)

    def caches(self):
        """(below_image, above_image_or_None, above_layers_if_not_cached)."""
        if not self._cache_valid:
            a = self.active_index
            below = self.layers[:a] if a >= 0 else self.layers
            above = self.layers[a + 1:] if a >= 0 else []
            self._below = composite(below, self.width, self.height)
            self._above = composite(above, self.width, self.height) if above and self._simple(above) else None
            self._cache_valid = True
        a = self.active_index
        above_layers = self.layers[a + 1:] if a >= 0 else []
        return self._below, self._above, above_layers

    def flatten(self, include_hidden: bool = False) -> QImage:
        layers = self.layers if not include_hidden else [l for l in self.layers]
        return composite(layers, self.width, self.height)

    def sample_color(self, x: int, y: int) -> QColor:
        img = new_image(1, 1)
        p = QPainter(img)
        p.translate(-x, -y)
        for l in self.layers:
            if l.visible:
                l.paint(p)
        p.end()
        return img.pixelColor(0, 0)

    # ------------------------------------------------------------------ layer ops (undoable)
    def add_layer(self, layer: Layer, index: Optional[int] = None, label: str = "Add layer") -> None:
        if index is None:
            index = self.active_index + 1 if self.active_index >= 0 else len(self.layers)
        index = max(0, min(index, len(self.layers)))
        prev_active = self.active_index

        def redo():
            self.layers.insert(index, layer)
            self.active_index = index
            self._structure_changed()

        def undo():
            self.layers.remove(layer)
            self.active_index = min(prev_active, len(self.layers) - 1)
            self._structure_changed()

        redo()
        self.undo.push(UndoCommand(label, undo, redo, _img_bytes(getattr(layer, "image", None))))

    def remove_layer(self, index: int) -> None:
        if not (0 <= index < len(self.layers)) or len(self.layers) <= 1:
            return
        layer = self.layers[index]

        def redo():
            self.layers.remove(layer)
            self.active_index = min(index, len(self.layers) - 1)
            self._structure_changed()

        def undo():
            self.layers.insert(index, layer)
            self.active_index = index
            self._structure_changed()

        redo()
        self.undo.push(UndoCommand("Delete layer", undo, redo, _img_bytes(getattr(layer, "image", None))))

    def move_layer(self, index: int, delta: int) -> None:
        new = index + delta
        if not (0 <= index < len(self.layers)) or not (0 <= new < len(self.layers)):
            return

        def swap(a, b):
            self.layers[a], self.layers[b] = self.layers[b], self.layers[a]
            self.active_index = b
            self._structure_changed()

        swap(index, new)
        self.undo.push(UndoCommand("Move layer", lambda: swap(new, index), lambda: swap(index, new)))

    def reorder_layer(self, src: int, dst: int) -> None:
        """Drag & drop in the layers panel (one undo step)."""
        if src == dst or not (0 <= src < len(self.layers)) or not (0 <= dst < len(self.layers)):
            return
        before = list(self.layers)
        after = list(self.layers)
        layer = after.pop(src)
        after.insert(dst, layer)

        def put(order, active):
            self.layers = list(order)
            self.active_index = active
            self._structure_changed()

        put(after, dst)
        self.undo.push(UndoCommand("Reorder layers", lambda: put(before, src), lambda: put(after, dst)))

    def duplicate_layer(self, index: int) -> None:
        if 0 <= index < len(self.layers):
            c = clone_layer(self.layers[index])
            c.name = self.layers[index].name + " copy"
            if isinstance(c, ItemLayer):
                c.x += 40
                c.y += 40
                c.clamp_to_clip()
            self.add_layer(c, index + 1, "Duplicate layer")

    def layer_via_copy(self, index: int, cut: bool = False) -> bool:
        """Photoshop's Ctrl+J with a selection: the selected pixels of a paint layer become a new layer."""
        layer = self.layers[index] if 0 <= index < len(self.layers) else None
        sel = self.selection
        if not isinstance(layer, RasterLayer) or not sel.active:
            return False
        r = sel.bbox()
        if r is None:
            return False
        from src.ui.editor.masks import apply_mask_to_image

        piece = new_image(self.width, self.height)
        crop = apply_mask_to_image(layer.image.copy(r), sel.mask[rect_slices(r)])
        p = QPainter(piece)
        p.drawImage(r.topLeft(), crop)
        p.end()
        new = RasterLayer(f"{layer.name} ({'cut' if cut else 'copy'})", piece)
        if cut and not layer.locked:
            before = layer.image.copy(r)
            p = QPainter(layer.image)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(r, Qt.GlobalColor.transparent)
            p.end()
            self.commit_raster(layer, r, before, "Cut selection")
        self.add_layer(new, index + 1, "Layer via copy")
        return True

    def rasterize_layer(self, index: int) -> None:
        layer = self.layers[index] if 0 <= index < len(self.layers) else None
        if not isinstance(layer, ItemLayer):
            return
        raster = rasterize_item(layer, self.width, self.height)

        def redo():
            self.layers[index] = raster
            self.active_index = index
            self._structure_changed()

        def undo():
            self.layers[index] = layer
            self.active_index = index
            self._structure_changed()

        redo()
        self.undo.push(UndoCommand("Rasterize layer", undo, redo, _img_bytes(raster.image)))

    def merge_down(self, index: int) -> None:
        if index <= 0 or index >= len(self.layers):
            return
        upper, lower = self.layers[index], self.layers[index - 1]
        lower_r = RasterLayer(lower.name, bake_layer(lower, self.width, self.height))
        props = lower.props()
        lower_r.set_props({k: props[k] for k in ("name", "visible", "opacity", "blend", "locked")})
        p = QPainter(lower_r.image)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if upper.visible:
            paint_exact(upper, p, self.width, self.height)
        p.end()

        def redo():
            self.layers[index - 1:index + 1] = [lower_r]
            self.active_index = index - 1
            self._structure_changed()

        def undo():
            self.layers[index - 1:index] = [lower, upper]
            self.active_index = index
            self._structure_changed()

        redo()
        self.undo.push(UndoCommand("Merge down", undo, redo, _img_bytes(lower_r.image)))

    def set_layer_props(self, layer: Layer, before: Dict, after: Dict, label: str = "Layer properties") -> None:
        """Pushes an already-applied property change (use for sliders: capture before on press)."""
        if before == after:
            return

        def apply(d):
            layer.set_props(d)
            self._props_changed()

        apply(after)
        self.undo.push(UndoCommand(label, lambda: apply(before), lambda: apply(after)))

    def set_layer_clip(self, layer: Layer, clip: Optional[LayerClip], label: str = "Clip layer") -> None:
        old = layer.clip

        def apply(c):
            layer.clip = c
            if isinstance(layer, ItemLayer):
                layer.clamp_to_clip()
            self._props_changed()

        apply(clip)
        cost = (0 if clip is None else clip.mask.nbytes) + (0 if old is None else old.mask.nbytes)
        self.undo.push(UndoCommand(label, lambda: apply(old), lambda: apply(clip), cost))

    def commit_raster(self, layer: RasterLayer, rect: QRect, before: QImage, label: str,
                      masked: bool = True) -> None:
        """
        `before` is the crop of `rect` taken before the edit; the edit is already on layer.image.
        With a selection active (and masked=True) everything outside the selection is restored, so every
        paint tool automatically stays inside the selected panels.
        """
        rect = rect.intersected(QRect(0, 0, self.width, self.height))
        if rect.isEmpty():
            return
        if before.size() != rect.size():
            before = before.copy(0, 0, rect.width(), rect.height())
        after = layer.image.copy(rect)
        if masked and self.selection.active:
            after = blend_masked(before, after, self.selection.mask[rect_slices(rect)])
            p = QPainter(layer.image)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            p.drawImage(rect.topLeft(), after)
            p.end()

        def put(img: QImage):
            p = QPainter(layer.image)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            p.drawImage(rect.topLeft(), img)
            p.end()
            self.invalidate()
            self.mark_dirty()

        self.undo.push(UndoCommand(label, lambda: put(before), lambda: put(after),
                                   _img_bytes(before) + _img_bytes(after)))
        self.invalidate()
        self.mark_dirty()

    def commit_transform(self, layer: ItemLayer, before: Dict, after: Dict) -> None:
        if before == after:
            return

        def apply(d):
            layer.set_transform_state(d)
            self.invalidate()
            self.mark_dirty()

        apply(after)
        self.undo.push(UndoCommand("Transform", lambda: apply(before), lambda: apply(after)))

    def replace_all(self, layers: List[Layer], active: int = -1) -> None:
        self.layers = layers
        self.active_index = active if active >= 0 else len(layers) - 1
        self.undo.clear()
        self._structure_changed()
        self.dirty = False

    # ------------------------------------------------------------------ internals
    def _structure_changed(self) -> None:
        self.mark_dirty()
        self._cache_valid = False
        self.layersChanged.emit()
        self.activeChanged.emit(self.active_index)
        self.changed.emit()

    def _props_changed(self) -> None:
        self.mark_dirty()
        self._cache_valid = False
        self.layersChanged.emit()
        self.changed.emit()

    def notify_props_changed(self) -> None:
        self._props_changed()
