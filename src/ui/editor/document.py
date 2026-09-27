"""
EditorDocument: the layer stack, the template guide, cached composites and the undo history.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter

from src.ui.editor.layers import (
    ItemLayer,
    Layer,
    RasterLayer,
    clone_layer,
    composite,
    new_image,
    rasterize_item,
)
from src.ui.editor.undo import UndoCommand, UndoStack


def _img_bytes(img: Optional[QImage]) -> int:
    return 0 if img is None else img.sizeInBytes()


class EditorDocument(QObject):
    changed = Signal()          # something visible changed -> repaint
    layersChanged = Signal()    # structure / names / visibility changed -> refresh layer list
    activeChanged = Signal(int)

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

    # ------------------------------------------------------------------ guide
    def set_guide(self, img: Optional[QImage]) -> None:
        if img is not None and not img.isNull():
            if img.width() != self.width or img.height() != self.height:
                img = img.scaled(self.width, self.height, Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            img = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        self.guide = img
        self.changed.emit()

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

    def duplicate_layer(self, index: int) -> None:
        if 0 <= index < len(self.layers):
            c = clone_layer(self.layers[index])
            c.name = self.layers[index].name + " copy"
            if isinstance(c, ItemLayer):
                c.x += 40
                c.y += 40
            self.add_layer(c, index + 1, "Duplicate layer")

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
        if not isinstance(lower, RasterLayer):
            lower_r = rasterize_item(lower, self.width, self.height)  # type: ignore[arg-type]
        else:
            lower_r = RasterLayer(lower.name, lower.image.copy())
            lower_r.set_props(lower.props())
        p = QPainter(lower_r.image)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if upper.visible:
            upper.paint(p)
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

    def commit_raster(self, layer: RasterLayer, rect: QRect, before: QImage, label: str) -> None:
        """`before` is the crop of `rect` taken before the edit; the edit is already on layer.image."""
        rect = rect.intersected(QRect(0, 0, self.width, self.height))
        if rect.isEmpty():
            return
        after = layer.image.copy(rect)

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
