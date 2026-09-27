"""
Layer model for the Livery Paint Editor.

  RasterLayer  - a full-size paintable bitmap (the base livery is one of these)
  ImageLayer   - a placed PNG / sponsor decal with move / scale / rotate / flip
  TextLayer    - live, editable text (race numbers, driver names) with outline, same transform as decals

All coordinates are texture ("document") pixels.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPolygonF,
    QTransform,
)

BLEND_MODES = {
    "Normal": QPainter.CompositionMode.CompositionMode_SourceOver,
    "Multiply": QPainter.CompositionMode.CompositionMode_Multiply,
    "Screen": QPainter.CompositionMode.CompositionMode_Screen,
    "Overlay": QPainter.CompositionMode.CompositionMode_Overlay,
    "Darken": QPainter.CompositionMode.CompositionMode_Darken,
    "Lighten": QPainter.CompositionMode.CompositionMode_Lighten,
}


def new_image(w: int, h: int) -> QImage:
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    return img


class Layer:
    kind = "base"

    def __init__(self, name: str):
        self.uid = uuid.uuid4().hex[:10]
        self.name = name
        self.visible = True
        self.opacity = 1.0
        self.blend = "Normal"
        self.locked = False

    # interface
    def paint(self, painter: QPainter) -> None:  # draws in document coordinates
        raise NotImplementedError

    def is_item(self) -> bool:
        return False

    def props(self) -> Dict:
        return {"name": self.name, "visible": self.visible, "opacity": self.opacity,
                "blend": self.blend, "locked": self.locked}

    def set_props(self, d: Dict) -> None:
        for k in ("name", "visible", "opacity", "blend", "locked"):
            if k in d:
                setattr(self, k, d[k])

    def _apply_common(self, painter: QPainter) -> None:
        painter.setOpacity(max(0.0, min(1.0, self.opacity)))
        painter.setCompositionMode(BLEND_MODES.get(self.blend, BLEND_MODES["Normal"]))


class RasterLayer(Layer):
    kind = "raster"

    def __init__(self, name: str, image: QImage):
        super().__init__(name)
        if image.format() != QImage.Format.Format_ARGB32_Premultiplied:
            image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
        self.image = image

    def paint(self, painter: QPainter) -> None:
        painter.save()
        self._apply_common(painter)
        painter.drawImage(0, 0, self.image)
        painter.restore()


class ItemLayer(Layer):
    """Common transform logic for things you place and move around (decals, text)."""

    def __init__(self, name: str):
        super().__init__(name)
        self.x = 0.0  # centre in document px
        self.y = 0.0
        self.sx = 1.0
        self.sy = 1.0
        self.rotation = 0.0  # degrees
        self.flip_h = False
        self.flip_v = False

    def is_item(self) -> bool:
        return True

    def source(self) -> QImage:
        raise NotImplementedError

    def source_size(self):
        s = self.source()
        return s.width(), s.height()

    def transform(self) -> QTransform:
        w, h = self.source_size()
        t = QTransform()
        t.translate(self.x, self.y)
        t.rotate(self.rotation)
        t.scale(self.sx * (-1 if self.flip_h else 1), self.sy * (-1 if self.flip_v else 1))
        t.translate(-w / 2.0, -h / 2.0)
        return t

    def corners(self) -> List[QPointF]:
        w, h = self.source_size()
        t = self.transform()
        return [t.map(QPointF(0, 0)), t.map(QPointF(w, 0)), t.map(QPointF(w, h)), t.map(QPointF(0, h))]

    def contains(self, pt: QPointF) -> bool:
        return QPolygonF(self.corners()).containsPoint(pt, Qt.FillRule.OddEvenFill)

    def bounding_rect(self) -> QRectF:
        return QPolygonF(self.corners()).boundingRect()

    def paint(self, painter: QPainter) -> None:
        src = self.source()
        if src.isNull():
            return
        painter.save()
        self._apply_common(painter)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setTransform(self.transform(), True)
        painter.drawImage(0, 0, src)
        painter.restore()

    def transform_state(self) -> Dict:
        return {"x": self.x, "y": self.y, "sx": self.sx, "sy": self.sy,
                "rotation": self.rotation, "flip_h": self.flip_h, "flip_v": self.flip_v}

    def set_transform_state(self, d: Dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)

    def props(self) -> Dict:
        d = super().props()
        d.update(self.transform_state())
        return d

    def set_props(self, d: Dict) -> None:
        super().set_props(d)
        self.set_transform_state({k: d[k] for k in ("x", "y", "sx", "sy", "rotation", "flip_h", "flip_v") if k in d})


class ImageLayer(ItemLayer):
    kind = "image"

    def __init__(self, name: str, image: QImage):
        super().__init__(name)
        self.image = image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)

    def source(self) -> QImage:
        return self.image


class TextLayer(ItemLayer):
    kind = "text"

    def __init__(self, name: str = "Text", text: str = "42"):
        super().__init__(name)
        self.text = text
        self.font_family = "Arial"
        self.font_size = 220
        self.bold = True
        self.italic = False
        self.color = QColor(255, 255, 255)
        self.outline_color = QColor(0, 0, 0)
        self.outline_width = 0
        self.letter_spacing = 0.0
        self._cache: Optional[QImage] = None
        self._cache_key = None

    def _key(self):
        return (self.text, self.font_family, self.font_size, self.bold, self.italic,
                self.color.rgba(), self.outline_color.rgba(), self.outline_width, self.letter_spacing)

    def invalidate(self) -> None:
        self._cache = None

    def source(self) -> QImage:
        key = self._key()
        if self._cache is not None and self._cache_key == key:
            return self._cache
        font = QFont(self.font_family)
        font.setPixelSize(max(4, int(self.font_size)))
        font.setBold(self.bold)
        font.setItalic(self.italic)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, self.letter_spacing)

        lines = (self.text or " ").split("\n")
        fm = QFontMetricsF(font)
        line_h = fm.lineSpacing()
        width = max(fm.horizontalAdvance(line) for line in lines)
        pad = self.outline_width + max(4, self.font_size * 0.08)
        w = int(width + pad * 2) + 2
        h = int(line_h * len(lines) + pad * 2) + 2

        path = QPainterPath()
        for i, line in enumerate(lines):
            lw = fm.horizontalAdvance(line)
            path.addText(QPointF(pad + (width - lw) / 2.0, pad + fm.ascent() + i * line_h), font, line)

        img = new_image(max(1, w), max(1, h))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.outline_width > 0:
            stroker = QPainterPathStroker()
            stroker.setWidth(self.outline_width * 2)
            stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.fillPath(stroker.createStroke(path).united(path), QBrush(self.outline_color))
        p.fillPath(path, QBrush(self.color))
        p.end()

        self._cache = img
        self._cache_key = key
        return img

    def text_props(self) -> Dict:
        return {"text": self.text, "font_family": self.font_family, "font_size": self.font_size,
                "bold": self.bold, "italic": self.italic, "color": self.color.name(QColor.NameFormat.HexArgb),
                "outline_color": self.outline_color.name(QColor.NameFormat.HexArgb),
                "outline_width": self.outline_width, "letter_spacing": self.letter_spacing}

    def set_text_props(self, d: Dict) -> None:
        for k, v in d.items():
            if k in ("color", "outline_color"):
                v = QColor(v) if not isinstance(v, QColor) else v
            setattr(self, k, v)
        self.invalidate()

    def props(self) -> Dict:
        d = super().props()
        d.update(self.text_props())
        return d

    def set_props(self, d: Dict) -> None:
        super().set_props(d)
        self.set_text_props({k: d[k] for k in self.text_props().keys() if k in d})


# ------------------------------------------------------------------------------
# Rasterising / compositing helpers
# ------------------------------------------------------------------------------


def composite(layers: List[Layer], w: int, h: int, background: Optional[QColor] = None) -> QImage:
    img = new_image(w, h)
    if background is not None:
        img.fill(background)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    for layer in layers:
        if layer.visible:
            layer.paint(p)
    p.end()
    return img


def rasterize_item(layer: ItemLayer, w: int, h: int) -> RasterLayer:
    img = new_image(w, h)
    p = QPainter(img)
    op, blend = layer.opacity, layer.blend
    layer.opacity, layer.blend = 1.0, "Normal"
    layer.paint(p)
    layer.opacity, layer.blend = op, blend
    p.end()
    r = RasterLayer(layer.name, img)
    r.opacity, r.blend, r.visible = op, blend, layer.visible
    return r


# ------------------------------------------------------------------------------
# Project (layered file) save / load  -  a zip with manifest.json + PNGs
# ------------------------------------------------------------------------------

PROJECT_EXT = ".ams2paint"


def _qimage_png_bytes(img: QImage) -> bytes:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def _qimage_from_bytes(data: bytes) -> QImage:
    img = QImage()
    img.loadFromData(data, "PNG")
    return img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


@dataclass
class ProjectData:
    width: int
    height: int
    layers: List[Layer]
    guide_png: Optional[bytes] = None
    meta: Optional[Dict] = None


def save_project(path, data: ProjectData) -> None:
    manifest = {"version": 1, "width": data.width, "height": data.height, "layers": [], "meta": data.meta or {}}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, layer in enumerate(data.layers):
            entry = {"kind": layer.kind, "props": layer.props()}
            if isinstance(layer, (RasterLayer, ImageLayer)):
                fname = f"layer_{i:02d}.png"
                zf.writestr(fname, _qimage_png_bytes(layer.image))
                entry["file"] = fname
            manifest["layers"].append(entry)
        if data.guide_png:
            zf.writestr("guide.png", data.guide_png)
            manifest["guide"] = "guide.png"
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))


def load_project(path) -> ProjectData:
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        layers: List[Layer] = []
        for entry in manifest["layers"]:
            kind, props = entry["kind"], entry.get("props", {})
            if kind == "raster":
                layer: Layer = RasterLayer(props.get("name", "Layer"), _qimage_from_bytes(zf.read(entry["file"])))
            elif kind == "image":
                layer = ImageLayer(props.get("name", "Decal"), _qimage_from_bytes(zf.read(entry["file"])))
            elif kind == "text":
                layer = TextLayer(props.get("name", "Text"))
            else:
                continue
            layer.set_props(props)
            layers.append(layer)
        guide = zf.read(manifest["guide"]) if manifest.get("guide") else None
    return ProjectData(manifest["width"], manifest["height"], layers, guide, manifest.get("meta"))


def clone_layer(layer: Layer) -> Layer:
    if isinstance(layer, RasterLayer):
        c: Layer = RasterLayer(layer.name, layer.image.copy())
    elif isinstance(layer, ImageLayer):
        c = ImageLayer(layer.name, layer.image.copy())
    elif isinstance(layer, TextLayer):
        c = TextLayer(layer.name, layer.text)
    else:
        raise TypeError(layer)
    c.set_props(layer.props())
    return c
