"""
Paint Editor: template panels, selection-clipped painting, panel-clipped decals, tabs integration.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

S = 512
LEFT_BOX = [40, 60, 230, 250]     # panel A
RIGHT_BOX = [280, 60, 470, 250]   # panel B
BOTTOM_BOX = [40, 300, 470, 470]  # panel C


def make_template() -> Image.Image:
    img = Image.new("RGBA", (S, S), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    for box in (LEFT_BOX, RIGHT_BOX, BOTTOM_BOX):
        d.rectangle(box, outline=(0, 0, 0, 255), width=3)
    return img


class PaintPanelsBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def make_editor(self):
        from src.ui.editor.livery_editor import PaintEditorWidget

        ed = PaintEditorWidget(embedded=False)
        ed.resize(1300, 850)
        ed.load(Image.new("RGBA", (S, S), (30, 60, 200, 255)), make_template())
        ed.show()
        QApplication.processEvents()
        pm = ed.doc.ensure_panels()
        self.assertIsNotNone(pm, "template panels should be detected")
        self.addCleanup(self._close, ed)
        return ed

    @staticmethod
    def _close(ed):
        ed.doc.dirty = False
        ed.close()
        ed.deleteLater()

    def click(self, ed, x, y, mods=Qt.KeyboardModifier.NoModifier):
        c = ed.canvas
        QTest.mouseClick(c, Qt.MouseButton.LeftButton, mods, c.to_widget(QPointF(x, y)).toPoint())

    def drag(self, ed, a, b, steps=12):
        c = ed.canvas
        pa, pb = c.to_widget(QPointF(*a)).toPoint(), c.to_widget(QPointF(*b)).toPoint()
        QTest.mousePress(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pa)
        for i in range(1, steps + 1):
            QTest.mouseMove(c, pa + (pb - pa) * (i / steps))
        QTest.mouseRelease(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pb)


class TestPanelSelection(PaintPanelsBase):
    def test_panels_detected_and_click_select(self):
        from src.ui.editor.canvas import Tool

        ed = self.make_editor()
        pm = ed.doc.panels
        self.assertEqual(len(pm), 3)
        a = pm.index_at(130, 150)
        b = pm.index_at(370, 150)
        self.assertGreaterEqual(a, 0)
        self.assertNotEqual(a, b)

        ed._set_tool(Tool.PANEL)
        self.click(ed, 130, 150)
        sel = ed.doc.selection
        self.assertTrue(sel.active)
        self.assertTrue(sel.contains(130, 150))
        self.assertFalse(sel.contains(370, 150))
        # Shift+click adds a second panel
        self.click(ed, 370, 150, Qt.KeyboardModifier.ShiftModifier)
        self.assertTrue(sel.contains(370, 150) and sel.contains(130, 150))
        cov = pm.coverage(sel.mask)
        self.assertEqual(sum(1 for c in cov if c >= 0.9), 2)
        # Alt+click removes one
        self.click(ed, 130, 150, Qt.KeyboardModifier.AltModifier)
        self.assertFalse(sel.contains(130, 150))
        self.assertTrue(sel.contains(370, 150))
        # selection changes are undoable
        ed._undo()
        self.assertTrue(sel.contains(130, 150))
        # clicking the background deselects
        self.click(ed, 5, 5)
        self.assertFalse(sel.active)

    def test_panels_dock_checkboxes_select(self):
        ed = self.make_editor()
        self.assertEqual(ed.list_panels.count(), 3)
        item = ed.list_panels.item(0)
        item.setCheckState(Qt.CheckState.Checked)
        QApplication.processEvents()
        idx = int(item.data(Qt.ItemDataRole.UserRole))
        cov = ed.doc.panels.coverage(ed.doc.selection.mask)
        self.assertGreaterEqual(cov[idx], 0.9)


class TestClippedPainting(PaintPanelsBase):
    def test_brush_stays_inside_selected_panel(self):
        from src.ui.editor.canvas import Tool

        ed = self.make_editor()
        ed._set_tool(Tool.PANEL)
        self.click(ed, 130, 150)
        ed._set_tool(Tool.BRUSH)
        ed.ss_size.setValue(40)
        ed._set_primary(QColor(255, 0, 0))
        self.drag(ed, (10, 150), (500, 150))  # stroke crosses both top panels
        paint = ed.doc.layers[1].image
        self.assertGreater(paint.pixelColor(130, 150).alpha(), 200)   # inside the selected panel
        self.assertEqual(paint.pixelColor(370, 150).alpha(), 0)       # other panel untouched
        self.assertEqual(paint.pixelColor(255, 150).alpha(), 0)       # gap between panels untouched

    def test_soft_eraser_and_fill_respect_selection(self):
        from src.ui.editor.canvas import Tool

        ed = self.make_editor()
        ed._set_primary(QColor(0, 200, 0))
        ed._fill_selection(True)  # no selection: whole layer
        paint = ed.doc.layers[1].image
        self.assertEqual(paint.pixelColor(370, 150).green(), 200)
        ed._set_tool(Tool.PANEL)
        self.click(ed, 370, 150)
        ed._set_tool(Tool.ERASER)
        ed.ss_size.setValue(60)
        ed.ss_hard.setValue(20)
        self.drag(ed, (10, 150), (500, 150))
        self.assertLess(paint.pixelColor(370, 150).alpha(), 30)    # erased inside panel B
        self.assertEqual(paint.pixelColor(130, 150).alpha(), 255)  # panel A untouched
        # bucket outside the selection does nothing
        ed._set_tool(Tool.FILL)
        ed._set_primary(QColor(255, 255, 0))
        self.click(ed, 130, 150)
        self.assertEqual(paint.pixelColor(130, 150).red(), 0)

    def test_gradient_and_delete_inside_selection(self):
        from src.ui.editor.canvas import Tool

        ed = self.make_editor()
        ed._set_tool(Tool.PANEL)
        self.click(ed, 250, 400)  # bottom panel
        ed._set_tool(Tool.GRADIENT)
        self.drag(ed, (50, 400), (460, 400))
        paint = ed.doc.layers[1].image
        self.assertGreater(paint.pixelColor(250, 400).alpha(), 200)
        self.assertEqual(paint.pixelColor(130, 150).alpha(), 0)
        ed._delete_key()
        self.assertEqual(paint.pixelColor(250, 400).alpha(), 0)

    def test_move_pixels_stays_within_selection(self):
        from src.ui.editor.canvas import Tool

        ed = self.make_editor()
        ed._set_primary(QColor(255, 0, 0))
        ed._fill_selection(True)  # whole paint layer red
        ed._set_tool(Tool.PANEL)
        self.click(ed, 130, 150)
        ed._set_tool(Tool.MOVE)
        self.drag(ed, (130, 150), (330, 150))  # try to drag panel A's paint over panel B
        paint = ed.doc.layers[1].image
        self.assertEqual(paint.pixelColor(370, 150).alpha(), 255)  # B keeps its own paint (unchanged)
        self.assertEqual(paint.pixelColor(255, 150).red(), 255)    # outside the selection: untouched
        self.assertEqual(paint.pixelColor(100, 150).alpha(), 0)    # A's content moved away -> hole


class TestClippedDecals(PaintPanelsBase):
    def _logo(self):
        img = QImage(200, 100, QImage.Format.Format_ARGB32)
        img.fill(QColor(255, 255, 0))
        return img

    def test_paste_into_selected_panel_is_clipped_and_saved(self):
        from src.ui.editor.canvas import Tool
        from src.ui.editor.layers import load_project, save_project

        ed = self.make_editor()
        ed._set_tool(Tool.PANEL)
        self.click(ed, 130, 150)
        ed._place_image(self._logo(), "sponsor")
        layer = ed.doc.active
        self.assertEqual(layer.kind, "image")
        self.assertIsNotNone(layer.clip)
        # centred in the panel
        self.assertAlmostEqual(layer.x, (LEFT_BOX[0] + LEFT_BOX[2]) / 2, delta=6)
        # make it huge so it would cover the other panel if it weren't clipped
        layer.sx = layer.sy = 6.0
        ed.doc.invalidate()
        ed._apply()
        out = ed.result_image()
        self.assertEqual(out.getpixel((130, 150))[:3], (255, 255, 0))
        self.assertNotEqual(out.getpixel((370, 150))[:3], (255, 255, 0))
        self.assertNotEqual(out.getpixel((255, 400))[:3], (255, 255, 0))
        # dragging it keeps its centre inside the panel
        ed._set_tool(Tool.MOVE)
        self.drag(ed, (130, 150), (480, 480))
        r = layer.clip.bbox()
        self.assertTrue(r.left() <= layer.x <= r.right() + 1 and r.top() <= layer.y <= r.bottom() + 1)
        # the clip survives a save / load
        tmp = Path(tempfile.mkdtemp()) / "p.ams2paint"
        try:
            save_project(tmp, ed.project_data())
            data = load_project(tmp)
            self.assertIsNotNone(data.layers[-1].clip)
            self.assertEqual(data.layers[-1].clip.mask.shape, (S, S))
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

    def test_release_clip_and_layer_via_copy(self):
        from src.ui.editor.canvas import Tool

        ed = self.make_editor()
        ed._set_tool(Tool.PANEL)
        self.click(ed, 130, 150)
        ed._place_image(self._logo(), "sponsor")
        layer = ed.doc.active
        ed._release_clip()
        self.assertIsNone(layer.clip)
        ed._undo()
        self.assertIsNotNone(layer.clip)
        # layer via copy: only the selected pixels go to the new layer
        ed.doc.set_active(1)
        ed._set_primary(QColor(255, 0, 0))
        ed._select_all()
        ed._fill_selection(True)
        ed.doc.set_selection(ed.doc.panels.mask([ed.doc.panels.index_at(370, 150)]), "B")
        n = len(ed.doc.layers)
        ed._layer_via_copy()
        self.assertEqual(len(ed.doc.layers), n + 1)
        new = ed.doc.active.image
        self.assertEqual(new.pixelColor(370, 150).alpha(), 255)
        self.assertEqual(new.pixelColor(130, 150).alpha(), 0)

    def test_snapshot_layers_are_independent(self):
        ed = self.make_editor()
        data = ed.project_data()
        ed._set_primary(QColor(255, 0, 0))
        ed._fill_selection(True)
        self.assertEqual(data.layers[1].image.pixelColor(10, 10).alpha(), 0)  # stored version unchanged


class TestMainWindowTabs(PaintPanelsBase):
    def test_paint_tab_edits_current_version_and_applies_new_one(self):
        os.environ.setdefault("HOME", tempfile.mkdtemp())
        from src.ui.main_window import MainWindow

        w = MainWindow()
        self.addCleanup(lambda: (setattr(w.paint_editor.doc, "dirty", False), w.close()))
        w.resize(1400, 900)
        w.show()
        w.preview_panel.reset_revisions(Image.new("RGBA", (S, S), (30, 60, 200, 255)), label="v1 (Initial)")
        w.tabs.setCurrentWidget(w.paint_editor)
        QApplication.processEvents()
        pe = w.paint_editor
        self.assertTrue(pe.has_content())
        self.assertIs(pe.source_key, w.preview_panel.current_revision())
        # editor menus only show on the Paint tab
        self.assertTrue(all(a.isVisible() for a in w._editor_menu_actions))
        pe._set_primary(QColor(255, 0, 0))
        pe._fill_selection(True)
        self.assertTrue(pe.is_dirty())
        pe._apply()
        self.assertEqual(len(w.preview_panel.revisions), 2)
        self.assertEqual(w.preview_panel.current_revision().image.getpixel((5, 5))[:3], (255, 0, 0))
        self.assertFalse(pe.is_dirty())
        self.assertIs(pe.source_key, w.preview_panel.current_revision())
        w.tabs.setCurrentIndex(0)
        self.assertFalse(any(a.isVisible() for a in w._editor_menu_actions))


if __name__ == "__main__":
    unittest.main()
