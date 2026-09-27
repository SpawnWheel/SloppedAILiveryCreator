"""Tests for the 'Copy for Other AI' prompt pack export and external-result import."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from PySide6.QtWidgets import QApplication

from src.core.ai_engine import AIEngine
from src.core.prompt_pack import (
    PACK_INFO_FILE, create_prompt_pack, find_pack_info, normalize_external_result,
)


class TestPromptPack(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # non-square, transparent template -> must be flattened + padded like the Gemini path
        self.template = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
        for x in range(50, 150):
            self.template.putpixel((x, 100), (0, 0, 0, 255))
        self.refs = [(Image.new("RGB", (3000, 1500), (200, 0, 0)), "Color Palette / Theme"),
                     (Image.new("RGBA", (64, 64), (0, 0, 255, 255)), "Sponsor / Decal")]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pack(self, **kw):
        user = AIEngine.build_default_user_prompt(user_guidance="Gulf blue, orange stripe, #42",
                                                  reference_roles=[r for _, r in self.refs])
        return create_prompt_pack(kw.get("wireframe", self.template), kw.get("refs", self.refs), user,
                                  AIEngine.build_default_system_instruction(), root=self.tmp, name="Gulf #42")

    def test_folder_contents(self):
        pack = self._pack()
        self.assertTrue(pack.folder.is_dir())
        names = [p.name for p in pack.image_paths]
        self.assertEqual(names, ["01_template.png", "02_color_palette_theme.png", "03_sponsor_decal.png"])
        self.assertTrue((pack.folder / "prompt.txt").exists())
        self.assertTrue((pack.folder / PACK_INFO_FILE).exists())
        tpl = Image.open(pack.image_paths[0])
        self.assertEqual(tpl.mode, "RGB")                      # flattened, not a black square
        self.assertAlmostEqual(tpl.size[0] / tpl.size[1], 2.0, delta=0.3)  # padded to a supported ratio
        self.assertLessEqual(max(Image.open(pack.image_paths[1]).size), 2048)

    def test_prompt_is_self_contained(self):
        pack = self._pack()
        t = pack.prompt_text
        self.assertIn("GENERATE AN IMAGE", t)
        self.assertIn("Gulf blue", t)
        self.assertIn("GEOMETRY LOCK", t)
        self.assertIn("FLAT UNLIT COLOUR", t)
        self.assertIn("expert automotive texture artist", t)   # system rules merged in
        self.assertIn("Image 1 = 01_template.png", t)
        self.assertEqual(t, (pack.folder / "prompt.txt").read_text(encoding="utf-8"))

    def test_without_template(self):
        pack = self._pack(wireframe=None)
        self.assertNotIn("GEOMETRY LOCK", pack.prompt_text)
        self.assertEqual(pack.image_paths[0].name, "01_color_palette_theme.png")

    def test_import_undoes_padding(self):
        pack = self._pack()
        info = find_pack_info(pack.folder / "result.png")
        self.assertIsNotNone(info)
        padded = Image.open(pack.image_paths[0])
        out = normalize_external_result(padded.resize((1024, 512)), info, target_resolution=1024)
        self.assertEqual(out.size, (1024, 512))
        self.assertEqual(out.mode, "RGBA")
        sq = normalize_external_result(Image.new("RGB", (300, 300)), None, target_resolution=512)
        self.assertEqual(sq.size, (512, 512))


class TestPromptPackUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_clipboard_and_dialog(self):
        from src.ui.prompt_pack_dialog import PromptPackDialog, copy_pack_to_clipboard
        tmp = Path(tempfile.mkdtemp())
        try:
            pack = create_prompt_pack(Image.new("RGB", (256, 256), "white"),
                                      [(Image.new("RGB", (32, 32), "red"), "Theme")], "brief", "rules", root=tmp)
            copy_pack_to_clipboard(pack, "all")
            mime = QApplication.clipboard().mimeData()
            self.assertIn("brief", mime.text())
            self.assertEqual(len(mime.urls()), 2)
            copy_pack_to_clipboard(pack, "text")
            self.assertFalse(QApplication.clipboard().mimeData().hasUrls())
            dlg = PromptPackDialog(pack)
            self.assertEqual(dlg.image_list.count(), 2)
            dlg.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_preview_button(self):
        from src.ui.preview_panel import PreviewPanel
        p = PreviewPanel()
        hits = []
        p.promptPackRequested.connect(lambda: hits.append(1))
        p.btn_prompt_pack.click()
        self.assertEqual(hits, [1])
        p.set_busy(True, "Generating")
        self.assertFalse(p.btn_prompt_pack.isEnabled())


if __name__ == "__main__":
    unittest.main()
