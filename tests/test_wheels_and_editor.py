"""
Tests for v1.1: car profiles (_dist.xml parsing), tolerant XML editing, wheel textures,
2048x600 previews and the layered Paint Editor.
"""

import shutil
import sys
import tempfile
import unittest

import numpy as np
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.core.car_profile import load_car_profile, parse_dist_text
from src.core.image_processor import ImageProcessor
from src.core.installer import LiveryInstaller
from src.core.xml_manager import XMLManager

# Trimmed copy of a real AMS2 _dist.xml, INCLUDING Reiza's stray </LIVERY_OVERRIDE> that makes it invalid XML
DIST_SAMPLE = """<?xml version="1.0" encoding="utf-8" ?>
<USER_OVERRIDES>
    <!--
        Additionally a PREVIEWIMAGE element with a PATH attribute can be used to note a texture for use
        in the game's UI. (2048 x 600 px )

AVAILABLE BASE LIVERIES
        =======================

                Default: Default body only skin
                Available textures:
                        BODY        - the main external body/window paint texture
                                    - 4096 x 4096
                        WHEELDIFF   - rim_diff - Wheel rim texture
                                    - 1024 x 1024
                    WHEELBLURDIFF   - rim_blur_diff - Wheel rim blur texture
                                    - 1024 x 1024
                Matte: Matte metallic effect
                Available textures:
                        BODY        - the main external body/window paint texture

        HELMETS
        =======================
                DEFAULT: default helmet used by all vehicles of the class
                        VISOR       - visor texture
        EXAMPLES
        ================================= -->
    <!-- <LIVERY_OVERRIDE LIVERY="51" NAME="My Racing Team #1" BASELIVERY="Default">
      <TEXTURE NAME="BODY" PATH="myracingteam\\body.dds" />
      OPTION 1: WHITE HEADLIGHTS -->
    <TEXTURE NAME="LIGHTSDIFF" PATH="x.dds" />
    </LIVERY_OVERRIDE>
    <!-- <LIVERY_OVERRIDE LIVERY="59" NAME="Example" BASELIVERY="Default">
      <TEXTURE NAME="BODY" PATH="body12.dds" />
    </LIVERY_OVERRIDE> -->
</USER_OVERRIDES>
"""

VIPER_STYLE = """<USER_OVERRIDES><!--
        AVAILABLE BASE PAINTS
        =======================
                -ViperWhite
                -AdrenalineRed
        WHEELS
        =======================
                -Black (default)
          <TEXTURE NAME="WHEELDIFF" PATH="\\Rims\\bronze_diff.dds" />
        HELMETS
        =======================
--></USER_OVERRIDES>"""


def fake_dds(image, output_dir, base_filename="body", format_type="BC3_UNORM", generate_mipmaps=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{base_filename}.dds"
    image.convert("RGBA").save(out, format="DDS")
    return out


class TestCarProfile(unittest.TestCase):
    def test_parse_standard_dist(self):
        p = parse_dist_text(DIST_SAMPLE, "bmw_m4_gt3")
        self.assertEqual(p.base_liveries[:2], ["Default", "Matte"])
        self.assertTrue(p.supports("WHEELDIFF"))
        self.assertTrue(p.supports("WHEELBLURDIFF"))
        self.assertTrue(p.supports_wheels)
        self.assertNotIn("VISOR", p.texture_names)  # helmet section ignored

    def test_parse_paint_style_dist(self):
        p = parse_dist_text(VIPER_STYLE, "dodge_viper_acr")
        self.assertEqual(p.base_liveries, ["ViperWhite", "AdrenalineRed"])
        self.assertNotIn("Default", p.base_liveries)
        self.assertTrue(p.supports("WHEELDIFF"))

    def test_unknown_car_defaults(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            p = load_car_profile(tmp, "nothing_here")
            self.assertEqual(p.base_liveries, ["Default"])
            self.assertFalse(p.supports_wheels)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTolerantXML(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.car = self.tmp / "bmw_m4_gt3"
        self.car.mkdir()
        (self.car / "bmw_m4_gt3_dist.xml").write_text(DIST_SAMPLE, encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_file_starts_from_dist_and_keeps_comments(self):
        xml = self.car / "bmw_m4_gt3.xml"
        XMLManager.upsert_livery(xml, 51, "Team #51", "t\\body.dds", "t\\preview.dds", "Matte",
                                 {"WHEELDIFF": "t\\wheel_diff.dds"})
        text = xml.read_text(encoding="utf-8")
        self.assertIn("AVAILABLE BASE LIVERIES", text)  # Reiza's docs preserved
        entries = XMLManager.parse_overrides(xml)
        self.assertEqual(len(entries), 1)  # commented examples are NOT treated as liveries
        self.assertEqual(entries[0]["base_livery"], "Matte")
        self.assertEqual(entries[0]["textures"]["WHEELDIFF"], "t\\wheel_diff.dds")

    def test_malformed_file_keeps_other_liveries(self):
        xml = self.car / "bmw_m4_gt3.xml"
        broken = DIST_SAMPLE.replace(
            "</USER_OVERRIDES>",
            '<LIVERY_OVERRIDE LIVERY="60" NAME="Friend Skin" BASELIVERY="Default">\n'
            '  <PREVIEWIMAGE PATH="f\\p.dds" />\n  <TEXTURE NAME="BODY" PATH="f\\b.dds" />\n'
            '  <TEXTURE NAME="BANNER" PATH="f\\banner.dds" />\n</LIVERY_OVERRIDE>\n</USER_OVERRIDES>',
        )
        xml.write_text(broken, encoding="utf-8")
        XMLManager.upsert_livery(xml, 61, "Mine", "m\\body.dds", "m\\preview.dds")
        ids = sorted(e["livery_id"] for e in XMLManager.parse_overrides(xml))
        self.assertEqual(ids, ["60", "61"])
        # updating 60 keeps its BANNER texture that we don't manage
        XMLManager.upsert_livery(xml, 60, "Friend Skin v2", "f\\b2.dds", "f\\p.dds")
        e60 = XMLManager.check_collision(xml, 60)
        self.assertEqual(e60["name"], "Friend Skin v2")
        self.assertEqual(e60["textures"]["BANNER"], "f\\banner.dds")
        self.assertEqual(e60["textures"]["BODY"], "f\\b2.dds")


class TestWheelInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p1 = patch.object(LiveryInstaller, "get_overrides_path", return_value=self.tmp)
        self.p2 = patch("src.core.dds_converter.DDSConverter.convert_image_to_dds", side_effect=fake_dds)
        self.p1.start()
        self.p2.start()

    def tearDown(self):
        self.p1.stop()
        self.p2.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_install_with_wheels_and_base(self):
        body = Image.new("RGBA", (256, 256), (200, 20, 20, 255))
        rim = Image.new("RGBA", (300, 300), (120, 120, 120, 255))
        target, _ = LiveryInstaller.install_livery(
            "bmw_m4_gt3", 70, "Rim Test", "Me", body, "Matte", base_livery="Chrome",
            extra_textures={"WHEELDIFF": rim, "WHEELBLURDIFF": rim},
        )
        self.assertTrue((target / "wheel_diff.dds").exists())
        self.assertTrue((target / "wheel_blur_diff.dds").exists())
        with Image.open(target / "wheel_diff.dds") as w:
            self.assertEqual(w.size, (1024, 1024))
        with Image.open(target / "preview.dds") as pv:
            self.assertEqual(pv.size, (2048, 600))
        entry = XMLManager.check_collision(LiveryInstaller.get_car_xml_path("bmw_m4_gt3"), 70)
        self.assertEqual(entry["base_livery"], "Chrome")
        self.assertIn("WHEELDIFF", entry["textures"])

        tex, prev, meta = LiveryInstaller.load_installed_livery("bmw_m4_gt3", 70)
        self.assertIn("WHEELDIFF", meta["extra_images"])

    def test_tint_keeps_alpha_and_changes_colour(self):
        rim = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        ImageDraw.Draw(rim).rectangle([0, 0, 10, 10], fill=(0, 0, 0, 0))
        out = ImageProcessor.tint_texture(rim, (220, 0, 0), 1.0)
        self.assertEqual(out.getpixel((2, 2))[3], 0)
        r, g, b, a = out.getpixel((40, 40))
        self.assertGreater(r, g + 40)
        self.assertEqual(a, 255)

    def test_dominant_colors_skip_background(self):
        img = Image.new("RGB", (100, 100), (10, 10, 12))
        ImageDraw.Draw(img).rectangle([0, 0, 60, 60], fill=(0, 90, 200))
        cols = ImageProcessor.dominant_colors(img, 3)
        self.assertTrue(any(abs(c[2] - 200) < 30 for c in cols))


class TestPaintEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def _editor(self):
        from src.ui.editor.livery_editor import LiveryPaintEditor

        base = Image.new("RGBA", (512, 512), (30, 60, 200, 255))
        guide = Image.new("RGBA", (512, 512), (255, 255, 255, 255))
        ImageDraw.Draw(guide).rectangle([100, 100, 300, 300], outline=(0, 0, 0, 255), width=3)
        ed = LiveryPaintEditor(base, guide)
        ed.resize(1300, 850)
        ed.show()
        QApplication.processEvents()
        return ed

    def test_bucket_fill_respects_template_and_undo(self):
        from src.ui.editor.canvas import Tool

        ed = self._editor()
        c = ed.canvas
        ed._set_tool(Tool.FILL)
        ed._set_primary(QColor(0, 200, 0))
        QTest.mouseClick(c, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         c.to_widget(QPointF(200, 200)).toPoint())
        paint = ed.doc.layers[1].image
        self.assertEqual(paint.pixelColor(200, 200).green(), 200)
        self.assertEqual(paint.pixelColor(400, 400).alpha(), 0)  # outside the template box untouched
        ed._undo()
        self.assertEqual(ed.doc.layers[1].image.pixelColor(200, 200).alpha(), 0)
        ed.doc.dirty = False
        ed.close()

    def test_decal_text_and_apply(self):
        from src.ui.editor.layers import TextLayer

        ed = self._editor()
        decal = QImage(100, 50, QImage.Format.Format_ARGB32)
        decal.fill(QColor(255, 255, 0))
        ed._place_image(decal, "sponsor", QPointF(400, 400))
        layer = ed.doc.active
        self.assertEqual(layer.kind, "image")
        before = layer.transform_state()
        ed._toggle_flip("flip_h")
        self.assertTrue(layer.flip_h)
        ed._undo()
        self.assertEqual(layer.transform_state(), before)

        t = TextLayer("42", "42")
        t.x, t.y, t.font_size = 150, 400, 60
        ed.doc.add_layer(t)
        ed._apply()
        out = ed.result_image()
        self.assertEqual(out.size, (512, 512))
        self.assertEqual(out.getpixel((400, 400))[:3], (255, 255, 0))
        data = ed.project_data()
        self.assertEqual([l.kind for l in data.layers], ["raster", "raster", "image", "text"])

    def test_project_roundtrip(self):
        from src.ui.editor.layers import load_project, save_project

        ed = self._editor()
        decal = QImage(40, 40, QImage.Format.Format_ARGB32)
        decal.fill(QColor(255, 0, 0))
        ed._place_image(decal, "logo", QPointF(50, 50))
        tmp = Path(tempfile.mkdtemp()) / "p.ams2paint"
        try:
            save_project(tmp, ed.project_data())
            data = load_project(tmp)
            self.assertEqual(len(data.layers), 3)
            self.assertAlmostEqual(data.layers[2].x, 50)
            self.assertIsNotNone(data.guide_png)
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
        ed.doc.dirty = False
        ed.close()


class TestWheelsPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_only_edited_textures_are_installed(self):
        from src.ui.wheels_panel import WheelsPanel

        body = Image.new("RGBA", (128, 128), (0, 80, 200, 255))
        panel = WheelsPanel(ai_engine=None, body_provider=lambda: body)
        card = panel.cards["WHEELDIFF"]
        card.template = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        card.refresh()
        self.assertEqual(panel.get_extra_textures(), {})
        panel._set_tint_color(QColor(200, 0, 0))
        panel._apply_tint()
        extras = panel.get_extra_textures()
        self.assertIn("WHEELDIFF", extras)
        self.assertNotIn("WHEELBLURDIFF", extras)


if __name__ == "__main__":
    unittest.main()


class TestTemplateLock(unittest.TestCase):
    """AI output must land on the template's panels."""

    def _template(self, size=1024):
        tpl = Image.new("RGBA", (size, size), (0, 0, 0, 0))  # black lines on TRANSPARENT (common template style)
        d = ImageDraw.Draw(tpl)
        s = size / 2048
        for poly in ([(100, 100), (900, 120), (950, 700), (120, 650)], [(1100, 150), (1900, 150), (1900, 500), (1100, 500)],
                     [(200, 900), (1800, 900), (1700, 1400), (300, 1400)], [(400, 1550), (900, 1550), (900, 1950), (400, 1950)]):
            d.polygon([(x * s, y * s) for x, y in poly], outline=(0, 0, 0, 255), width=4)
        return tpl, s

    def test_transparent_template_is_visible_to_ai(self):
        from src.core.template_fit import prepare_for_ai

        tpl, _ = self._template()
        sent = prepare_for_ai(tpl).image
        lo, hi = sent.convert("L").getextrema()
        self.assertEqual(lo, 0)          # black panel outlines ...
        self.assertGreater(hi, 200)      # ... on light panels - not a solid black square

    def test_generate_realigns_drifted_output(self):
        from unittest.mock import MagicMock
        from src.core.ai_engine import AIEngine
        from src.core.template_fit import island_mask, lock_to_template, _iou

        tpl, s = self._template()
        drifted = Image.new("RGBA", (1024, 1024), (18, 20, 24, 255))
        d = ImageDraw.Draw(drifted)
        for poly, col in (([(100, 100), (900, 120), (950, 700), (120, 650)], (200, 30, 30)),
                          ([(1100, 150), (1900, 150), (1900, 500), (1100, 500)], (30, 90, 200)),
                          ([(200, 900), (1800, 900), (1700, 1400), (300, 1400)], (240, 200, 0)),
                          ([(400, 1550), (900, 1550), (900, 1950), (400, 1950)], (30, 160, 60))):
            d.polygon([(x * s, y * s) for x, y in poly], fill=col)
        drifted = drifted.transform((1024, 1024), Image.AFFINE, (1 / 0.95, 0, -25, 0, 1 / 0.95, -18),
                                    resample=Image.BICUBIC, fillcolor=(18, 20, 24, 255))

        part = MagicMock()
        part.as_image.return_value = drifted
        cand = MagicMock()
        cand.content.parts = [part]
        client = MagicMock()
        client.models.generate_content.return_value = MagicMock(candidates=[cand])
        with patch("google.genai.Client", return_value=client):
            out = AIEngine(api_key="dummy_key_123").generate_livery(tpl, [], "red car", target_resolution=1024)
        sent = client.models.generate_content.call_args.kwargs["contents"][0]
        self.assertGreater(sent.convert("L").getextrema()[1], 200)

        fixed, report = lock_to_template(out, tpl)
        ref = island_mask(tpl, 512)
        self.assertGreater(_iou(ref, island_mask(fixed, 512)), _iou(ref, island_mask(out, 512)) + 0.05)
        self.assertTrue(report.aligned)


class TestLowContrastTemplate(unittest.TestCase):
    """White mesh lines on light grey (e.g. AMS2 Caterham template) used to reach the AI as a blank sheet."""

    @staticmethod
    def _grey_template(size=1024):
        from PIL import ImageDraw
        img = Image.new("RGB", (size, size), (233, 233, 233))
        d = ImageDraw.Draw(img)
        s = size / 1024
        boxes = [(100, 20, 950, 150), (100, 870, 950, 1000), (340, 300, 700, 720), (20, 350, 300, 680)]
        for l, t, r, b in boxes:
            box = [v * s for v in (l, t, r, b)]
            d.rectangle(box, outline=(255, 255, 255), width=2)
            for x in range(int(box[0]), int(box[2]), int(40 * s)):   # mesh
                d.line([(x, box[1]), (x, box[3])], fill=(255, 255, 255), width=1)
        d.text((600 * s, 60 * s), "35", fill=(0, 0, 0))
        return img, len(boxes)

    def test_panels_detected_and_sent_as_high_contrast_map(self):
        from src.core import template_fit as tf

        tpl, n = self._grey_template()
        self.assertIsNotNone(tf.island_mask(tpl, 512))
        prep = tf.prepare_for_ai(tpl)
        self.assertEqual(prep.panel_count, n)
        arr = np.asarray(prep.image)
        self.assertLess(arr[2, 2].max(), 40)                   # dark background, as the prompt demands
        self.assertIn("TEMPLATE LAYOUT", prep.layout_text)
        self.assertIn("P4:", prep.layout_text)

    def test_prompt_contains_map_rules_and_measured_layout(self):
        from unittest.mock import patch as _patch
        from src.core.ai_engine import AIEngine

        tpl, _ = self._grey_template()
        engine = AIEngine(api_key="dummy_key_123")
        seen = {}

        def fake_run(contents, *a, **kw):
            seen["prompt"], seen["img"] = contents[-1], contents[0]
            return Image.new("RGBA", (64, 64))

        with _patch.object(engine, "_run_image_request", side_effect=fake_run):
            engine.generate_livery(tpl, [], "custom prompt", target_resolution=64)
        self.assertEqual(seen["prompt"].count("HOW TO READ IMAGE 1"), 1)
        self.assertIn("TEMPLATE LAYOUT (measured", seen["prompt"])
        self.assertLess(np.asarray(seen["img"])[2, 2].max(), 40)

    def test_warns_when_ai_ignores_layout(self):
        from src.core import template_fit as tf

        tpl, _ = self._grey_template()
        wrong = Image.new("RGBA", (1024, 1024), (18, 20, 24, 255))
        from PIL import ImageDraw
        ImageDraw.Draw(wrong).ellipse((200, 200, 800, 800), fill=(220, 60, 0, 255))
        _, report = tf.lock_to_template(wrong, tpl)
        self.assertIn("ignored the template layout", report.summary())


class TestRotatedPanels(unittest.TestCase):
    """PSD placeholder rotations -> explicit per-panel orientation + upright repaint pass."""

    @staticmethod
    def _template():
        from src.core import template_fit as tf
        img = Image.new("RGB", (1024, 1024), (233, 233, 233))
        d = ImageDraw.Draw(img)
        strips = [(100, 40, 924, 240), (100, 784, 924, 984)]
        for l, t, r, b in strips:
            d.rectangle((l, t, r, b), outline=(255, 255, 255), width=2)
            for x in range(l, r, 40):
                d.line([(x, t), (x, b)], fill=(255, 255, 255), width=1)
        # top strip stored upside-down, bottom upright (like Reiza's RSide / LSide '35' layers)
        tf.register_markers(img, [("35", (500, 100, 600, 180), 180.0), ("35", (500, 844, 600, 924), 0.0)])
        return img

    def test_orientation_is_explicit_in_layout(self):
        from src.core import template_fit as tf

        tpl = self._template()
        a = tf.analyse_template(tpl)
        self.assertEqual(len(a.panels), 2)
        rotated = a.rotated_panels()
        self.assertEqual([(i, r) for i, r, _ in rotated], [(0, 180)])
        text = tf.prepare_for_ai(tpl).layout_text
        self.assertIn("P1 is stored UPSIDE-DOWN", text)
        self.assertIn("P2 is stored UPRIGHT", text)

    def test_upright_pass_rotates_repaint_back_into_panel(self):
        from src.core.ai_engine import AIEngine

        tpl = self._template()
        base = Image.new("RGBA", (1024, 1024), (18, 20, 24, 255))
        calls = []

        def fake(contents, *a, **kw):
            calls.append(kw.get("label"))
            up = contents[1].convert("RGB").copy()          # panel shown upright
            w, h = up.size
            ImageDraw.Draw(up).rectangle((0, h * 0.6, w * 0.25, h), fill=(0, 255, 0))  # "bottom-left" mark
            return kw["postprocess"](up)

        engine = AIEngine(api_key="dummy_key_123")
        with patch.object(engine, "_run_image_request", side_effect=fake):
            out = engine.fix_rotated_panels(base, tpl, [], "black with flames")
        self.assertEqual(len(calls), 1)                       # only the rotated panel gets a pass
        arr = np.asarray(out.convert("RGB")).astype(int)
        green = (arr[..., 1] > 200) & (arr[..., 0] < 80)
        ys, xs = np.nonzero(green)
        self.assertTrue(len(xs) > 0)
        # upright bottom-left -> after turning back 180 deg it must sit top-right of the top strip
        self.assertGreater(xs.mean(), 600)
        self.assertLess(ys.mean(), 140)
        self.assertFalse(green[700:].any())                  # the upright bottom strip is untouched
        self.assertIn("P1 repainted upright", engine.last_notes[0])
