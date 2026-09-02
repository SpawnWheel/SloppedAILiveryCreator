"""
Comprehensive Unit & Integration Test Suite for AMS2 AI Livery Studio.
"""

import io
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

from src.core.ai_engine import AIEngine
from src.core.dds_converter import DDSConverter
from src.core.exporter import LiveryExporter
from src.core.image_processor import ImageProcessor
from src.core.installer import LiveryInstaller
from src.core.template_loader import TemplateLoader
from src.core.xml_manager import XMLManager
from src.utils.constants import FINISH_ALPHA_MAP


class TestXMLManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.xml_path = Path(self.temp_dir) / "bmw_m4_gt3.xml"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upsert_and_parse(self):
        # 1. First insert
        XMLManager.upsert_livery(
            xml_path=self.xml_path,
            livery_id=51,
            team_name="Gulf Sim Racing",
            rel_body_path="gulf_51\\body.dds",
            rel_preview_path="gulf_51\\preview.dds",
            base_livery="Default",
        )

        entries = XMLManager.parse_overrides(self.xml_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["livery_id"], "51")
        self.assertEqual(entries[0]["name"], "Gulf Sim Racing")
        self.assertEqual(entries[0]["texture_path"], "gulf_51\\body.dds")
        self.assertEqual(entries[0]["preview_path"], "gulf_51\\preview.dds")

        # 2. Collision detection
        col = XMLManager.check_collision(self.xml_path, 51)
        self.assertIsNotNone(col)
        self.assertEqual(col["name"], "Gulf Sim Racing")

        no_col = XMLManager.check_collision(self.xml_path, 99)
        self.assertIsNone(no_col)

        # 3. Surgical update (same ID)
        XMLManager.upsert_livery(
            xml_path=self.xml_path,
            livery_id=51,
            team_name="Gulf Sim Racing Updated",
            rel_body_path="gulf_51_v2\\body.dds",
            rel_preview_path="gulf_51_v2\\preview.dds",
            base_livery="Default",
        )
        entries2 = XMLManager.parse_overrides(self.xml_path)
        self.assertEqual(len(entries2), 1)
        self.assertEqual(entries2[0]["name"], "Gulf Sim Racing Updated")

        # 4. Append another livery (different ID)
        XMLManager.upsert_livery(
            xml_path=self.xml_path,
            livery_id=42,
            team_name="Team Porsche",
            rel_body_path="porsche_42\\body.dds",
            rel_preview_path="porsche_42\\preview.dds",
            base_livery="Default",
        )
        entries3 = XMLManager.parse_overrides(self.xml_path)
        self.assertEqual(len(entries3), 2)


class TestImageProcessor(unittest.TestCase):
    def test_alpha_finish_injection(self):
        base_img = Image.new("RGB", (100, 100), (255, 0, 0))

        for finish, expected_alpha in FINISH_ALPHA_MAP.items():
            processed = ImageProcessor.apply_finish(base_img, finish)
            self.assertEqual(processed.mode, "RGBA")
            # Check alpha channel value
            _, _, _, a = processed.split()
            self.assertEqual(a.getpixel((50, 50)), expected_alpha)

    def test_resize_and_thumbnail(self):
        base_img = Image.new("RGBA", (800, 600), (0, 255, 0, 255))
        resized = ImageProcessor.resize_texture(base_img, 4096)
        self.assertEqual(resized.size, (4096, 4096))

        thumb = ImageProcessor.generate_preview_thumbnail(base_img, 252, 64)
        self.assertEqual(thumb.size, (252, 64))


class TestDDSConverter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dds_conversion(self):
        img = Image.new("RGBA", (512, 512), (100, 150, 200, 80))
        dds_file = DDSConverter.convert_image_to_dds(
            image=img,
            output_dir=self.temp_dir,
            base_filename="body_test",
            format_type="BC3_UNORM",
            generate_mipmaps=True,
        )
        self.assertTrue(dds_file.exists())
        self.assertGreater(dds_file.stat().st_size, 1000)

        # Check preview conversion (BC1)
        prev_img = Image.new("RGB", (252, 64), (200, 50, 50))
        prev_dds = DDSConverter.convert_image_to_dds(
            image=prev_img,
            output_dir=self.temp_dir,
            base_filename="preview_test",
            format_type="BC1_UNORM",
            generate_mipmaps=False,
        )
        self.assertTrue(prev_dds.exists())


class TestTemplateLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_png(self):
        img_path = self.temp_dir / "wireframe.png"
        sample = Image.new("RGB", (2048, 2048), (255, 255, 255))
        sample.save(img_path)

        loaded, name = TemplateLoader.load_from_path(img_path)
        self.assertEqual(loaded.size, (2048, 2048))
        self.assertEqual(name, "wireframe.png")

    def test_load_zip_archive(self):
        # Create a ZIP containing a small icon and a large wireframe
        zip_path = self.temp_dir / "car_template.zip"
        small_img = Image.new("RGB", (128, 128), (0, 0, 0))
        large_img = Image.new("RGB", (4096, 4096), (255, 0, 0))

        buf_small = io.BytesIO()
        small_img.save(buf_small, format="PNG")

        buf_large = io.BytesIO()
        large_img.save(buf_large, format="PNG")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("preview_thumb.png", buf_small.getvalue())
            zf.writestr("UV_BODY_WIREFRAME_4K.png", buf_large.getvalue())

        loaded, name = TemplateLoader.load_from_path(zip_path)
        # Should pick the highest resolution image
        self.assertEqual(loaded.size, (4096, 4096))
        self.assertIn("UV_BODY_WIREFRAME_4K.png", name)


class TestExporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_zip_export(self):
        sample_img = Image.new("RGBA", (512, 512), (50, 100, 200, 255))
        zip_output = self.temp_dir / "test_export.zip"

        LiveryExporter.export_to_zip(
            car_folder_name="bmw_m4_gt3",
            livery_id=51,
            team_name="Sim Gulf",
            driver="D. Born",
            texture_image=sample_img,
            output_zip_path=zip_output,
            finish_type="Matte",
        )

        self.assertTrue(zip_output.exists())
        with zipfile.ZipFile(zip_output, "r") as zf:
            namelist = zf.namelist()
            self.assertTrue(any("bmw_m4_gt3.xml" in n for n in namelist))
            self.assertTrue(any("body.dds" in n for n in namelist))
            self.assertTrue(any("preview.dds" in n for n in namelist))
            self.assertTrue(any("README.txt" in n for n in namelist))


if __name__ == "__main__":
    unittest.main()
