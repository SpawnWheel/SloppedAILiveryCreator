"""
Unit and Integration tests for Livery Studio Refinement, Annotation Canvas,
Existing Livery Loading, and Custom Preview Uploads.
"""

import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.core.ai_engine import AIEngine
from src.core.config import config
from src.core.dds_converter import DDSConverter
from src.core.exporter import LiveryExporter
from src.core.image_processor import ImageProcessor
from src.core.installer import LiveryInstaller
from src.core.xml_manager import XMLManager
from src.ui.preview_panel import PreviewPanel
from src.ui.widgets.annotation_canvas import AnnotationCanvas, ToolType
from src.ui.widgets.preview_thumbnail_box import PreviewThumbnailBox


from unittest.mock import patch

class TestRefinementCore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.patcher = patch.object(LiveryInstaller, "get_overrides_path", return_value=self.temp_dir)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dds_loader(self):
        """Tests that DDSConverter.load_dds_image can decompress BC3 and BC1 textures."""
        sample_img = Image.new("RGBA", (256, 256), (120, 200, 80, 255))
        dds_file = DDSConverter.convert_image_to_dds(
            image=sample_img,
            output_dir=self.temp_dir,
            base_filename="test_body",
            format_type="BC3_UNORM",
            generate_mipmaps=True,
        )
        self.assertTrue(dds_file.exists())

        loaded = DDSConverter.load_dds_image(dds_file)
        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.size, (256, 256))
        self.assertEqual(loaded.mode, "RGBA")

    def test_mock_refine(self):
        """Tests that AIEngine.refine_mock_livery produces 4K modified textures."""
        base_img = Image.new("RGBA", (512, 512), (20, 30, 40, 255))
        notes = "Make the stripes neon green"
        refined = AIEngine.refine_mock_livery(
            current_livery=base_img,
            notes=notes,
            iteration=2,
            target_resolution=4096,
        )
        self.assertEqual(refined.size, (4096, 4096))
        self.assertEqual(refined.mode, "RGBA")

    def test_refine_livery_requests_4k(self):
        """Tests that AIEngine.refine_livery configures native 4K output on Gemini API."""
        from unittest.mock import MagicMock
        engine = AIEngine(api_key="test_dummy_key_12345")

        mock_candidate = MagicMock()
        mock_part = MagicMock()
        mock_img_4k = Image.new("RGBA", (4096, 4096), (10, 20, 30, 255))
        mock_part.as_image.return_value = mock_img_4k
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            curr_img = Image.new("RGBA", (4096, 4096), (255, 0, 0, 255))
            result = engine.refine_livery(
                current_livery=curr_img,
                notes="Add black stripes",
                target_resolution=4096,
            )
            self.assertEqual(result.size, (4096, 4096))

            # Verify generate_content was called and image_size="4K" was specified
            mock_client.models.generate_content.assert_called()
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            gen_cfg = call_kwargs.get("config")
            self.assertIsNotNone(gen_cfg)
            self.assertEqual(gen_cfg.image_config.image_size, "4K")
            self.assertEqual(gen_cfg.image_config.aspect_ratio, "1:1")

    def test_install_and_load_existing_livery(self):
        """Tests full lifecycle: install with custom preview, then load back into editor."""
        car_folder = "porsche_992_gt3_r"
        livery_id = 99
        team = "Apex Racing"
        driver = "A. Senna"
        base_texture = Image.new("RGBA", (256, 256), (255, 50, 50, 255))
        custom_preview = Image.new("RGB", (300, 100), (0, 120, 255))

        target_dir, backup = LiveryInstaller.install_livery(
            car_folder_name=car_folder,
            livery_id=livery_id,
            team_name=team,
            driver=driver,
            texture_image=base_texture,
            finish_type="Gloss",
            custom_preview_image=custom_preview,
        )

        self.assertTrue(target_dir.exists())
        self.assertTrue((target_dir / "body.dds").exists())
        self.assertTrue((target_dir / "preview.dds").exists())

        # Now test loading this installed livery back
        loaded_tex, loaded_prev, meta = LiveryInstaller.load_installed_livery(car_folder, livery_id)
        self.assertIsNotNone(loaded_tex)
        self.assertIsNotNone(loaded_prev)
        self.assertEqual(meta["livery_id"], "99")
        self.assertIn("Apex Racing", meta["name"])

    def test_export_with_custom_preview(self):
        """Tests that LiveryExporter.export_to_zip incorporates custom preview."""
        zip_path = self.temp_dir / "shareable.zip"
        base_texture = Image.new("RGBA", (256, 256), (50, 50, 150, 255))
        custom_prev = Image.new("RGB", (400, 200), (255, 200, 0))

        out = LiveryExporter.export_to_zip(
            car_folder_name="bmw_m4_gt3",
            livery_id=77,
            team_name="Bavaria Sim",
            driver="M. Stuck",
            texture_image=base_texture,
            output_zip_path=zip_path,
            custom_preview_image=custom_prev,
        )
        self.assertTrue(out.exists())
        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
            self.assertTrue(any("preview.dds" in n for n in names))
            self.assertTrue(any("body.dds" in n for n in names))
            self.assertTrue(any("bmw_m4_gt3.xml" in n for n in names))


class TestUIWidgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_annotation_canvas_tools(self):
        """Tests drawing tools and composite image export on AnnotationCanvas."""
        canvas = AnnotationCanvas()
        base_img = Image.new("RGBA", (1024, 1024), (30, 40, 50, 255))
        canvas.set_texture(base_img)

        self.assertFalse(canvas.has_annotations())

        # Simulate drawing an arrow and pen stroke on overlay
        canvas.set_tool(ToolType.PEN)
        canvas.set_color(QColor(255, 0, 0))
        canvas.set_brush_size(8)
        canvas._draw_pen_stroke(QPoint(100, 100), QPoint(200, 200))
        canvas._draw_arrow(QPoint(300, 300), QPoint(400, 400))

        self.assertTrue(canvas.has_annotations())

        # Test composite image generation
        composite = canvas.get_annotated_image()
        self.assertIsNotNone(composite)
        self.assertEqual(composite.size, (1024, 1024))

        # Test clear
        canvas.clear_annotations()
        self.assertFalse(canvas.has_annotations())

    def test_preview_thumbnail_box(self):
        """Tests PreviewThumbnailBox auto-generation vs custom preview upload."""
        box = PreviewThumbnailBox()
        texture = Image.new("RGBA", (512, 512), (100, 150, 200, 255))
        box.set_auto_texture(texture)

        eff_auto = box.get_effective_preview_thumbnail()
        self.assertIsNotNone(eff_auto)
        self.assertEqual(eff_auto.size, (252, 64))
        self.assertIsNone(box.get_custom_preview())

        # Set custom preview
        custom_shot = Image.new("RGB", (800, 400), (255, 0, 128))
        box.set_custom_preview(custom_shot)
        self.assertIsNotNone(box.get_custom_preview())
        eff_custom = box.get_effective_preview_thumbnail()
        self.assertEqual(eff_custom.size, (252, 64))

        # Reset
        box.clear_custom_preview()
        self.assertIsNone(box.get_custom_preview())

    def test_preview_panel_revision_flow(self):
        """Tests multi-turn revision tracking, stepping, and commit."""
        panel = PreviewPanel()
        img_v1 = Image.new("RGBA", (512, 512), (200, 50, 50, 255))
        panel.set_generated_image(img_v1, is_refinement=False)

        self.assertEqual(len(panel.revisions), 1)
        self.assertEqual(panel.current_rev_idx, 0)
        self.assertFalse(panel.revisions[0].committed)

        # Refine to v2
        img_v2 = Image.new("RGBA", (512, 512), (50, 200, 50, 255))
        panel.set_generated_image(img_v2, is_refinement=True, notes="Change to green")

        self.assertEqual(len(panel.revisions), 2)
        self.assertEqual(panel.current_rev_idx, 1)

        # Step back to v1
        panel._prev_revision()
        self.assertEqual(panel.current_rev_idx, 0)
        self.assertEqual(panel.get_generated_image().getpixel((10, 10)), (200, 50, 50, 255))

        # Step forward to v2 and commit
        panel._next_revision()
        self.assertEqual(panel.current_rev_idx, 1)
        panel._on_commit_clicked()
        self.assertTrue(panel.revisions[1].committed)


if __name__ == "__main__":
    unittest.main()
