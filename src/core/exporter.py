"""
Standalone ZIP Exporter for sharing custom liveries on modding hubs (e.g., Overtake.gg).
"""

import io
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from PIL import Image

from src.core.dds_converter import DDSConverter
from src.core.image_processor import ImageProcessor
from src.core.xml_manager import XMLManager
from src.utils.file_helpers import slugify


class LiveryExporter:
    """Packages livery assets into a clean, shareable ZIP archive."""

    @staticmethod
    def export_to_zip(
        car_folder_name: str,
        livery_id: int,
        team_name: str,
        driver: str,
        texture_image: Image.Image,
        output_zip_path: Path,
        finish_type: str = "Matte",
        base_livery: str = "Default",
    ) -> Path:
        """
        Creates a distributable ZIP containing the body DDS, preview DDS, and override XML.
        """
        team_slug = slugify(f"{team_name}_{driver}_{livery_id}")
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="livery_export_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Process 4K Livery Texture & Inject Alpha
            processed_texture = ImageProcessor.apply_finish(texture_image, finish_type)
            processed_texture = ImageProcessor.resize_texture(processed_texture, 4096)

            # 2. Process Preview Thumbnail
            preview_thumb = ImageProcessor.generate_preview_thumbnail(texture_image)

            # 3. Convert Livery to DDS (BC3 DXT5 with mipmaps)
            skin_dir = tmp_path / team_slug
            skin_dir.mkdir(parents=True, exist_ok=True)

            body_dds = DDSConverter.convert_image_to_dds(
                image=processed_texture,
                output_dir=skin_dir,
                base_filename="body",
                format_type="BC3_UNORM",
                generate_mipmaps=True,
            )

            # 4. Convert Preview to DDS (BC1 DXT1)
            preview_dds = DDSConverter.convert_image_to_dds(
                image=preview_thumb,
                output_dir=skin_dir,
                base_filename="preview",
                format_type="BC1_UNORM",
                generate_mipmaps=False,
            )

            # 5. Generate Standalone Car XML
            xml_file = tmp_path / f"{car_folder_name}.xml"
            rel_body = f"{team_slug}\\body.dds"
            rel_preview = f"{team_slug}\\preview.dds"
            display_title = f"{team_name} #{livery_id}" if str(livery_id) not in team_name else team_name

            XMLManager.upsert_livery(
                xml_path=xml_file,
                livery_id=livery_id,
                team_name=display_title,
                rel_body_path=rel_body,
                rel_preview_path=rel_preview,
                base_livery=base_livery,
            )

            # 6. Generate Readme
            readme_file = tmp_path / "README.txt"
            readme_text = (
                f"Automobilista 2 Custom Livery: {team_name} #{livery_id}\n"
                f"Car Model: {car_folder_name}\n"
                f"Driver: {driver}\n"
                f"Finish: {finish_type}\n"
                f"Generated with AMS2 AI Livery Studio\n"
                f"{'='*50}\n\n"
                f"INSTALLATION INSTRUCTIONS:\n"
                f"1. Extract this archive into your Automobilista 2 Overrides directory:\n"
                f"   Documents/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides/\n\n"
                f"2. Ensure the '{car_folder_name}' folder contains '{car_folder_name}.xml' and '{team_slug}' subfolder.\n"
                f"3. Launch Automobilista 2 and select '{team_name}' in the vehicle showroom!\n"
            )
            readme_file.write_text(readme_text, encoding="utf-8")

            # 7. Build the ZIP archive
            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Structure: Vehicles/Textures/CustomLiveries/Overrides/<car_folder>/...
                base_arc = f"Vehicles/Textures/CustomLiveries/Overrides/{car_folder_name}"
                zf.write(xml_file, f"{base_arc}/{car_folder_name}.xml")
                zf.write(body_dds, f"{base_arc}/{team_slug}/body.dds")
                zf.write(preview_dds, f"{base_arc}/{team_slug}/preview.dds")
                zf.write(readme_file, "README.txt")

        return output_zip_path
