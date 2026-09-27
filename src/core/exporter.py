"""
Standalone ZIP Exporter for sharing custom liveries on modding hubs (e.g., Overtake.gg).
"""

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

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
        finish_type: str = "Gloss",
        base_livery: str = "Default",
        custom_preview_image: Optional[Image.Image] = None,
        extra_textures: Optional[Dict[str, Image.Image]] = None,
    ) -> Path:
        """
        Creates a distributable ZIP: body/preview (+ wheel) DDS files, the car XML override and a README.
        The archive mirrors the game folder structure so it can be extracted straight into AMS2.
        """
        from src.core.installer import LiveryInstaller

        team_slug = slugify(f"{team_name}_{driver}_{livery_id}")
        output_zip_path = Path(output_zip_path)
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="livery_export_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            skin_dir = tmp_path / team_slug

            built = LiveryInstaller.build_livery_assets(
                skin_dir=skin_dir,
                texture_image=texture_image,
                finish_type=finish_type,
                custom_preview_image=custom_preview_image,
                extra_textures=extra_textures,
            )

            xml_file = tmp_path / f"{car_folder_name}.xml"
            display_title = f"{team_name} #{livery_id}" if str(livery_id) not in team_name else team_name
            XMLManager.upsert_livery(
                xml_path=xml_file,
                livery_id=livery_id,
                team_name=display_title,
                rel_body_path=f"{team_slug}\\body.dds",
                rel_preview_path=f"{team_slug}\\preview.dds",
                base_livery=base_livery,
                extra_textures={
                    n: f"{team_slug}\\{p.name}" for n, p in built.items() if n not in ("BODY", "PREVIEW")
                },
            )

            # Snippet for people who already have their own <car>.xml
            block_start = xml_file.read_text(encoding="utf-8").find("<LIVERY_OVERRIDE")
            snippet = xml_file.read_text(encoding="utf-8")[block_start:].split("</USER_OVERRIDES>")[0].rstrip()

            wheels_line = ""
            if any(n.startswith("WHEEL") for n in built):
                wheels_line = "Includes custom wheel rim textures.\n"

            readme_text = (
                f"Automobilista 2 Custom Livery: {team_name} #{livery_id}\n"
                f"Car Model: {car_folder_name}\n"
                f"Driver: {driver}\n"
                f"Finish: {finish_type}   Base livery: {base_livery}\n"
                f"{wheels_line}"
                f"Generated with AMS2 AI Livery Studio\n"
                f"{'=' * 50}\n\n"
                f"INSTALLATION:\n"
                f"1. Extract this archive into your Automobilista 2 install folder\n"
                f"   (the folder that contains 'Vehicles').\n"
                f"   Files land in Vehicles/Textures/CustomLiveries/Overrides/{car_folder_name}/\n\n"
                f"2. If you ALREADY have a {car_folder_name}.xml with your own liveries, do NOT overwrite it.\n"
                f"   Instead paste this block inside its <USER_OVERRIDES> section:\n\n"
                f"{snippet}\n\n"
                f"3. Launch AMS2 (add -showLiveryIDs to launch options to see slot numbers).\n"
            )
            readme_file = tmp_path / "README.txt"
            readme_file.write_text(readme_text, encoding="utf-8")

            with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                base_arc = f"Vehicles/Textures/CustomLiveries/Overrides/{car_folder_name}"
                zf.write(xml_file, f"{base_arc}/{car_folder_name}.xml")
                for path in built.values():
                    zf.write(path, f"{base_arc}/{team_slug}/{path.name}")
                zf.write(readme_file, "README.txt")

        return output_zip_path
