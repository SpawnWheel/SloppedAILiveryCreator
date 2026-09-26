"""
AMS2 Livery Installer with auto-discovery, timestamped backups, and one-click rollback.
"""

import datetime
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

from src.core.config import config
from src.core.dds_converter import DDSConverter
from src.core.image_processor import ImageProcessor
from src.core.xml_manager import XMLManager
from src.utils.constants import get_default_ams2_path
from src.utils.file_helpers import slugify

# Built-in car folder suggestions when Overrides folder is empty / not yet created
POPULAR_AMS2_CARS = [
    "bmw_m4_gt3",
    "porsche_992_gt3_r",
    "mercedes_amg_gt3_evo",
    "mclaren_720s_gt3_evo",
    "audi_r8_lms_gt3",
    "aston_martin_vantage_gt3",
    "corvette_c8r_gte",
    "formula_ultimate_gen2",
    "formula_v10_g1",
    "formula_v12",
    "stockcar_2024_corolla",
    "stockcar_2024_cruze",
    "porsche_gt3_cup",
    "kart_125cc",
]


class LiveryInstaller:
    """Handles discovery, backup, installation, and rollback of AMS2 liveries."""

    @staticmethod
    def get_overrides_path() -> Path:
        """Returns the active AMS2 Overrides path from config or default."""
        return config.ams2_overrides_path

    @staticmethod
    def is_ams2_installed() -> bool:
        """Checks if the AMS2 overrides folder exists."""
        return LiveryInstaller.get_overrides_path().exists()

    @staticmethod
    def list_car_folders() -> List[str]:
        """
        Discovers all available car model folders in the AMS2 Overrides directory.
        Falls back to a curated list if the directory is empty or not created yet.
        """
        overrides = LiveryInstaller.get_overrides_path()
        found_cars = set()

        if overrides.exists():
            for entry in overrides.iterdir():
                if entry.is_dir() and not entry.name.startswith("."):
                    found_cars.add(entry.name.lower())
                elif entry.is_file() and entry.suffix.lower() == ".xml":
                    stem = entry.stem.lower().replace("_dist", "")
                    if stem and stem != "customliveries":
                        found_cars.add(stem)

        # Merge with curated car list and sort
        all_cars = sorted(list(found_cars | set(POPULAR_AMS2_CARS)))
        return all_cars

    @staticmethod
    def get_car_xml_path(car_folder_name: str) -> Path:
        """
        Returns the primary XML path for a car model.
        In AMS2, it is located at: Overrides/[car_folder_name]/[car_folder_name].xml
        """
        car_dir = LiveryInstaller.get_overrides_path() / car_folder_name
        return car_dir / f"{car_folder_name}.xml"

    @staticmethod
    def create_backup(xml_path: Path) -> Optional[Path]:
        """
        Creates a timestamped backup before modifying an XML file:
        e.g. bmw_m4_gt3_backup_20260901_193000.xml.bak
        """
        if not xml_path.exists() or xml_path.stat().st_size == 0:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{xml_path.stem}_backup_{timestamp}.xml.bak"
        backup_path = xml_path.parent / backup_name

        shutil.copy2(xml_path, backup_path)
        print(f"[LiveryInstaller] Created backup at {backup_path}")
        return backup_path

    @staticmethod
    def list_backups(car_folder_name: str) -> List[Tuple[Path, str]]:
        """
        Lists all available .bak files for a car folder, sorted newest first.
        Returns a list of tuples: (backup_file_path, formatted_timestamp_string).
        """
        car_dir = LiveryInstaller.get_overrides_path() / car_folder_name
        if not car_dir.exists():
            return []

        backups = []
        for file in car_dir.glob("*.xml.bak"):
            mtime = file.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            backups.append((file, f"{file.name} ({formatted})"))

        # Sort newest first
        backups.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
        return backups

    @staticmethod
    def restore_backup(backup_path: Path, car_folder_name: str) -> Path:
        """Restores a selected backup file over the active car XML."""
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        target_xml = LiveryInstaller.get_car_xml_path(car_folder_name)
        target_xml.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(backup_path, target_xml)
        print(f"[LiveryInstaller] Restored {backup_path} -> {target_xml}")
        return target_xml

    @staticmethod
    def list_installed_liveries(car_folder_name: str) -> List[dict]:
        """Returns all installed livery overrides parsed from the car XML."""
        xml_path = LiveryInstaller.get_car_xml_path(car_folder_name)
        return XMLManager.parse_overrides(xml_path)

    @staticmethod
    def load_installed_livery(
        car_folder_name: str,
        livery_id: int,
    ) -> Tuple[Image.Image, Optional[Image.Image], dict]:
        """
        Loads an installed livery from the AMS2 Overrides folder.
        Returns: (texture_image, preview_image_or_none, metadata_dict)
        """
        xml_path = LiveryInstaller.get_car_xml_path(car_folder_name)
        if not xml_path.exists():
            raise FileNotFoundError(f"Configuration XML not found for '{car_folder_name}' at {xml_path}")

        meta = XMLManager.check_collision(xml_path, livery_id)
        if not meta:
            raise ValueError(f"Livery slot #{livery_id} not found in {xml_path.name}")

        car_dir = LiveryInstaller.get_overrides_path() / car_folder_name
        rel_texture = meta.get("texture_path", "").replace("/", "\\")
        texture_file = car_dir / rel_texture

        if not texture_file.exists():
            raise FileNotFoundError(f"Livery texture file '{texture_file}' does not exist.")

        texture_image = DDSConverter.load_dds_image(texture_file)

        preview_image = None
        rel_preview = meta.get("preview_path", "").replace("/", "\\")
        if rel_preview:
            preview_file = car_dir / rel_preview
            if preview_file.exists():
                try:
                    preview_image = DDSConverter.load_dds_image(preview_file)
                except Exception as e:
                    print(f"[LiveryInstaller] Could not load preview DDS: {e}")

        return texture_image, preview_image, meta

    @staticmethod
    def install_livery(
        car_folder_name: str,
        livery_id: int,
        team_name: str,
        driver: str,
        texture_image: Image.Image,
        finish_type: str = "Matte",
        base_livery: str = "Default",
        custom_preview_image: Optional[Image.Image] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Performs full installation:
        1. Injects alpha reflectivity channel according to finish_type.
        2. Resizes texture to 4096x4096px.
        3. Generates 252x64 UI preview thumbnail (from custom_preview_image or texture).
        4. Compresses texture to body.dds (BC3/DXT5 + Mipmaps).
        5. Compresses thumbnail to preview.dds (BC1/DXT1).
        6. Creates timestamped backup of the XML.
        7. Surgically updates the car XML override.

        Returns: (installed_dir_path, backup_path_or_none)
        """
        car_dir = LiveryInstaller.get_overrides_path() / car_folder_name
        car_dir.mkdir(parents=True, exist_ok=True)

        team_slug = slugify(f"{team_name}_{driver}_{livery_id}")
        target_skin_dir = car_dir / team_slug
        target_skin_dir.mkdir(parents=True, exist_ok=True)

        # 1. Process 4K Livery Texture & Inject Alpha
        if finish_type == "__preserve__":
            processed_texture = texture_image.convert("RGBA") if texture_image.mode != "RGBA" else texture_image
        else:
            processed_texture = ImageProcessor.apply_finish(texture_image, finish_type)
        processed_texture = ImageProcessor.resize_texture(processed_texture, 4096)

        # 2. Process Preview Thumbnail (custom or flatpack fallback)
        if custom_preview_image is not None:
            preview_thumb = ImageProcessor.generate_preview_thumbnail(custom_preview_image)
        else:
            preview_thumb = ImageProcessor.generate_preview_thumbnail(texture_image)

        # 3. Convert Livery to DDS (BC3 DXT5 with full mipmaps)
        body_dds_path = DDSConverter.convert_image_to_dds(
            image=processed_texture,
            output_dir=target_skin_dir,
            base_filename="body",
            format_type="BC3_UNORM",
            generate_mipmaps=True,
        )

        # 4. Convert Preview to DDS (BC1 DXT1)
        preview_dds_path = DDSConverter.convert_image_to_dds(
            image=preview_thumb,
            output_dir=target_skin_dir,
            base_filename="preview",
            format_type="BC1_UNORM",
            generate_mipmaps=False,
        )

        # 5. XML Management & Backup
        xml_path = LiveryInstaller.get_car_xml_path(car_folder_name)
        backup_path = LiveryInstaller.create_backup(xml_path)

        rel_body = f"{team_slug}\\body.dds"
        rel_preview = f"{team_slug}\\preview.dds"
        display_title = f"{team_name} #{livery_id}" if str(livery_id) not in team_name else team_name

        XMLManager.upsert_livery(
            xml_path=xml_path,
            livery_id=livery_id,
            team_name=display_title,
            rel_body_path=rel_body,
            rel_preview_path=rel_preview,
            base_livery=base_livery,
        )

        return target_skin_dir, backup_path

    @staticmethod
    def install_livery_from_file(
        file_path: Path,
        car_folder_name: str,
        livery_id: int,
        team_name: str,
        driver: str,
        finish_type: str = "Matte",
        base_livery: str = "Default",
        custom_preview_image: Optional[Image.Image] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """
        Installs a pre-made livery file (DDS, PNG, JPG, TGA, BMP, etc.) directly without AI:
        - If DDS: loads via DDSConverter.load_dds_image and preserves the original alpha channel
        - If other image format: loads via PIL and applies selected finish_type
        - Generates preview thumbnail (custom or auto-generated)
        - Converts to DDS and surgically updates the car XML override

        Returns: (installed_dir_path, backup_path_or_none)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Livery file not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext == ".dds":
            texture_img = DDSConverter.load_dds_image(file_path)
            effective_finish = "__preserve__"
        else:
            with Image.open(file_path) as img:
                texture_img = img.convert("RGBA").copy()
            effective_finish = finish_type

        return LiveryInstaller.install_livery(
            car_folder_name=car_folder_name,
            livery_id=livery_id,
            team_name=team_name,
            driver=driver,
            texture_image=texture_img,
            finish_type=effective_finish,
            base_livery=base_livery,
            custom_preview_image=custom_preview_image,
        )

