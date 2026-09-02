"""
DDS texture converter using Microsoft DirectXTex texconv.exe.
Supports automatic downloading of texconv.exe if not present.
"""

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Callable
from PIL import Image

from src.utils.constants import TEXCONV_DOWNLOAD_URL
from src.utils.file_helpers import get_tools_dir


class DDSConverter:
    """Manages DDS texture conversion via texconv.exe."""

    @staticmethod
    def get_texconv_path() -> Path:
        """Returns the path to texconv.exe, downloading it if needed."""
        tools_dir = get_tools_dir()
        local_texconv = tools_dir / "texconv.exe"

        if local_texconv.exists() and local_texconv.stat().st_size > 100000:
            return local_texconv

        # Check if texconv is available on system PATH
        system_texconv = shutil.which("texconv.exe") or shutil.which("texconv")
        if system_texconv:
            return Path(system_texconv)

        # Automatically download texconv.exe
        DDSConverter.download_texconv()
        if local_texconv.exists():
            return local_texconv

        raise FileNotFoundError(
            f"texconv.exe could not be found or downloaded to {local_texconv}"
        )

    @staticmethod
    def download_texconv(progress_callback: Optional[Callable[[int, int], None]] = None) -> Path:
        """Downloads the official standalone texconv.exe from DirectXTex GitHub releases."""
        tools_dir = get_tools_dir()
        target_path = tools_dir / "texconv.exe"
        temp_target = tools_dir / "texconv.exe.tmp"

        print(f"[DDSConverter] Downloading texconv.exe from {TEXCONV_DOWNLOAD_URL}...")

        def _reporthook(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                progress_callback(min(downloaded, total_size), total_size)

        headers = {"User-Agent": "AMS2-AI-Livery-Studio/1.0"}
        req = urllib.request.Request(TEXCONV_DOWNLOAD_URL, headers=headers)

        with urllib.request.urlopen(req) as response, open(temp_target, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        if temp_target.exists() and temp_target.stat().st_size > 100000:
            if target_path.exists():
                target_path.unlink()
            temp_target.rename(target_path)
            print(f"[DDSConverter] texconv.exe installed successfully at {target_path}")
            return target_path
        else:
            if temp_target.exists():
                temp_target.unlink()
            raise RuntimeError("Downloaded texconv.exe was invalid or incomplete.")

    @staticmethod
    def convert_image_to_dds(
        image: Image.Image,
        output_dir: Path,
        base_filename: str = "body",
        format_type: str = "BC3_UNORM",
        generate_mipmaps: bool = True,
    ) -> Path:
        """
        Converts a PIL Image to a .dds file using texconv.exe.
        - format_type: "BC3_UNORM" for DXT5 with full alpha (body livery)
                       "BC1_UNORM" for DXT1 (preview thumbnail)
        - generate_mipmaps: True creates a full mipmap chain (-m 0).
        """
        texconv = DDSConverter.get_texconv_path()
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="dds_conv_") as tmp_dir:
            temp_png = Path(tmp_dir) / f"{base_filename}.png"
            image.save(temp_png, format="PNG")

            cmd = [
                str(texconv),
                "-f",
                format_type,
                "-y",  # Overwrite
                "-srgb",  # sRGB color space
                "-o",
                str(output_dir),
            ]

            if generate_mipmaps:
                cmd.extend(["-m", "0"])  # Full mipmap chain
            else:
                cmd.extend(["-m", "1"])  # 1 level (no extra mipmaps)

            cmd.append(str(temp_png))

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if process.returncode != 0:
                raise RuntimeError(
                    f"texconv failed with exit code {process.returncode}:\n{process.stderr}\n{process.stdout}"
                )

            expected_dds = output_dir / f"{base_filename}.dds"
            # Note: texconv may uppercase the extension (.DDS)
            if not expected_dds.exists():
                upper_dds = output_dir / f"{base_filename}.DDS"
                if upper_dds.exists():
                    upper_dds.rename(expected_dds)

            if not expected_dds.exists():
                raise FileNotFoundError(
                    f"Expected converted DDS file was not found at {expected_dds}"
                )

            return expected_dds
