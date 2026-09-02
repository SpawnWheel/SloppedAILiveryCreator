"""
Image processing utilities: Alpha channel injection for AMS2 reflectivity shaders,
resolution scaling, and AMS2 UI preview thumbnail generation.
"""

from typing import Tuple
from PIL import Image, ImageOps

from src.utils.constants import (
    DEFAULT_FINISH,
    DEFAULT_RESOLUTION,
    FINISH_ALPHA_MAP,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)


class ImageProcessor:
    """Handles livery alpha masking, resizing, and thumbnail generation."""

    @staticmethod
    def apply_finish(image: Image.Image, finish_type: str = DEFAULT_FINISH) -> Image.Image:
        """
        Injects the Alpha channel value into the image based on the selected finish.
        In AMS2, the Alpha channel dictates surface reflectivity / glossiness:
        - 255: Full Gloss / Highly reflective
        - 210: Semi-Gloss
        - 170: Satin
        - 80:  Matte (diffuse reflection only)
        - 200: Metallic
        """
        alpha_val = FINISH_ALPHA_MAP.get(finish_type, FINISH_ALPHA_MAP[DEFAULT_FINISH])

        # Ensure image is in RGBA mode
        if image.mode != "RGBA":
            rgba_img = image.convert("RGBA")
        else:
            rgba_img = image.copy()

        # Split channels
        r, g, b, _ = rgba_img.split()

        # Create new alpha channel with constant alpha_val
        alpha_channel = Image.new("L", rgba_img.size, color=alpha_val)

        # Merge back into RGBA
        result = Image.merge("RGBA", (r, g, b, alpha_channel))
        return result

    @staticmethod
    def resize_texture(
        image: Image.Image, target_dim: int = DEFAULT_RESOLUTION
    ) -> Image.Image:
        """
        Resizes the texture to the target square resolution (default 4096x4096px).
        Uses high-quality Lanczos resampling.
        """
        if image.size == (target_dim, target_dim):
            return image
        return image.resize((target_dim, target_dim), resample=Image.Resampling.LANCZOS)

    @staticmethod
    def generate_preview_thumbnail(
        image: Image.Image, width: int = PREVIEW_WIDTH, height: int = PREVIEW_HEIGHT
    ) -> Image.Image:
        """
        Generates an AMS2 UI preview thumbnail (252x64 px).
        Uses intelligent center cropping and lanczos downscaling.
        """
        rgb_img = image.convert("RGB")

        # Fit and crop to exact 252x64 aspect ratio
        thumb = ImageOps.fit(
            rgb_img, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )
        return thumb
