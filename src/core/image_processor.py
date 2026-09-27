"""
Image processing utilities: Alpha channel injection for AMS2 reflectivity shaders,
resolution scaling, and AMS2 UI preview thumbnail generation.
"""

from typing import List, Tuple
from PIL import Image, ImageChops, ImageOps

from src.utils.constants import (
    DEFAULT_FINISH,
    DEFAULT_RESOLUTION,
    FINISH_ALPHA_MAP,
    normalize_finish,
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
)


class ImageProcessor:
    """Handles livery alpha masking, resizing, and thumbnail generation."""

    @staticmethod
    def apply_finish(image: Image.Image, finish_type: str = DEFAULT_FINISH) -> Image.Image:
        """
        Writes the BODY alpha channel ("paint fx" mask) for the selected gloss level.
        Alpha scales the base livery's paint shader: 255 = Gloss, 210 = Semi-Gloss, 170 = Satin, 80 = Matte.
        Legacy names (Full Gloss, Metallic, Chrome) are accepted.
        """
        alpha_val = FINISH_ALPHA_MAP[normalize_finish(finish_type)]

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
        Generates the AMS2 UI preview image (2048 x 600 px by default, per the car _dist.xml spec).
        Uses centre cropping and Lanczos resampling.
        """
        rgb_img = image.convert("RGB")

        # Fit and crop to the exact target aspect ratio
        thumb = ImageOps.fit(
            rgb_img, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )
        return thumb

    @staticmethod
    def tint_texture(
        image: Image.Image,
        color: Tuple[int, int, int],
        strength: float = 1.0,
        keep_highlights: bool = True,
    ) -> Image.Image:
        """
        Recolours a texture (e.g. a wheel rim template) while keeping its shading/detail.
        Luminance is mapped black -> colour -> white (or black -> colour when keep_highlights is off),
        then blended over the original by `strength` (0..1). Alpha is preserved.
        """
        src = image.convert("RGBA")
        gray = ImageOps.grayscale(src)
        if keep_highlights:
            colored = ImageOps.colorize(gray, black=(0, 0, 0), white=(255, 255, 255), mid=tuple(color), midpoint=110)
        else:
            colored = ImageOps.colorize(gray, black=(0, 0, 0), white=tuple(color))
        colored = colored.convert("RGBA")
        colored.putalpha(src.getchannel("A"))
        strength = max(0.0, min(1.0, float(strength)))
        if strength >= 0.999:
            return colored
        return Image.blend(src, colored, strength)

    @staticmethod
    def tint_masked(
        image: Image.Image, mask: Image.Image, color: Tuple[int, int, int], strength: float = 1.0
    ) -> Image.Image:
        """Tints only where `mask` (L, white = affected) is set."""
        tinted = ImageProcessor.tint_texture(image, color, strength)
        base = image.convert("RGBA")
        return Image.composite(tinted, base, mask.convert("L").resize(base.size))

    @staticmethod
    def dominant_colors(image: Image.Image, count: int = 5) -> List[Tuple[int, int, int]]:
        """
        Returns the most common colours of a livery (ignoring near-black background),
        used to suggest rim colours that match the body.
        """
        small = image.convert("RGB").copy()
        small.thumbnail((256, 256))
        quant = small.quantize(colors=max(count * 3, 8), method=Image.Quantize.MEDIANCUT)
        palette = quant.getpalette() or []
        counts = sorted(quant.getcolors() or [], reverse=True)
        result: List[Tuple[int, int, int]] = []
        for _, idx in counts:
            r, g, b = palette[idx * 3: idx * 3 + 3]
            if max(r, g, b) < 28:  # skip the dark UV background
                continue
            if any(abs(r - c[0]) + abs(g - c[1]) + abs(b - c[2]) < 40 for c in result):
                continue
            result.append((r, g, b))
            if len(result) >= count:
                break
        return result or [(200, 200, 200)]

    @staticmethod
    def flatten_on(image: Image.Image, background: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
        """Composites an RGBA image over a solid colour, returning RGBA with full alpha."""
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, background + (255,))
        bg.alpha_composite(rgba)
        return bg
