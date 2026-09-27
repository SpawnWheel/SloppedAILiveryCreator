"""
Single entry point for opening any image the studio deals with
(.png/.jpg/.tga/.bmp/.webp, Photoshop .psd, DirectX .dds) as an RGBA PIL image.
"""

from pathlib import Path
from PIL import Image


def load_any_image(path) -> Image.Image:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".dds":
        from src.core.dds_converter import DDSConverter
        return DDSConverter.load_dds_image(p)
    if ext == ".psd":
        from src.core.template_loader import TemplateLoader
        img, _ = TemplateLoader._load_from_psd(p, p.name)
        return img.convert("RGBA")
    if ext == ".zip":
        from src.core.template_loader import TemplateLoader
        img, _ = TemplateLoader.load_from_path(p)
        return img.convert("RGBA")
    with Image.open(p) as img:
        return img.convert("RGBA").copy()
