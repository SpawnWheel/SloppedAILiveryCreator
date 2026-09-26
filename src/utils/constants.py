"""
Constants and application configuration defaults for AMS2 AI Livery Studio.
"""

from pathlib import Path
import os

# Application Info
APP_NAME = "AMS2 AI Livery Studio"
APP_VERSION = "1.0.0"
APP_ORGANIZATION = "SimModding"

# Finish Types to Alpha Values (0-255)
# AMS2 Shader rules:
# Alpha controls reflectivity / glossiness in the livery DDS (BC3 / DXT5 format)
# 255 (White) = High Gloss / Highly reflective
# 210 = Semi-Gloss
# 170 = Satin / Eggshell
# 80  = Matte (low reflectivity)
# Metallic = 200 (combined with BASELIVERY metallic shaders)
# Chrome = 255
FINISH_ALPHA_MAP = {
    "Full Gloss": 255,
    "Semi-Gloss": 210,
    "Satin": 170,
    "Matte": 80,
    "Metallic": 200,
    "Chrome": 255,
}

DEFAULT_FINISH = "Matte"

# Texture Resolution
DEFAULT_RESOLUTION = 4096  # Standard 4K livery texture

# UI Preview Thumbnail specs for AMS2
PREVIEW_WIDTH = 252
PREVIEW_HEIGHT = 64

# Reference Image Roles
REFERENCE_ROLES = [
    "Color Palette / Theme",
    "Graphic Pattern / Accents",
    "Sponsor / Decal",
]

# Supported Input File Extensions
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".psd", ".webp"}
SUPPORTED_ARCHIVE_EXTS = {".zip"}
SUPPORTED_LIVERY_INSTALL_EXTS = {".dds", ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp"}

# External Tools
TEXCONV_DOWNLOAD_URL = (
    "https://github.com/microsoft/DirectXTex/releases/latest/download/texconv.exe"
)

# AI Models (Gemini Flash Image / Gemini Pro Image)
DEFAULT_AI_MODEL = "gemini-3.1-flash-image"
FALLBACK_AI_MODEL = "gemini-3-pro-image"
AVAILABLE_AI_MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]

# Auto-discovery for AMS2 Overrides Path across Steam libraries and Documents
def get_default_ams2_path() -> Path:
    import string

    candidates = []
    # 1. Search Steam library paths across all active system drive letters
    for letter in string.ascii_uppercase:
        drive_path = f"{letter}:\\"
        if not os.path.exists(drive_path):
            continue
        candidates.extend([
            Path(f"{letter}:/SteamLibrary/steamapps/common/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides"),
            Path(f"{letter}:/Program Files (x86)/Steam/steamapps/common/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides"),
            Path(f"{letter}:/Program Files/Steam/steamapps/common/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides"),
            Path(f"{letter}:/Steam/steamapps/common/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides"),
            Path(f"{letter}:/Games/Steam/steamapps/common/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides"),
            Path(f"{letter}:/Games/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides"),
        ])

    # 2. Documents folder fallback
    user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    candidates.append(
        Path(user_profile)
        / "Documents"
        / "Automobilista 2"
        / "Vehicles"
        / "Textures"
        / "CustomLiveries"
        / "Overrides"
    )

    # First preference: any existing directory that actually contains car folders
    for cand in candidates:
        if cand.exists():
            try:
                sub_count = sum(1 for _ in cand.iterdir())
                if sub_count > 0:
                    return cand
            except Exception:
                pass

    # Second preference: any directory that exists
    for cand in candidates:
        if cand.exists():
            return cand

    return candidates[0]

