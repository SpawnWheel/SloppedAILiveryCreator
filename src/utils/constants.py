"""
Constants and application configuration defaults for AMS2 AI Livery Studio.
"""

from pathlib import Path
import os

# Application Info
APP_NAME = "AMS2 AI Livery Studio"
APP_VERSION = "1.1.0"
APP_ORGANIZATION = "SimModding"

# Gloss level -> BODY alpha value (0-255).
# In AMS2 the BODY texture's alpha channel is the "paint fx" mask (see any car's _dist.xml): it controls how
# strongly the BASE LIVERY's paint shader (clear-coat gloss, metallic flake, chrome...) shows through.
#   255 = full effect of the base paint (glossy on a normal base)
#    80 = mostly flat / matte
# The *type* of paint (metallic, chrome, matte shader) comes from the Base Livery chosen in the Install & Export tab;
# this setting only controls how much of it you get. That's why "Metallic" / "Chrome" are not options here -
# pick a Metallic / Chrome base livery instead (if the car has one).
FINISH_ALPHA_MAP = {
    "Gloss": 255,
    "Semi-Gloss": 210,
    "Satin": 170,
    "Matte": 80,
}

# Older names (saved configs, exported ZIP notes, tests) -> current names
FINISH_ALIASES = {
    "Full Gloss": "Gloss",
    "Chrome": "Gloss",
    "Metallic": "Semi-Gloss",
}

DEFAULT_FINISH = "Gloss"


def normalize_finish(name: str) -> str:
    """Maps any finish name (current or legacy, any case) onto a key of FINISH_ALPHA_MAP."""
    if not name:
        return DEFAULT_FINISH
    for key in FINISH_ALPHA_MAP:
        if key.lower() == name.strip().lower():
            return key
    for old, new in FINISH_ALIASES.items():
        if old.lower() == name.strip().lower():
            return new
    return DEFAULT_FINISH

# Texture Resolution
DEFAULT_RESOLUTION = 4096  # Standard 4K livery texture

# UI Preview image spec for AMS2.
# Every car's <car>_dist.xml states: "a PREVIEWIMAGE element ... for use in the game's UI (2048 x 600 px)"
PREVIEW_WIDTH = 2048
PREVIEW_HEIGHT = 600

# Texture slots the studio knows how to build, keyed by the AMS2 <TEXTURE NAME="..."> value.
# Specs come straight from the car _dist.xml files shipped with AMS2:
#   BODY          - 4096 x 4096, DXT5 with MIPs (alpha channel = paint fx)
#   WHEELDIFF     - rim_diff      - Wheel rim texture            - 1024 x 1024, DXT1 with MIPs
#   WHEELBLURDIFF - rim_blur_diff - Wheel rim (spinning) texture - 1024 x 1024, DXT5 with MIPs
TEXTURE_SPECS = {
    "BODY": {"file": "body", "size": 4096, "format": "BC3_UNORM", "label": "Body Livery"},
    "WHEELDIFF": {"file": "wheel_diff", "size": 1024, "format": "BC1_UNORM", "label": "Wheel Rim"},
    "WHEELBLURDIFF": {"file": "wheel_blur_diff", "size": 1024, "format": "BC3_UNORM", "label": "Wheel Rim (Blur / Spinning)"},
}
WHEEL_TEXTURE_NAMES = ["WHEELDIFF", "WHEELBLURDIFF"]

DEFAULT_BASE_LIVERY = "Default"

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

# File dialog filter for anything the studio can open as a texture / template
IMAGE_OPEN_FILTER = (
    "Images & Templates (*.png *.jpg *.jpeg *.bmp *.tga *.webp *.psd *.dds);;"
    "PNG (*.png);;DDS Textures (*.dds);;Photoshop (*.psd);;All Files (*.*)"
)

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
    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview",
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

