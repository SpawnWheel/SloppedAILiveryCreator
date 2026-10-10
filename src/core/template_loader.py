"""
Wireframe and template ingestion with archive handling and offline caching.
"""

import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import requests
from PIL import Image
from psd_tools import PSDImage

from src.utils.constants import SUPPORTED_IMAGE_EXTS
from src.utils.file_helpers import compute_string_hash, get_cache_dir


class TemplateLoader:
    """Handles loading, downloading, extracting, and caching UV wireframe templates."""

    @staticmethod
    def load_from_path(file_path: Path) -> Tuple[Image.Image, str]:
        """
        Loads a template from a local file (.png, .jpg, .psd, .zip, etc.).
        Returns (PIL.Image, description_or_filename).
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".zip":
            with open(file_path, "rb") as f:
                return TemplateLoader._extract_primary_image_from_zip(f.read(), file_path.name)

        if suffix == ".psd":
            return TemplateLoader._load_from_psd(file_path, file_path.name)

        if suffix in SUPPORTED_IMAGE_EXTS:
            img = Image.open(file_path)
            return img.convert("RGBA"), file_path.name

        raise ValueError(f"Unsupported file format: {suffix}")

    @staticmethod
    def _load_from_psd(source, display_name: str) -> Tuple[Image.Image, str]:
        """
        Robustly loads a Photoshop PSD file, compositing visible layers and handling
        various color modes (CMYK, 16-bit, Grayscale, RGBA).
        """
        psd = PSDImage.open(source)

        # 1. Try standard composite
        composite = None
        try:
            composite = psd.composite()
        except Exception:
            pass

        # 2. Try forced composite without preview
        if composite is None:
            try:
                composite = psd.composite(ignore_preview=True)
            except Exception:
                pass

        # 3. Fallback: Composite visible layers manually
        if composite is None:
            try:
                base_w, base_h = psd.width, psd.height
                composite = Image.new("RGBA", (base_w, base_h), (0, 0, 0, 0))
                for layer in psd:
                    if layer.is_visible():
                        layer_img = layer.composite()
                        if layer_img:
                            composite.paste(layer_img, (layer.left, layer.top), layer_img.convert("RGBA"))
            except Exception as e:
                print(f"[TemplateLoader] Manual layer composite error: {e}")

        if composite is None:
            raise ValueError(f"Could not render or composite PSD file '{display_name}'.")

        # Convert to 8-bit RGBA
        if composite.mode != "RGBA":
            composite = composite.convert("RGBA")

        TemplateLoader._register_psd_markers(psd, composite)
        return composite, display_name

    @staticmethod
    def _register_psd_markers(psd, composite: Image.Image) -> None:
        """
        Placeholder text layers (e.g. the '35' race numbers in Reiza's templates) carry their exact rotation in the
        PSD. Recording it lets the app tell the AI - with certainty - which panels are stored sideways/upside-down.
        """
        import math

        from src.core import template_fit

        markers = []
        try:
            for layer in psd.descendants():
                if layer.kind != "type" or not layer.is_visible():
                    continue
                text = (getattr(layer, "text", "") or "").strip()
                if not text or len(text) > 12:
                    continue
                xx, xy = layer.transform[0], layer.transform[1]
                rot = math.degrees(math.atan2(xy, xx)) % 360
                markers.append((text, tuple(layer.bbox), rot))
        except Exception as e:  # never fail a template load over this
            print(f"[TemplateLoader] Could not read PSD text layers: {e}")
        if markers:
            template_fit.register_markers(composite, markers)

    @staticmethod
    def load_from_url(url: str) -> Tuple[Image.Image, str]:
        """
        Downloads a template from a URL or loads from offline cache if already downloaded.
        Supports direct image URLs as well as ZIP archive URLs.
        """
        url = url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")

        # Check offline cache
        cache_dir = get_cache_dir()
        url_hash = compute_string_hash(url)
        cached_meta = cache_dir / f"{url_hash}.meta"
        cached_data = cache_dir / f"{url_hash}.bin"

        if cached_data.exists() and cached_meta.exists():
            try:
                with open(cached_meta, "r", encoding="utf-8") as f:
                    filename = f.read().strip()
                raw_bytes = cached_data.read_bytes()
                return TemplateLoader._load_from_bytes(raw_bytes, filename)
            except Exception as e:
                print(f"[TemplateLoader] Cache read failed: {e}. Re-downloading...")

        # Download from network
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AMS2-Livery-Studio/1.0"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Extract filename from URL or header
        filename = url.split("/")[-1].split("?")[0] or "downloaded_template"
        raw_bytes = response.content

        # Save to offline cache
        try:
            cached_data.write_bytes(raw_bytes)
            cached_meta.write_text(filename, encoding="utf-8")
        except Exception as e:
            print(f"[TemplateLoader] Failed to write cache: {e}")

        return TemplateLoader._load_from_bytes(raw_bytes, filename)

    @staticmethod
    def _load_from_bytes(raw_bytes: bytes, filename: str) -> Tuple[Image.Image, str]:
        """Parses image or zip data from in-memory bytes."""
        # Check if ZIP
        if zipfile.is_zipfile(io.BytesIO(raw_bytes)):
            return TemplateLoader._extract_primary_image_from_zip(raw_bytes, filename)

        # Check if PSD
        try:
            if raw_bytes.startswith(b"8BPS"):
                return TemplateLoader._load_from_psd(io.BytesIO(raw_bytes), filename)
        except Exception:
            pass

        # Normal image
        img = Image.open(io.BytesIO(raw_bytes))
        return img.convert("RGBA"), filename

    @staticmethod
    def _extract_primary_image_from_zip(zip_bytes: bytes, zip_name: str) -> Tuple[Image.Image, str]:
        """
        Extracts the primary wireframe image from a ZIP archive by finding
        the candidate image file with the highest resolution.
        """
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            candidates = []
            for name in zf.namelist():
                # Ignore macOS metadata or hidden files
                if "__MACOSX" in name or Path(name).name.startswith("."):
                    continue
                ext = Path(name).suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    candidates.append(name)

            if not candidates:
                raise ValueError("No supported image wireframe found in ZIP archive.")

            best_image: Optional[Image.Image] = None
            best_pixels = -1
            best_filename = candidates[0]

            for name in candidates:
                try:
                    data = zf.read(name)
                    ext = Path(name).suffix.lower()
                    if ext == ".psd":
                        img, _ = TemplateLoader._load_from_psd(io.BytesIO(data), name)
                        pixels = img.width * img.height
                        if pixels > best_pixels:
                            best_pixels = pixels
                            best_image = img
                            best_filename = name
                    else:
                        img = Image.open(io.BytesIO(data))
                        pixels = img.width * img.height
                        if pixels > best_pixels:
                            best_pixels = pixels
                            best_image = img.convert("RGBA")
                            best_filename = name
                except Exception as e:
                    print(f"[TemplateLoader] Error checking image '{name}': {e}")

            if best_image is None:
                raise ValueError("Could not decode any candidate wireframe image from ZIP archive.")

            display_name = f"{zip_name} ({Path(best_filename).name})"
            return best_image, display_name
