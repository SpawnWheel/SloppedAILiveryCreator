"""
"Prompt Pack" export: instead of calling Gemini, write everything the AI would have received into a folder
so it can be pasted into any AI website (ChatGPT, Gemini web, Midjourney, ...):

    <pack folder>/
        prompt.txt            - one self-contained prompt (system rules + design prompt + layout/flat-colour rules)
        01_template.png       - the UV template exactly as the app sends it (flattened onto white, padded to a
                                supported aspect ratio)
        02_<role>.png ...     - reference images, in the order the prompt names them
        pack_info.json        - geometry info so "Import AI Result" can undo the padding afterwards

Web chat UIs have no separate "system instruction" box, so the system rules are merged into prompt.txt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from src.core import template_fit

PACK_INFO_FILE = "pack_info.json"
PROMPT_FILE = "prompt.txt"
MAX_REF_DIM = 2048  # web uploads: keep files a sensible size


@dataclass
class PromptPack:
    folder: Path
    prompt_text: str
    image_paths: List[Path] = field(default_factory=list)
    prompt_path: Optional[Path] = None
    aspect_ratio: str = "1:1"


def default_packs_root() -> Path:
    return Path.home() / "Documents" / "AMS2 Livery Studio" / "Prompt Packs"


def _slug(text: str, fallback: str = "ref") -> str:
    s = re.sub(r"[^\w\s-]", "", (text or "").lower()).strip()
    s = re.sub(r"[\s-]+", "_", s)
    return s[:40] or fallback


def build_pack_prompt(user_prompt: str, system_instruction: str, has_template: bool,
                      n_refs: int, aspect_ratio: str = "1:1", prep=None) -> str:
    """Merges system rules + user prompt + the rules the app normally appends into one pasteable prompt."""
    from src.core.ai_engine import (FLAT_COLOUR_RULES, LAYOUT_LOCK_RULES, OPEN_HOLES_RULES, ORIENTATION_RULES,
                                    SYMMETRY_RULES, template_block)

    attach = []
    if has_template:
        attach.append("Image 1 = 01_template.png (the UV layout map / canvas)")
    first_ref = 2 if has_template else 1
    for i in range(n_refs):
        attach.append(f"Image {first_ref + i} = reference image {i + 1}")

    parts = [
        "Please GENERATE AN IMAGE (not a description) following the brief below.",
        f"Attached images, in order: {'; '.join(attach)}." if attach else "",
        f"Output: one flat, square-on texture sheet, aspect ratio {aspect_ratio}, as high resolution as you can "
        "(ideally 4096 px on the long side).",
        "",
        "=== ROLE & RULES ===",
        (system_instruction or "").strip(),
        "",
        "=== BRIEF ===",
        (user_prompt or "").strip(),
    ]
    if has_template and "HOW TO READ IMAGE 1" not in (user_prompt or ""):
        parts += ["", template_block(prep)]
    if has_template and "GEOMETRY LOCK" not in (user_prompt or ""):
        parts += ["", LAYOUT_LOCK_RULES]
    if has_template and "ARTWORK ORIENTATION (" not in (user_prompt or ""):
        parts += ["", ORIENTATION_RULES]
    if has_template and "OPEN HOLES / CUT-OUTS" not in (user_prompt or ""):
        parts += ["", OPEN_HOLES_RULES]
    if "LEFT/RIGHT SYMMETRY (" not in (user_prompt or ""):
        parts += ["", SYMMETRY_RULES]
    if "NO BAKED LIGHTING" not in (user_prompt or ""):
        parts += ["", FLAT_COLOUR_RULES]
    return "\n".join(p for p in parts if p is not None).strip() + "\n"


def create_prompt_pack(
    wireframe: Optional[Image.Image],
    references: List[Tuple[Image.Image, str]],
    user_prompt: str,
    system_instruction: str,
    root: Optional[Path] = None,
    name: str = "",
) -> PromptPack:
    """Writes the prompt pack folder and returns what was written."""
    root = Path(root) if root else default_packs_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = root / f"{_slug(name, 'livery')}_{stamp}"
    n = 1
    while folder.exists():
        n += 1
        folder = root / f"{_slug(name, 'livery')}_{stamp}_{n}"
    folder.mkdir(parents=True, exist_ok=True)

    images: List[Path] = []
    info = {"created": stamp, "has_template": wireframe is not None}
    aspect = "1:1"
    idx = 1
    prep = None

    if wireframe is not None:
        prep = template_fit.prepare_for_ai(wireframe)
        aspect = prep.aspect_ratio
        p = folder / f"{idx:02d}_template.png"
        prep.image.save(p)
        images.append(p)
        info.update(aspect_ratio=aspect, content_box=list(prep.content_box),
                    original_size=list(prep.original_size))
        idx += 1

    for ref_img, role in references:
        img = ref_img.convert("RGB") if ref_img.mode != "RGB" else ref_img.copy()
        if max(img.size) > MAX_REF_DIM:
            img.thumbnail((MAX_REF_DIM, MAX_REF_DIM), Image.Resampling.LANCZOS)
        p = folder / f"{idx:02d}_{_slug(role)}.png"
        img.save(p)
        images.append(p)
        idx += 1

    text = build_pack_prompt(user_prompt, system_instruction, wireframe is not None,
                             len(references), aspect, prep=prep)
    prompt_path = folder / PROMPT_FILE
    prompt_path.write_text(text, encoding="utf-8")
    (folder / PACK_INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")

    return PromptPack(folder=folder, prompt_text=text, image_paths=images,
                      prompt_path=prompt_path, aspect_ratio=aspect)


def find_pack_info(image_path: Path) -> Optional[dict]:
    """Looks for pack_info.json next to (or one level above) an imported result image."""
    image_path = Path(image_path)
    for d in (image_path.parent, image_path.parent.parent):
        f = d / PACK_INFO_FILE
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
    return None


def normalize_external_result(img: Image.Image, info: Optional[dict],
                              target_resolution: int = 4096) -> Image.Image:
    """
    Turns an image downloaded from an external AI into a texture sheet: crops off the padding the pack
    added (if any) and scales to the template's proportions at `target_resolution` on the long side.
    """
    img = img.convert("RGBA")
    if info and info.get("content_box") and info.get("original_size"):
        w, h = info["original_size"]
        if w >= h:
            out_size = (target_resolution, max(1, round(target_resolution * h / w)))
        else:
            out_size = (max(1, round(target_resolution * w / h)), target_resolution)
        prep = template_fit.PreparedTemplate(
            image=img, aspect_ratio=info.get("aspect_ratio", "1:1"),
            content_box=tuple(info["content_box"]), original_size=(w, h),
        )
        return template_fit.restore_geometry(img, prep, out_size).convert("RGBA")
    if img.size != (target_resolution, target_resolution):
        img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
    return img
