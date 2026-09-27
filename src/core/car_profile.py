"""
Per-car capability profile read from the AMS2 `<car>_dist.xml` files.

Every car folder in `Vehicles/Textures/CustomLiveries/Overrides/<car>/` ships with a
`<car>_dist.xml` whose big header comment documents:
  * the BASE LIVERIES you can build on (e.g. "Default", "Matte", or paint names like "ViperWhite")
  * which TEXTURE NAMEs can be overridden (BODY, WHEELDIFF, WHEELBLURDIFF, TIREDIFF, ...)

That text is free-form, so this parser is deliberately forgiving.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.utils.constants import DEFAULT_BASE_LIVERY, WHEEL_TEXTURE_NAMES

_HEADER_UNDERLINE = re.compile(r"^\s*={3,}\s*$")
_BASE_SECTION = re.compile(r"AVAILABLE\s+BASE\s+(LIVERIES|LIVERY|PAINTS|PAINT|SKINS)", re.I)
_SKIP_SECTION = re.compile(r"HELMET|OUTFIT|EXAMPLE", re.I)
_BASE_NAME_COLON = re.compile(r"^\s*-?\s*([A-Za-z0-9][\w\-]*(?: [\w\-]+){0,3})\s*:\s*\S")
_BASE_NAME_DASH = re.compile(r"^\s*-\s*([A-Za-z0-9][\w\-]*)\s*(?:\(.*\))?\s*$")
_TEX_DOC = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s+-\s")
_TEX_ATTR = re.compile(r'TEXTURE\s+NAME\s*=\s*"([A-Za-z0-9_]+)"')
_NOT_BASE_NAMES = {"available textures", "eg", "note", "livery ids"}


@dataclass
class CarProfile:
    car_folder: str
    base_liveries: List[str] = field(default_factory=lambda: [DEFAULT_BASE_LIVERY])
    texture_names: List[str] = field(default_factory=lambda: ["BODY"])
    dist_xml: Optional[Path] = None

    def supports(self, texture_name: str) -> bool:
        return texture_name.upper() in self.texture_names

    @property
    def supports_wheels(self) -> bool:
        return any(self.supports(n) for n in WHEEL_TEXTURE_NAMES)

    @property
    def default_base_livery(self) -> str:
        return self.base_liveries[0] if self.base_liveries else DEFAULT_BASE_LIVERY


def _split_sections(text: str) -> List[tuple]:
    """Splits the dist comment into (header, body_lines) using the '=====' underline convention."""
    lines = text.splitlines()
    headers = []
    for i, line in enumerate(lines):
        if _HEADER_UNDERLINE.match(line):
            # header = closest non-empty line above
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0:
                headers.append((j, i))
    sections = []
    for idx, (h_line, underline) in enumerate(headers):
        end = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        sections.append((lines[h_line].strip(), lines[underline + 1:end]))
    return sections


def parse_dist_text(text: str, car_folder: str = "") -> CarProfile:
    """Parses the raw text of a _dist.xml (comments included) into a CarProfile."""
    base_liveries: List[str] = []
    textures: List[str] = []

    def add_tex(name: str):
        name = name.upper()
        if name not in textures:
            textures.append(name)

    for header, body in _split_sections(text):
        if _SKIP_SECTION.search(header):
            continue
        is_base = bool(_BASE_SECTION.search(header))
        for line in body:
            if is_base:
                m = _BASE_NAME_COLON.match(line) or _BASE_NAME_DASH.match(line)
                if m and not _TEX_DOC.match(line):
                    cand = m.group(1).strip()
                    if cand.lower() not in _NOT_BASE_NAMES and cand not in base_liveries:
                        base_liveries.append(cand)
            tm = _TEX_DOC.match(line)
            if tm:
                add_tex(tm.group(1))
            for am in _TEX_ATTR.finditer(line):
                add_tex(am.group(1))

    # Real (uncommented) overrides already in the file also tell us about bases
    try:
        root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
        for elem in root.iter("LIVERY_OVERRIDE"):
            b = elem.get("BASELIVERY")
            if b and b not in base_liveries:
                base_liveries.append(b)
            for t in elem.iter("TEXTURE"):
                if t.get("NAME"):
                    add_tex(t.get("NAME"))
    except Exception:
        pass

    if "BODY" not in textures:
        textures.insert(0, "BODY")
    if not base_liveries:
        base_liveries = [DEFAULT_BASE_LIVERY]
    return CarProfile(car_folder=car_folder, base_liveries=base_liveries, texture_names=textures)


def find_dist_xml(car_dir: Path, car_folder: str) -> Optional[Path]:
    if not car_dir.exists():
        return None
    exact = car_dir / f"{car_folder}_dist.xml"
    if exact.exists():
        return exact
    for p in car_dir.glob("*_dist.xml"):
        return p
    return None


def load_car_profile(overrides_root: Path, car_folder: str) -> CarProfile:
    """Builds the profile for a car. Falls back to a BODY-only 'Default' profile when unknown."""
    car_dir = Path(overrides_root) / car_folder
    dist = find_dist_xml(car_dir, car_folder)
    profile = CarProfile(car_folder=car_folder)
    if dist is not None:
        try:
            profile = parse_dist_text(dist.read_text(encoding="utf-8", errors="replace"), car_folder)
            profile.dist_xml = dist
        except Exception as e:
            print(f"[CarProfile] Could not parse {dist}: {e}")

    # Merge any base liveries the user's own XML already uses
    user_xml = car_dir / f"{car_folder}.xml"
    if user_xml.exists():
        try:
            extra = parse_dist_text(user_xml.read_text(encoding="utf-8", errors="replace"), car_folder)
            for b in extra.base_liveries:
                if b not in profile.base_liveries and b != DEFAULT_BASE_LIVERY:
                    profile.base_liveries.append(b)
            for t in extra.texture_names:
                if t not in profile.texture_names:
                    profile.texture_names.append(t)
        except Exception:
            pass
    return profile
