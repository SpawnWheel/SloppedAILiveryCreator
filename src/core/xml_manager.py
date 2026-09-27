"""
AMS2 XML override manager.

Format (from the `<car>_dist.xml` files shipped with AMS2):

    <USER_OVERRIDES>
        <LIVERY_OVERRIDE LIVERY="51" NAME="My Racing Team #1" BASELIVERY="Default">
            <PREVIEWIMAGE PATH="myracingteam\\preview.dds" />
            <TEXTURE NAME="BODY" PATH="myracingteam\\body.dds" />
            <TEXTURE NAME="WHEELDIFF" PATH="myracingteam\\wheel_diff.dds" />
        </LIVERY_OVERRIDE>
    </USER_OVERRIDES>

Why this works on raw text instead of a DOM:
  Reiza's own _dist.xml files (and therefore every car XML people create by copying them) are
  frequently NOT well-formed XML - e.g. bmw_m4_gt3_dist.xml has a stray </LIVERY_OVERRIDE> after a
  comment that closes early. The game's parser shrugs that off, but a strict XML parser refuses the
  whole file, and the old implementation then rebuilt the file from scratch - silently dropping every
  other livery the user had configured. So we now:
    * blank out comments, find real <LIVERY_OVERRIDE> blocks with a regex,
    * parse each block on its own, and
    * splice our block into the original text, leaving everything else byte-for-byte untouched.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from xml.sax.saxutils import quoteattr

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_BLOCK = re.compile(r"<LIVERY_OVERRIDE\b[^>]*?(?:/>|>.*?</LIVERY_OVERRIDE\s*>)", re.S)
_ATTR = re.compile(r'([A-Za-z_][\w\-]*)\s*=\s*"([^"]*)"')
_ROOT_CLOSE = re.compile(r"</USER_OVERRIDES\s*>")


def _mask_comments(text: str) -> str:
    """Replaces comment bodies with spaces so offsets stay aligned with the original text."""
    return _COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def _iter_blocks(text: str) -> Iterator[Tuple[int, int, str]]:
    masked = _mask_comments(text)
    for m in _BLOCK.finditer(masked):
        yield m.start(), m.end(), text[m.start():m.end()]


def _parse_block(block: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(block)
    except Exception:
        pass
    # Lenient fallback: rebuild element from attributes + child tags found by regex
    head = re.match(r"<LIVERY_OVERRIDE\b([^>]*)>", block, re.S)
    if not head:
        return None
    elem = ET.Element("LIVERY_OVERRIDE", dict(_ATTR.findall(head.group(1))))
    for tag, attrs in re.findall(r"<(PREVIEWIMAGE|TEXTURE)\b([^>]*)/?>", _mask_comments(block)):
        ET.SubElement(elem, tag, dict(_ATTR.findall(attrs)))
    return elem


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


class XMLManager:
    """Reads, checks, and surgically edits AMS2 custom livery XML files."""

    @staticmethod
    def parse_overrides(xml_path: Path) -> List[Dict]:
        """
        Returns every active (non-commented) LIVERY_OVERRIDE as a dict:
        livery_id, name, base_livery, preview_path, texture_path (BODY), textures {NAME: PATH}.
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            return []
        try:
            text = _read_text(xml_path)
        except Exception as e:
            print(f"[XMLManager] Failed to read '{xml_path}': {e}")
            return []

        entries = []
        for _, _, block in _iter_blocks(text):
            elem = _parse_block(block)
            if elem is None:
                continue
            preview_node = elem.find("PREVIEWIMAGE")
            textures: Dict[str, str] = {}
            for tex in elem.findall("TEXTURE"):
                name = (tex.get("NAME") or "").upper()
                if name and name not in textures:
                    textures[name] = tex.get("PATH", "")
            body_path = textures.get("BODY", "") or (next(iter(textures.values())) if textures else "")
            entries.append(
                {
                    "livery_id": str(elem.get("LIVERY") or elem.get("LIVERY_ID") or "").strip(),
                    "name": elem.get("NAME", ""),
                    "base_livery": elem.get("BASELIVERY", "Default"),
                    "preview_path": preview_node.get("PATH", "") if preview_node is not None else "",
                    "texture_path": body_path,
                    "textures": textures,
                }
            )
        return entries

    @staticmethod
    def check_collision(xml_path: Path, livery_id: int) -> Optional[Dict]:
        """Returns the existing entry for a livery slot (ID), or None if the slot is free."""
        id_str = str(livery_id).strip()
        for entry in XMLManager.parse_overrides(xml_path):
            if entry["livery_id"] == id_str:
                return entry
        return None

    @staticmethod
    def _starting_text(xml_path: Path) -> str:
        """
        Existing car XML, or - for a brand-new one - a verbatim copy of `<car>_dist.xml`,
        which is exactly what AMS2's instructions ask you to do by hand.
        """
        if xml_path.exists() and xml_path.stat().st_size > 0:
            return _read_text(xml_path)
        dist = xml_path.with_name(f"{xml_path.stem}_dist.xml")
        if dist.exists() and dist.stat().st_size > 0:
            return _read_text(dist)
        return '<?xml version="1.0" encoding="utf-8" ?>\n<USER_OVERRIDES>\n</USER_OVERRIDES>\n'

    @staticmethod
    def _build_block(
        livery_id: str,
        team_name: str,
        base_livery: str,
        preview_path: str,
        textures: Dict[str, str],
        old_elem: Optional[ET.Element],
        indent: str,
        nl: str,
    ) -> str:
        inner = indent + "    "
        lines = [
            f"<LIVERY_OVERRIDE LIVERY={quoteattr(livery_id)} NAME={quoteattr(team_name)} "
            f"BASELIVERY={quoteattr(base_livery)}>",
            f"{inner}<PREVIEWIMAGE PATH={quoteattr(preview_path)} />",
        ]
        for name, rel in textures.items():
            lines.append(f"{inner}<TEXTURE NAME={quoteattr(name)} PATH={quoteattr(rel)} />")

        # Keep anything we don't manage (other TEXTURE names, HEADLIGHTPROJECTION, ...)
        if old_elem is not None:
            managed = {n.upper() for n in textures}
            for child in list(old_elem):
                if not isinstance(child.tag, str) or child.tag == "PREVIEWIMAGE":
                    continue
                if child.tag == "TEXTURE" and (child.get("NAME") or "").upper() in managed:
                    continue
                child.tail = None
                lines.append(inner + ET.tostring(child, encoding="unicode").strip())
        lines.append(f"{indent}</LIVERY_OVERRIDE>")
        return nl.join(lines)

    @staticmethod
    def upsert_livery(
        xml_path: Path,
        livery_id: int,
        team_name: str,
        rel_body_path: str,
        rel_preview_path: str,
        base_livery: str = "Default",
        extra_textures: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Inserts or replaces the LIVERY_OVERRIDE for `livery_id`. Every other override, every comment
        and every texture entry we don't manage is kept exactly as it was.
        extra_textures: {TEXTURE_NAME: relative_path}, e.g. {"WHEELDIFF": "team\\wheel_diff.dds"}
        """
        xml_path = Path(xml_path)
        id_str = str(livery_id).strip()
        text = XMLManager._starting_text(xml_path)
        nl = "\r\n" if "\r\n" in text else "\n"

        textures = {"BODY": rel_body_path.replace("/", "\\")}
        for k, v in (extra_textures or {}).items():
            textures[k.upper()] = v.replace("/", "\\")
        preview = rel_preview_path.replace("/", "\\")

        target: Optional[Tuple[int, int, ET.Element]] = None
        for start, end, block in _iter_blocks(text):
            elem = _parse_block(block)
            if elem is not None and str(elem.get("LIVERY") or elem.get("LIVERY_ID") or "").strip() == id_str:
                target = (start, end, elem)
                break

        if target is not None:
            start, end, old = target
            line_start = text.rfind("\n", 0, start) + 1
            indent = re.match(r"[ \t]*", text[line_start:start]).group(0)
            new_block = XMLManager._build_block(
                id_str, team_name, base_livery or "Default", preview, textures, old, indent, nl
            )
            text = text[:start] + new_block + text[end:]
        else:
            new_block = XMLManager._build_block(
                id_str, team_name, base_livery or "Default", preview, textures, None, "    ", nl
            )
            closes = list(_ROOT_CLOSE.finditer(_mask_comments(text)))
            if closes:
                pos = closes[-1].start()
                before = text[:pos].rstrip(" \t")
                if not before.endswith(("\n", "\r")):
                    before += nl
                text = before + "    " + new_block + nl + text[pos:]
            else:
                text = text.rstrip() + nl + "<USER_OVERRIDES>" + nl + "    " + new_block + nl + "</USER_OVERRIDES>" + nl

        XMLManager._atomic_write(xml_path, text)

    @staticmethod
    def _atomic_write(output_path: Path, text: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = output_path.with_name(output_path.name + ".tmp")
        with open(temp_file, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(temp_file, output_path)
