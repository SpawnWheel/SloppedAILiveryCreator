"""
AMS2 XML override manager.
Provides surgical parsing, collision checking, and non-destructive upserting
for customliveries.xml files.
"""

import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


class XMLManager:
    """Manages reading, collision checking, and surgical editing of AMS2 custom liveries XML files."""

    @staticmethod
    def parse_overrides(xml_path: Path) -> List[Dict[str, str]]:
        """
        Parses an existing AMS2 override XML file and returns a list of livery metadata dicts.
        """
        if not xml_path.exists():
            return []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            entries = []

            for elem in root.findall(".//LIVERY_OVERRIDE"):
                livery_id = elem.get("LIVERY") or elem.get("LIVERY_ID") or ""
                name = elem.get("NAME", "")
                base_livery = elem.get("BASELIVERY", "Default")

                preview_node = elem.find("PREVIEWIMAGE")
                preview_path = preview_node.get("PATH", "") if preview_node is not None else ""

                texture_node = elem.find(".//TEXTURE[@NAME='BODY']")
                if texture_node is None:
                    texture_node = elem.find("TEXTURE")
                texture_path = texture_node.get("PATH", "") if texture_node is not None else ""

                entries.append(
                    {
                        "livery_id": str(livery_id),
                        "name": name,
                        "base_livery": base_livery,
                        "preview_path": preview_path,
                        "texture_path": texture_path,
                    }
                )
            return entries
        except Exception as e:
            print(f"[XMLManager] Failed to parse XML '{xml_path}': {e}")
            return []

    @staticmethod
    def check_collision(xml_path: Path, livery_id: int) -> Optional[Dict[str, str]]:
        """
        Checks if a livery slot (ID) is already occupied in the target XML.
        Returns the existing entry details if occupied, or None if free.
        """
        id_str = str(livery_id).strip()
        entries = XMLManager.parse_overrides(xml_path)
        for entry in entries:
            if entry["livery_id"] == id_str:
                return entry
        return None

    @staticmethod
    def upsert_livery(
        xml_path: Path,
        livery_id: int,
        team_name: str,
        rel_body_path: str,
        rel_preview_path: str,
        base_livery: str = "Default",
    ) -> None:
        """
        Surgically inserts or updates a LIVERY_OVERRIDE entry in the target XML file.
        Preserves other overrides and maintains clean XML formatting.
        """
        id_str = str(livery_id).strip()
        # Normalise Windows path slashes for AMS2 compatibility
        norm_body = rel_body_path.replace("/", "\\")
        norm_preview = rel_preview_path.replace("/", "\\")

        root = None
        tree = None

        if xml_path.exists() and xml_path.stat().st_size > 0:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"[XMLManager] Existing XML malformed ({e}), creating fresh root.")
                root = ET.Element("USER_OVERRIDES")
                tree = ET.ElementTree(root)
        else:
            root = ET.Element("USER_OVERRIDES")
            tree = ET.ElementTree(root)

        # Locate existing override node for this ID
        target_node = None
        for elem in root.findall(".//LIVERY_OVERRIDE"):
            elem_id = elem.get("LIVERY") or elem.get("LIVERY_ID")
            if str(elem_id).strip() == id_str:
                target_node = elem
                break

        if target_node is not None:
            # Update existing node attributes
            target_node.set("LIVERY", id_str)
            target_node.set("NAME", team_name)
            target_node.set("BASELIVERY", base_livery)

            # Update or create PREVIEWIMAGE
            prev_elem = target_node.find("PREVIEWIMAGE")
            if prev_elem is None:
                prev_elem = ET.SubElement(target_node, "PREVIEWIMAGE")
            prev_elem.set("PATH", norm_preview)

            # Update or create TEXTURE
            tex_elem = target_node.find(".//TEXTURE[@NAME='BODY']")
            if tex_elem is None:
                tex_elem = target_node.find("TEXTURE")
            if tex_elem is None:
                tex_elem = ET.SubElement(target_node, "TEXTURE")
            tex_elem.set("NAME", "BODY")
            tex_elem.set("PATH", norm_body)
        else:
            # Create new LIVERY_OVERRIDE node
            new_override = ET.SubElement(root, "LIVERY_OVERRIDE")
            new_override.set("LIVERY", id_str)
            new_override.set("NAME", team_name)
            new_override.set("BASELIVERY", base_livery)

            prev_elem = ET.SubElement(new_override, "PREVIEWIMAGE")
            prev_elem.set("PATH", norm_preview)

            tex_elem = ET.SubElement(new_override, "TEXTURE")
            tex_elem.set("NAME", "BODY")
            tex_elem.set("PATH", norm_body)

        # Write out formatted XML
        XMLManager._save_pretty_xml(root, xml_path)

    @staticmethod
    def _save_pretty_xml(root_elem: ET.Element, output_path: Path) -> None:
        """Serializes the ElementTree root with clean indentation and XML header."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_xml = ET.tostring(root_elem, encoding="utf-8")
        parsed = xml.dom.minidom.parseString(raw_xml)
        pretty = parsed.toprettyxml(indent="    ", encoding="utf-8")

        # Strip extra blank lines introduced by minidom
        lines = [line for line in pretty.decode("utf-8").splitlines() if line.strip()]
        clean_xml = "\n".join(lines) + "\n"

        # Safe atomic write via temporary file
        temp_file = output_path.with_suffix(".tmp")
        temp_file.write_text(clean_xml, encoding="utf-8")
        if output_path.exists():
            output_path.unlink()
        temp_file.rename(output_path)
