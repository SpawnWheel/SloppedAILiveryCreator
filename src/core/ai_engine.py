"""
AI Generation Engine integrating Google Gemini multimodal image generation
(Gemini 3.1 Flash Image / Gemini 3 Pro Image).
"""

import io
import re
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.core import template_fit
from src.core.config import config
from src.utils.constants import AVAILABLE_AI_MODELS, DEFAULT_AI_MODEL, FALLBACK_AI_MODEL


LAYOUT_LOCK_RULES = (
    "GEOMETRY LOCK (most important): Image 1 is the exact canvas. Your output must overlay it pixel-for-pixel: "
    "same framing, same zoom, no added border or margin, no rotation, no mirroring, no cropping. Every UV island "
    "(panel) must stay at exactly the same position, size and outline as in Image 1 - only fill them with paint "
    "and graphics. Do not add, remove, merge or re-arrange panels. Paint must not spill outside the islands. "
    "(The sheet and its islands never rotate - but the ARTWORK inside each island must be rotated to suit that "
    "island's own orientation; see ARTWORK ORIENTATION.)"
)

# How to read the high-contrast layout map template_fit.prepare_for_ai() sends as Image 1.
TEMPLATE_MAP_RULES = (
    "HOW TO READ IMAGE 1 (the layout map):\n"
    "- Dark charcoal areas = empty background. They must stay exactly that dark colour in your output - never paint "
    "there, never put flames, logos, numbers or texture there.\n"
    "- Light-grey shapes with thick black outlines = the UV panels. Paint INSIDE them only, filling each one edge to "
    "edge, following its exact outline (including notches, holes and cut-outs). Faint grey lines inside are the 3D "
    "mesh - they show how the panel curves; remove them in the final paint.\n"
    "- Black placeholder digits (if any) = race-number positions and orientation markers.\n"
    "- OPENINGS ARE HOLES, NOT PAINT: see OPEN HOLES below - small enclosed shapes inside a panel are cut-outs "
    "that stay unpainted.\n"
    "- This layout is unique to this car. Do NOT draw a layout from memory of other templates, do NOT change the "
    "panel SHAPES to make them symmetrical unless Image 1 is (this is about the outlines only - the paint DESIGN is "
    "mirrored left/right, see LEFT/RIGHT SYMMETRY), and do NOT invent panel shapes from the reference photos. The "
    "reference images show the assembled car only to give you colours and graphics - their shape tells you nothing "
    "about the sheet.\n"
    "- Self-check: if your output were laid over Image 1, every painted area must sit on a grey panel (and not inside "
    "an opening) and every grey panel must be painted. If not, redo it."
)

# Templates are full of openings inside the panels (headlight/tail-light housings, vents, intakes, fuel filler,
# mirror mounts, door handles, number-light recesses...). The model used to paint straight over them, because in
# the layout map they sit inside a panel. They must stay open so the real geometry behind them shows in game.
OPEN_HOLES_RULES = (
    "OPEN HOLES / CUT-OUTS (important - leave them unpainted):\n"
    "- Many panels contain openings: small closed shapes INSIDE a panel - rounded rectangles, circles, ovals, "
    "slots and irregular notches - for headlights, tail-lights, vents, air intakes, grille openings, fuel fillers, "
    "mirror mounts, door handles, tow hooks, number-plate recesses and the like.\n"
    "- How to spot one in the template: a shape inside a panel that is ringed by its own tight closed outline (often "
    "a double line or a ring of dense mesh edges around it), and whose inside is EMPTY - the panel's mesh grid does "
    "not run through it, it is darker than the panel, or it simply has no wireframe lines inside. Mesh lines bend "
    "and bunch up AROUND an opening but never cross it.\n"
    "- Every opening stays exactly as it is in Image 1: do NOT fill it with paint, colour, stripes, logos, numbers, "
    "flames or texture, and do NOT let graphics run across it. Paint the panel right up to the opening's outline and "
    "stop there, so the hole keeps its exact shape and size. If a graphic (number, logo, stripe) would cross an "
    "opening, move or resize it so it sits on solid panel instead.\n"
    "- If unsure whether a small enclosed shape is an opening or part of the panel, treat it as an opening and leave "
    "it unpainted."
)

# Image models paint each side as its own picture, so the left and right of the car came out with different
# stripes/graphics. Real liveries are mirrored left/right unless the brief deliberately says otherwise.
SYMMETRY_RULES = (
    "LEFT/RIGHT SYMMETRY (default - unless the design instructions explicitly ask for an asymmetric design):\n"
    "- The car's left and right sides carry the SAME design, mirrored across the car's centreline: same base colour, "
    "same stripes, flames, shapes and colour splits, at the same positions along the car (same distance from the "
    "front and from the sill/roofline), and the same race number, sponsors and driver name in matching places on "
    "both sides. Seen from either side, the car must look like the same livery.\n"
    "- Top surfaces that straddle the centreline (hood/bonnet, roof, nose, front and rear bumpers, rear wing, "
    "boot/engine cover) are symmetrical about that centreline: whatever is on the left half is mirrored on the right "
    "half; centre stripes stay centred.\n"
    "- Match the sides by car part, not by position on the sheet: the two side islands are usually mirror copies "
    "and may be rotated or placed far apart on the sheet. Use the ORIENTATION map to find each side's front and "
    "ground edge, then mirror the design front-to-front and ground-to-ground.\n"
    "- Mirror the DESIGN, never the lettering: numbers, words and logos are repainted on the other side so they read "
    "normally (not back-to-front) - only their position and the surrounding graphics are mirrored.\n"
    "- Only break the symmetry if the design instructions say so (e.g. 'different design on each side', 'number on "
    "the left door only', 'asymmetric'). Then follow those instructions for exactly what they mention and keep "
    "everything else mirrored."
)

# Image models tend to paint the whole sheet "upright" as if it were a poster, but UV islands are unwrapped in
# arbitrary directions (side panels are often sideways or upside-down, left/right are mirror copies), so numbers,
# text and illustrations ended up sideways or upside-down on the car. This forces a per-island orientation pass.
ORIENTATION_RULES = (
    "ARTWORK ORIENTATION (very important - think this through BEFORE painting anything):\n"
    "The UV sheet is NOT a poster with one 'up' direction. Each island is a piece of the car unwrapped in its own "
    "direction: side panels are often lying sideways or upside-down on the sheet, the left and right sides are "
    "usually mirror images of each other (one is often rotated 180 degrees), and the hood/roof/bumpers can point "
    "any way. Artwork that is upright on the sheet will therefore often end up sideways or upside-down on the car.\n"
    "STEP 1 - Build an orientation map (silently, before painting). For EVERY island in the template decide:\n"
    "  a) which car part it is (left side, right side, hood/bonnet, roof, front bumper, rear bumper/boot, "
    "wing/spoiler, mirror...). Use any text labels on the template (e.g. LEFT/RIGHT/ROOF/FRONT) first.\n"
    "  b) where the car's GROUND side is (for vertical panels) - clues: wheel-arch cut-outs and the sill/rocker "
    "line are on the ground edge; window openings, the roofline and the beltline are on the up edge.\n"
    "  c) where the car's FRONT is - clues: headlight, grille and front-wheel-arch cut-outs are at the front; "
    "tail-light, exhaust and rear-wheel-arch cut-outs are at the rear; the front wheel arch is nearer the "
    "front than the door shut-lines.\n"
    "STEP 2 - Paint each graphic in the island's own frame, not the sheet's frame:\n"
    "  - Side panels: the TOP of every number, letter, logo and figure points toward the roof/windows and the "
    "BOTTOM toward the wheel arches/sill - regardless of how that island is turned on the sheet. On an island "
    "lying upside-down on the sheet, the artwork must look upside-down on the sheet so it is upright on the car.\n"
    "  - Text must NEVER be mirrored. Both sides of the car read normally (left-to-right) when viewed from "
    "outside that side. If the right-side island is a mirror copy of the left, do NOT copy the left-side "
    "artwork across flipped - repaint it so it reads correctly.\n"
    "  - Hood/bonnet and roof: unless the design instructions say otherwise, the top of numbers/text points "
    "toward the FRONT of the car (upright from the chase/TV camera behind and above). Keep every top-surface "
    "graphic in the same reading direction.\n"
    "  - Front and rear bumpers: upright relative to the ground (top of letters toward the hood / boot-lid edge).\n"
    "  - Illustrations, portraits and scenic artwork: 'up' in the picture = up on the car (toward the roof on "
    "sides; toward the front on top surfaces). Stripes that run along the car follow the car's front-to-rear axis "
    "on every island, so they line up across panel seams.\n"
    "  - If the TEMPLATE LAYOUT section lists a PANEL ORIENTATION for a panel, that is read from the template file "
    "and overrides every default above.\n"
    "STEP 3 - Before finishing, check every number, word, logo and figure against the orientation map. Anything "
    "sideways, upside-down or mirrored relative to its car part must be fixed."
)

# The texture is wrapped onto a 3D car and lit by the game engine; baked-in *scene* light would be lit a second
# time. But an earlier, stricter "flat unlit colour" rule also flattened the ARTWORK itself (cartoon flames, empty
# panels), so this separates the two: no lighting from the car's shape/environment, full richness in the graphics.
FLAT_COLOUR_RULES = (
    "NO BAKED LIGHTING - BUT RICH, DETAILED ARTWORK (important):\n"
    "- This is an albedo texture: AMS2 adds the car's lighting, reflections, gloss and shadows itself. So do NOT "
    "paint light that comes from the car's shape or the environment: no specular highlights or glare, no sky or "
    "environment reflections, no clear-coat shine, no shadows cast by the bodywork, no dark creases/ambient "
    "occlusion, no darkening or vignetting toward panel edges. Ignore that kind of lighting in the reference photos.\n"
    "- The ARTWORK itself should have full depth and detail, like a professional airbrushed or printed race livery: "
    "realistic flames with glowing hot cores, colour transitions and wispy tips; gradients, fades, glows, outlines, "
    "drop shadows and 3D-styled lettering are part of the design and are welcome when they fit the brief or the "
    "reference. Match the richness, brightness and density of the reference livery - do not simplify it into flat "
    "cartoon shapes, and do not leave panels as large plain areas unless the design calls for that.\n"
    "- Rule of thumb: a panel's base paint is even (not lit); the graphics painted on it can be as detailed and "
    "vivid as the design needs."
)


def template_block(prep, image_no: int = 1) -> str:
    """Map-reading rules + the measured panel/marker list for a prepared template."""
    text = TEMPLATE_MAP_RULES
    if prep is not None and prep.layout_text:
        text += "\n\n" + prep.layout_text
    if image_no != 1:
        text = text.replace("IMAGE 1", f"IMAGE {image_no}").replace("Image 1", f"Image {image_no}")
    return text


class AIEngine:
    """Manages AI-driven livery texture generation with multimodal input."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or config.api_key

    def set_api_key(self, key: str) -> None:
        self._api_key = key.strip()
        config.api_key = self._api_key

    def is_configured(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    @staticmethod
    def build_default_system_instruction() -> str:
        """Returns the system instruction enforcing strict 2D UV texture mapping rules."""
        return (
            "You are an expert automotive texture artist and livery designer for racing simulators (Automobilista 2).\n"
            "Your task is to take the provided 2D UV wireframe template (Image 1) and perform an IN-PLACE 2D TEXTURE PAINT to create a finished, game-ready 2D livery texture map sheet.\n\n"
            "CRITICAL MANDATORY RULES:\n"
            "1. STRICT FLAT 2D TEXTURE MAP: The output must be an orthographic, flat 2D unwrapped texture map sheet with a 1:1 square aspect ratio.\n"
            "2. ABSOLUTELY NO 3D VEHICLE RENDERS: Under NO circumstances should you render a 3D perspective car, assembled vehicle, wheels, tires, ground, or asphalt. You are painting an unwrapped 2D texture sheet, NOT rendering a car photograph.\n"
            "3. IN-PLACE PAINT ON THE LAYOUT MAP: Image 1 is a layout map of THIS car's UV sheet - light-grey panels with black outlines on a dark background. Each panel is an unrolled piece of the body (sides, nose, bonnet, cockpit surround, wings/fenders, tail...). "
            "The layout is unique to this car: never substitute a generic or symmetrical layout, and never derive panel shapes from the reference photos.\n"
            "   - In the exact positions, orientation, and silhouettes of these islands from Image 1, paint solid opaque automotive body paint, racing stripes, sponsor logos, and racing numbers.\n"
            "   - Completely eliminate and replace all wireframe/grid lines with smooth, solid paint and vinyl graphics. No wireframe lines should remain on the painted panels.\n"
            "4. TRANSFER GRAPHICS FROM REFERENCES: Reference images (Images 2..N) are strictly for transferring color scheme, logos, and styling onto the flat panels. Never draw or paste the 3D car or photo scene from reference images.\n"
            "5. CLEAN BACKGROUND: Outside of the UV panel islands, keep the background solid dark (#121418).\n"
            "6. PRINT-QUALITY GRAPHICS: Sharp, legible sponsor decals, racing numbers and text; rich, detailed artwork (realistic airbrushed flames, glows, gradients, outlined/shadowed lettering) matching the depth of the reference livery - never simplified into flat cartoon shapes.\n"
            "7. NO BAKED LIGHTING: The game engine adds all lighting, gloss and reflections, so never paint light from the car's shape or surroundings (highlights, glare, reflections, clear-coat shine, bodywork shadows, edge darkening). "
            "Depth inside the artwork itself is welcome.\n"
            "8. ARTWORK ORIENTATION: Islands are unwrapped in different directions - side panels are often sideways or upside-down on the sheet and left/right are mirror copies. "
            "Before painting, work out for each island which car part it is, where its front is and where its ground/roof edges are (wheel arches = ground edge, windows/roofline = up edge). "
            "Then orient every number, text, logo and illustration to be upright ON THE CAR, not on the sheet, and never mirror text.\n"
            "9. OPEN HOLES: Small closed shapes inside a panel (light housings, vents, intakes, fuel filler, handles, mirror mounts) are cut-outs. "
            "Leave every one of them unpainted with its exact shape - paint around them, never over them.\n"
            "10. LEFT/RIGHT SYMMETRY: Unless the design instructions explicitly ask for an asymmetric design, the left and right sides of the car carry the same, mirrored design "
            "(same colours, stripes, graphics, number and sponsor placement), and centreline surfaces (hood, roof, bumpers, wing) are symmetrical. Mirror the design, never the lettering."
        )

    @staticmethod
    def build_default_user_prompt(
        car_name: str = "",
        template_name: str = "",
        user_guidance: str = "",
        reference_roles: Optional[List[str]] = None,
        finish: str = "",
    ) -> str:
        """Assembles a structured text prompt strictly from the reference images, template layout, and user style text."""
        parts = [
            "TASK: Paint the 2D unwrapped UV livery texture map based on the provided wireframe layout.",
            "",
        ]
        if car_name or template_name:
            parts.append("--- VEHICLE SPECIFICATION ---")
            if car_name:
                parts.append(f"- Vehicle / Car Model: {car_name}")
            if template_name:
                parts.append(f"- Template Reference: {template_name}")
            # NOTE: the gloss/finish setting is deliberately NOT sent to the AI. It is applied in game via the
            # texture's alpha channel; mentioning "matte"/"gloss" makes the model paint a lit, textured surface.
            parts.append("")

        parts.append("--- INPUT ASSETS ---")
        parts.append("- Image 1: UV layout map of this car (light-grey panels = paint here, dark = background). Paint it in place - its geometry is fixed.")
        if reference_roles:
            for i, role in enumerate(reference_roles, start=2):
                parts.append(f"- Image {i}: Reference Image [{role}] (Take colors, logos and decals only - not its shapes or camera view).")
        else:
            parts.append("- Images 2+: Reference images for color scheme, sponsor decals, and styling only (not shapes).")

        parts.append("")
        parts.append("--- DESIGN & STYLING INSTRUCTIONS ---")
        if user_guidance and user_guidance.strip():
            parts.append(user_guidance.strip())
        else:
            parts.append(
                "Extract colors, stripes, and sponsor branding from the reference image(s). "
                "Fit and apply them onto the flat panels (hood, roof, doors, bumpers, mirrors) of Image 1."
            )

        parts.append("")
        parts.append("--- FINAL EXECUTION INSTRUCTIONS ---")
        parts.append("0. FIRST identify every island in Image 1 (which car part, where its front is, which edge is the ground/wheel-arch side and which is the roof/window side). "
                     "Plan each graphic in that island's own orientation so it is upright on the car - not upright on the sheet.")
        parts.append("1. Paint all UV islands (hood, roof, sides, bumpers, mirrors) with solid paint and decals strictly following the design instructions and reference images above.")
        parts.append("   Make the artwork rich and detailed like a professional airbrushed/printed livery (match the reference's depth), but paint no scene lighting - no highlights, reflections, shine or bodywork shadows; the game adds those.")
        parts.append("2. Strictly maintain the exact 1:1 square 2D layout and island coordinates from Image 1.")
        parts.append("3. Output ONLY the flat 2D unwrapped texture map sheet (4K). Zero 3D car perspective, zero wheels/ground.")
        parts.append("4. Leave every hole/cut-out inside the panels (light housings, vents, intakes, fuel filler, handles...) unpainted, keeping its exact outline - paint around it, never across it.")
        parts.append("5. Unless the design instructions above explicitly ask for an asymmetric design, mirror the design between the car's left and right sides "
                     "(same colours, stripes, graphics, number and sponsor positions) and keep hood/roof/bumpers symmetrical about the centreline - but never mirror lettering.")
        parts.append("6. Final check: no number, word, logo or figure may be sideways, upside-down or mirrored relative to the car part it sits on; "
                     "every opening is still open; both sides match.")

        return "\n".join(parts)

    def generate_livery(
        self,
        wireframe: Optional[Image.Image],
        references: List[Tuple[Image.Image, str]],
        prompt: str = "",
        system_instruction: Optional[str] = None,
        target_resolution: int = 4096,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Generates a 2D livery texture map matching the UV wireframe guide
        using Gemini 3.1 Flash Image / Gemini 3 Pro Image multimodal generation.
        Returns a 4K (4096 x 4096) RGBA PIL Image.
        """
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please enter your Google AI Studio API key in Settings."
            )

        effective_system_instruction = (
            system_instruction.strip()
            if system_instruction and system_instruction.strip()
            else self.build_default_system_instruction()
        )

        effective_prompt = prompt.strip() if prompt and prompt.strip() else (
            self.build_default_user_prompt(
                reference_roles=[role for _, role in references]
            )
        )

        # Build contents array with images followed by the full specification prompt
        contents = []

        # 1. Add Wireframe as Image 1 (primary structural canvas).
        #    Flattened onto white (transparent templates used to arrive as a solid black square) and padded to
        #    an aspect ratio the model supports so nothing gets stretched.
        prep = None
        aspect = "1:1"
        if wireframe is not None:
            prep = template_fit.prepare_for_ai(wireframe)
            aspect = prep.aspect_ratio
            contents.append(prep.image)

        # 2. Add Reference Images
        for ref_img, _ in references:
            img_copy = ref_img.copy()
            if img_copy.mode != "RGB":
                img_copy = img_copy.convert("RGB")
            if max(img_copy.size) > 1536:
                img_copy.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            contents.append(img_copy)

        # 3. Add User Prompt (+ the geometry rules that stop the model re-framing the sheet)
        if wireframe is not None:
            if "HOW TO READ IMAGE 1" not in effective_prompt:
                effective_prompt += "\n\n" + template_block(prep)
            effective_prompt += "\n\n" + LAYOUT_LOCK_RULES
            if "ARTWORK ORIENTATION (" not in effective_prompt:
                effective_prompt += "\n\n" + ORIENTATION_RULES
            if "OPEN HOLES / CUT-OUTS" not in effective_prompt:
                effective_prompt += "\n\n" + OPEN_HOLES_RULES
        if "LEFT/RIGHT SYMMETRY (" not in effective_prompt:
            effective_prompt += "\n\n" + SYMMETRY_RULES
        if "NO BAKED LIGHTING" not in effective_prompt:
            effective_prompt += "\n\n" + FLAT_COLOUR_RULES
        contents.append(effective_prompt)

        post = None
        if prep is not None:
            w, h = prep.original_size
            out_size = (target_resolution, max(1, round(target_resolution * h / w))) if w >= h else \
                (max(1, round(target_resolution * w / h)), target_resolution)
            post = lambda img: template_fit.restore_geometry(img.convert("RGBA"), prep, out_size)  # noqa: E731

        result = self._run_image_request(
            contents, effective_system_instruction, target_resolution, model_name, label="livery",
            aspect_ratio=aspect, postprocess=post,
        )
        self.last_notes = []
        if wireframe is not None and config.get("orientation_pass", True):
            result = self.fix_rotated_panels(result, wireframe, references,
                                             self._design_brief(prompt or effective_prompt), model_name)
        return result

    # ------------------------------------------------------------------
    # Orientation pass: panels the template says are stored rotated
    # ------------------------------------------------------------------

    @staticmethod
    def _design_brief(prompt: str) -> str:
        """The user's design text (without the sheet-level layout/orientation rules)."""
        m = re.search(r"--- DESIGN & STYLING INSTRUCTIONS ---\s*(.*?)(?:\n--- |\Z)", prompt or "", re.S)
        text = m.group(1) if m else (prompt or "")
        for marker in ("HOW TO READ IMAGE", "GEOMETRY LOCK", "ARTWORK ORIENTATION (", "OPEN HOLES / CUT-OUTS",
                       "LEFT/RIGHT SYMMETRY (", "NO BAKED LIGHTING"):
            i = text.find(marker)
            if i > 0:
                text = text[:i]
        return text.strip()[:3000]

    @staticmethod
    def build_panel_system_instruction() -> str:
        return (
            "You are an expert automotive livery painter for the racing simulator Automobilista 2. You repaint ONE flat "
            "UV panel cut out of a car's texture sheet. The panel is shown UPRIGHT: its top edge is the top of the car and "
            "its bottom edge is the ground side. Output a flat 2D texture of that panel only - never a 3D car or a scene."
        )

    @staticmethod
    def build_panel_prompt(brief: str, has_marker: bool, n_refs: int) -> str:
        lines = [
            "TASK: Repaint this single livery panel so that ALL of its graphics are UPRIGHT.",
            "",
            "- Image 1: the outline of the panel (light grey = the panel, dark = empty background), already turned "
            "upright: top edge = top of the car, bottom edge = ground side.",
            "- Image 2: the panel's current paint, same framing. It was painted the wrong way round - text and numbers "
            "may be upside-down, sideways or misplaced, and flames may point the wrong way.",
        ]
        if n_refs:
            lines.append(f"- Images 3-{2 + n_refs}: the design reference images (colours, logos, style only).")
        if brief:
            lines += ["", "DESIGN BRIEF:", brief]
        lines += [
            "",
            "Repaint the panel:",
            "1. Keep the same design as Image 2 and the brief: same base colour, same flame/graphic style and colours, "
            "same race number, driver name and sponsor logos, same level of detail.",
            "2. Every word, number and logo must be UPRIGHT and read left-to-right. Flames and similar graphics rise "
            "from the BOTTOM edge upward. Never mirror text.",
        ]
        if has_marker:
            lines.append("3. The black placeholder digits in Image 1 show where the race number goes: paint the race "
                         "number there, upright, about the same size.")
        lines += [
            "4. Paint only inside the light-grey panel of Image 1, filling it edge to edge and following its exact "
            "outline. Everything outside it stays exactly the dark background colour. Same framing as Image 1 - no "
            "zoom, shift, border, crop or rotation.",
            "5. Any opening inside the panel (light housing, vent, intake, handle... - a closed shape with no paint "
            "area inside) stays unpainted with its exact outline, as in Image 2. Paint around it, never over it.",
            "6. Output only this flat panel texture.",
            "",
            FLAT_COLOUR_RULES,
        ]
        return "\n".join(lines)

    def fix_rotated_panels(self, livery: Image.Image, wireframe: Image.Image,
                           references: List[Tuple[Image.Image, str]], brief: str = "",
                           model_name: Optional[str] = None) -> Image.Image:
        """
        Image models paint upright text reliably but struggle to paint it upside-down or sideways. For every panel
        whose placeholder text (read from the PSD) is rotated, cut the panel out, turn it upright, let the model
        repaint it upright, turn it back and paste it in. One extra request per rotated panel.
        """
        self.last_notes = getattr(self, "last_notes", [])
        analysis = template_fit.analyse_template(wireframe)
        todo = analysis.rotated_panels() if analysis is not None else []
        if not todo:
            return livery
        base = livery.convert("RGBA")
        base = template_fit.align(base, wireframe, report=template_fit.FitReport())  # masks must line up
        for idx, rot, _marker in todo:
            try:
                base = self._repaint_panel_upright(base, wireframe, analysis, idx, rot, references, brief, model_name)
                self.last_notes.append(f"P{idx + 1} repainted upright (stored {rot}\u00b0 rotated)")
            except Exception as e:  # keep the main result if a panel pass fails
                print(f"[AIEngine] orientation pass P{idx + 1} failed: {e}")
                self.last_notes.append(f"P{idx + 1} orientation pass failed: {str(e)[:80]}")
        return base

    def _repaint_panel_upright(self, base: Image.Image, wireframe: Image.Image, analysis, idx: int, rot: int,
                               references, brief: str, model_name: Optional[str]) -> Image.Image:
        W, H = base.size
        mask = Image.fromarray(analysis.panel_mask(idx).astype(np.uint8) * 255).resize((W, H), Image.Resampling.NEAREST)
        l, t, r, b = analysis.panels[idx]
        pad = 0.01
        box = (max(0, int((l - pad) * W)), max(0, int((t - pad) * H)),
               min(W, int(np.ceil((r + pad) * W))), min(H, int(np.ceil((b + pad) * H))))

        guide = template_fit.render_ai_guide(wireframe, analysis, only_panel=idx)
        if guide.size != (W, H):
            guide = guide.resize((W, H), Image.Resampling.LANCZOS)
        guide_crop = guide.crop(box)
        mask_crop = mask.crop(box)
        dark = Image.new("RGB", guide_crop.size, template_fit.GUIDE_BG)
        paint_crop = Image.composite(base.crop(box).convert("RGB"), dark, mask_crop)

        # PIL rotates anticlockwise: turning by the text's clockwise angle makes the panel upright
        up_guide = guide_crop.rotate(rot, expand=True)
        up_paint = paint_crop.rotate(rot, expand=True)
        prep = template_fit.prepare_for_ai(up_guide, as_guide=False)
        paint_img = template_fit.prepare_for_ai(up_paint, as_guide=False).image

        has_marker = bool((analysis.markers_px & analysis.panel_mask(idx)).any())
        contents: list = [prep.image, paint_img]
        for ref_img, _ in references:
            rc = ref_img.convert("RGB")
            rc.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            contents.append(rc)
        contents.append(self.build_panel_prompt(brief, has_marker, len(references)))

        up_size = up_guide.size
        post = lambda img: template_fit.restore_geometry(img.convert("RGBA"), prep, up_size)  # noqa: E731
        upright = self._run_image_request(
            contents, self.build_panel_system_instruction(), max(up_size), model_name,
            sizes=("2K", "1K", None), label=f"panel P{idx + 1} (upright pass)",
            aspect_ratio=prep.aspect_ratio, postprocess=post,
        )
        back = upright.rotate(-rot, expand=True)
        if back.size != guide_crop.size:
            back = back.resize(guide_crop.size, Image.Resampling.LANCZOS)
        back = template_fit.align(back, guide_crop, report=template_fit.FitReport())
        out = base.copy()
        out.paste(back.convert("RGBA"), box[:2], mask_crop)
        return out


    # ------------------------------------------------------------------
    # Shared request runner
    # ------------------------------------------------------------------

    _FATAL_ERROR_MARKERS = ("API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED", "UNAUTHENTICATED", "401")

    def _models_to_try(self, model_name: Optional[str]) -> List[str]:
        preferred = model_name or config.get("ai_model", DEFAULT_AI_MODEL)
        ordered = [preferred] + list(AVAILABLE_AI_MODELS) + [FALLBACK_AI_MODEL]
        seen = set()
        return [m for m in ordered if not (m in seen or seen.add(m))]

    @staticmethod
    def _extract_image(response) -> Tuple[Optional[Image.Image], str]:
        """Pulls the first image out of a Gemini response. Returns (image, text_or_reason)."""
        texts = []
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None):
            for part in candidates[0].content.parts or []:
                try:
                    img_obj = part.as_image()
                    if isinstance(img_obj, Image.Image):
                        return img_obj, ""
                    if getattr(img_obj, "image_bytes", None):
                        return Image.open(io.BytesIO(img_obj.image_bytes)), ""
                except Exception:
                    pass
                inline = getattr(part, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    try:
                        return Image.open(io.BytesIO(inline.data)), ""
                    except Exception:
                        pass
                t = getattr(part, "text", None)
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            if reason is not None and not texts:
                texts.append(f"finish_reason={reason}")
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            texts.append(f"blocked: {feedback.block_reason}")
        return None, " | ".join(texts)[:400]

    def _run_image_request(
        self,
        contents: list,
        system_instruction: str,
        target_resolution: int,
        model_name: Optional[str] = None,
        sizes: Tuple = ("4K", "2K", None),
        label: str = "livery",
        aspect_ratio: str = "1:1",
        postprocess=None,
    ) -> Image.Image:
        """
        Sends one multimodal request, walking through models and output sizes until an image comes back.
        Size fallback happens at request time (4K -> 2K -> model default), so models that reject 4K still work.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        errors: List[str] = []

        for current_model in self._models_to_try(model_name):
            for size_opt in sizes:
                try:
                    image_cfg = (
                        types.ImageConfig(aspect_ratio=aspect_ratio, image_size=size_opt)
                        if size_opt else types.ImageConfig(aspect_ratio=aspect_ratio)
                    )
                    gen_config = types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        system_instruction=system_instruction,
                        image_config=image_cfg,
                    )
                    print(f"[AIEngine] {label}: {current_model} @ {size_opt or 'default'}...")
                    response = client.models.generate_content(
                        model=current_model, contents=contents, config=gen_config
                    )
                    img, info = self._extract_image(response)
                    if img is not None:
                        print(f"[AIEngine] {label}: got {img.size} from {current_model}")
                        if postprocess is not None:
                            return postprocess(img)
                        return self._finalize_image(img, target_resolution)
                    errors.append(f"{current_model}: no image returned ({info or 'empty response'})")
                    break  # a text-only answer won't change with a different size; try next model
                except Exception as e:
                    msg = str(e)
                    print(f"[AIEngine] {current_model} @ {size_opt}: {msg[:200]}")
                    if any(marker in msg for marker in self._FATAL_ERROR_MARKERS):
                        raise RuntimeError(f"Gemini rejected the API key / permissions: {msg}") from e
                    errors.append(f"{current_model} @ {size_opt or 'default'}: {msg[:160]}")
                    if "NOT_FOUND" in msg or "404" in msg:
                        break  # model doesn't exist for this key; no point trying other sizes
                    continue

        detail = "\n".join(errors[-4:])
        raise RuntimeError(f"Failed to generate {label} with Gemini image models.\n{detail}")

    @staticmethod
    def _finalize_image(img: Image.Image, target_dim: int = 4096) -> Image.Image:
        """Converts to RGBA and scales to exact target resolution (4K) if needed."""
        rgba = img.convert("RGBA")
        if rgba.size != (target_dim, target_dim):
            print(f"[AIEngine] Resampling texture from {rgba.size} to {target_dim}x{target_dim} (4K)...")
            rgba = rgba.resize((target_dim, target_dim), resample=Image.Resampling.LANCZOS)
        return rgba

    @staticmethod
    def generate_mock_livery(
        wireframe: Optional[Image.Image],
        prompt: str,
        finish: str,
        team_name: str,
        driver: str,
        livery_id: int,
        target_resolution: int = 4096,
    ) -> Image.Image:
        """
        Generates a test/preview pattern (useful for offline testing or instant preview).
        """
        size = (target_resolution, target_resolution)
        base = Image.new("RGBA", size, (20, 24, 35, 255))
        draw = ImageDraw.Draw(base)

        # Draw motorsport racing stripes and styling
        stripe_colors = [
            (220, 50, 40, 255),
            (255, 180, 0, 255),
            (0, 140, 220, 255),
            (240, 240, 240, 255),
        ]

        # Dynamic angular stripes
        w, h = size
        for idx, col in enumerate(stripe_colors):
            offset = idx * (w // 10)
            draw.polygon(
                [
                    (offset, 0),
                    (offset + w // 8, 0),
                    (offset + w // 4, h),
                    (offset + w // 8, h),
                ],
                fill=col,
            )

        # Draw racing number circle and text
        center_x, center_y = w // 2, h // 2
        radius = w // 8
        draw.ellipse(
            [
                (center_x - radius, center_y - radius),
                (center_x + radius, center_y + radius),
            ],
            fill=(255, 255, 255, 255),
            outline=(20, 20, 20, 255),
            width=w // 200,
        )

        # Draw wireframe overlay if provided
        if wireframe is not None:
            wf_resized = wireframe.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
            # Blend wireframe lightly
            base = Image.blend(base, wf_resized, alpha=0.35)

        return base

    @staticmethod
    def build_refinement_system_instruction() -> str:
        """System instruction for iterative livery texture refinement."""
        return (
            "You are an expert automotive texture artist and livery designer for racing simulators (Automobilista 2).\n"
            "Your task is to refine, update, and modify an existing 2D livery texture map sheet based on user annotations and feedback notes.\n\n"
            "CRITICAL MANDATORY REFINEMENT RULES:\n"
            "1. IN-PLACE TEXTURE MODIFICATION: You are modifying the existing texture map sheet. Maintain overall design consistency, panel alignment, and colors where no changes were requested.\n"
            "2. APPLY USER MARKUP & NOTES: Look closely at the annotated markup image where the user has hand-drawn arrows, circles, lines, or notes indicating changes. Apply the requested changes (e.g. adjust sponsor logos, recolor stripes, change numbers, tweak accents) exactly as instructed.\n"
            "3. REMOVE MARKUP LINES: The output must be a clean, finished, professional 2D paint livery texture map sheet. Under NO circumstances should any sketch or annotation pen lines from the markup remain on the output texture.\n"
            "4. STRICT FLAT 2D UV MAP: The output must be an orthographic flat 2D unwrapped texture map sheet (1:1 square aspect ratio). Zero 3D vehicle renders, zero wheels, zero perspective scenery.\n"
            "5. PRINT-QUALITY GRAPHICS: Sharp, legible sponsor decals, racing numbers and text, with the same rich detail as the current livery.\n"
            "6. NO BAKED LIGHTING: Never add scene lighting (highlights, reflections, shine, bodywork shadows) - the game adds it. Keep the artwork's own depth and detail (flames, glows, gradients, lettering effects). "
            "If the current livery has baked-in reflections or sheen, remove them while applying the changes.\n"
            "7. ARTWORK ORIENTATION: Every number, text, logo and illustration you add or move must be upright ON THE CAR, not on the sheet - "
            "islands are unwrapped in different directions (side panels are often sideways or upside-down, left/right are mirror copies). "
            "Never mirror text. If existing graphics in the current livery are sideways, upside-down or mirrored relative to their car part, correct them.\n"
            "8. OPEN HOLES: Holes/cut-outs inside panels (light housings, vents, intakes, fuel filler, handles) stay unpainted with their exact shape. "
            "If the current livery has painted over one, clear it again.\n"
            "9. LEFT/RIGHT SYMMETRY: Unless the user explicitly asks for something one-sided or asymmetric, keep the left and right sides mirrored - "
            "a change requested on one side is applied to the matching place on the other side too (mirror the design, never the lettering)."
        )

    def refine_livery(
        self,
        current_livery: Image.Image,
        annotated_livery: Optional[Image.Image] = None,
        notes: str = "",
        wireframe: Optional[Image.Image] = None,
        references: Optional[List[Tuple[Image.Image, str]]] = None,
        finish: str = "",
        target_resolution: int = 4096,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """
        Refines an existing 2D livery texture map using Gemini multimodal generation
        guided by user hand-drawn annotations and refinement notes.
        Returns a 4K (4096 x 4096) RGBA PIL Image.
        """
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please enter your Google AI Studio API key in Settings."
            )

        system_instruction = self.build_refinement_system_instruction()

        # Build prompt & contents
        contents = []
        prompt_parts = [
            "TASK: Refine and update the 2D unwrapped UV livery texture map based on user annotations and notes.",
            "",
            "--- INPUT ASSETS ---",
        ]

        # 1. Wireframe (as the high-contrast layout map)
        wf_prep = None
        wf_idx = 0
        if wireframe is not None:
            wf_prep = template_fit.prepare_for_ai(wireframe)
            contents.append(wf_prep.image)
            wf_idx = len(contents)
            prompt_parts.append(f"- Image {wf_idx}: UV layout map of this car (light-grey panels = paint area, dark = background).")

        # 2. Current Livery (preserve full 4K resolution)
        curr_copy = current_livery.copy().convert("RGB")
        if max(curr_copy.size) > 4096:
            curr_copy.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        contents.append(curr_copy)
        curr_idx = len(contents)
        prompt_parts.append(f"- Image {curr_idx}: Current Livery Texture Sheet (Base design to be modified).")

        # 3. Annotated Markup (preserve full 4K resolution)
        if annotated_livery is not None:
            annot_copy = annotated_livery.copy().convert("RGB")
            if max(annot_copy.size) > 4096:
                annot_copy.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            contents.append(annot_copy)
            annot_idx = len(contents)
            prompt_parts.append(f"- Image {annot_idx}: Annotated Markup (User's hand-drawn notes, arrows, and circles showing exact changes).")

        # 4. Supplementary references
        if references:
            for ref_img, role in references:
                r_copy = ref_img.copy().convert("RGB")
                if max(r_copy.size) > 2048:
                    r_copy.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                contents.append(r_copy)
                prompt_parts.append(f"- Image {len(contents)}: Reference Image [{role}].")

        prompt_parts.extend([
            "",
            "--- USER REFINEMENT INSTRUCTIONS ---",
            notes.strip() if notes and notes.strip() else "Apply modifications as highlighted on the annotated markup image.",
            "",
            "--- FINAL EXECUTION INSTRUCTIONS ---",
            f"1. Apply the user's requested changes directly to Image {curr_idx}.",
            "2. Follow the visual guidance, arrows, and annotations from the markup image.",
            "3. Remove all annotation/sketch lines from the output; the final output must be clean automotive paint and decals.",
            "4. Keep all untouched panels and styling consistent with the current livery.",
            "5. Output ONLY the flat 2D unwrapped texture map sheet (4K). Zero 3D car perspective.",
            "",
            LAYOUT_LOCK_RULES.replace("Image 1", f"Image {curr_idx}"),
            "",
        ])
        if wf_prep is not None:
            prompt_parts.extend([
                template_block(wf_prep, wf_idx),
                f"If Image {curr_idx} does not match the panels of Image {wf_idx}, move its paint onto the correct "
                f"panels of Image {wf_idx} - the layout map wins.",
                "",
            ])
        prompt_parts.extend([
            ORIENTATION_RULES,
            "",
            OPEN_HOLES_RULES,
            "",
            SYMMETRY_RULES,
            "For refinements: a change marked or described on one side of the car is applied to the mirrored place on "
            "the other side as well, unless the notes say it is for one side only.",
            "",
            FLAT_COLOUR_RULES,
        ])

        full_prompt = "\n".join(prompt_parts)
        contents.append(full_prompt)

        return self._run_image_request(
            contents, system_instruction, target_resolution, model_name, label="refinement"
        )

    @staticmethod
    def refine_mock_livery(
        current_livery: Image.Image,
        annotated_livery: Optional[Image.Image] = None,
        notes: str = "",
        finish: str = "",
        iteration: int = 2,
        target_resolution: int = 4096,
    ) -> Image.Image:
        """
        Generates a visually refined test livery pattern for offline testing.
        """
        refined = current_livery.copy().resize((target_resolution, target_resolution), Image.Resampling.LANCZOS).convert("RGBA")
        draw = ImageDraw.Draw(refined)
        w, h = refined.size

        # Draw a dynamic revised accent banner
        banner_h = h // 16
        banner_y = (iteration * (h // 8)) % (h - banner_h)
        accent_color = (0, 220, 130, 230) if iteration % 2 == 0 else (255, 140, 0, 230)

        draw.rectangle(
            [0, banner_y, w, banner_y + banner_h],
            fill=accent_color,
        )

        # Draw revision label
        label_text = f"REV {iteration}: {notes[:30]}" if notes else f"REVISION {iteration}"
        center_x = w // 2
        draw.text(
            (center_x - 100, banner_y + banner_h // 4),
            label_text,
            fill=(255, 255, 255, 255),
        )

        return refined

    # ------------------------------------------------------------------
    # Wheel / rim textures
    # ------------------------------------------------------------------

    @staticmethod
    def build_wheel_system_instruction() -> str:
        return (
            "You are an expert automotive texture artist for the racing simulator Automobilista 2.\n"
            "You repaint WHEEL RIM texture maps (rim_diff). The first image is the car's existing rim texture/template: "
            "a flat UV texture sheet containing the rim face, spokes, lip, centre-lock nut and bolt details.\n\n"
            "MANDATORY RULES:\n"
            "1. Keep EXACTLY the same UV layout, shapes, positions and scale as the rim template. Do not move, add or remove spokes, holes or islands.\n"
            "2. Only change colours, finish and small decals (e.g. rim manufacturer logo, a pin-stripe on the lip).\n"
            "3. Preserve the template's shading, highlights, ambient occlusion and machined detail so the rim still reads as 3D in game.\n"
            "4. Keep transparent / black background areas of the template the same.\n"
            "5. Output a flat 1:1 texture sheet only. Never render a 3D wheel, tyre, car or scene."
        )

    def generate_wheel(
        self,
        rim_template: Image.Image,
        body_livery: Optional[Image.Image] = None,
        prompt: str = "",
        is_blur: bool = False,
        target_resolution: int = 1024,
        model_name: Optional[str] = None,
    ) -> Image.Image:
        """Repaints a rim texture (rim_diff / rim_blur_diff) to match the body livery and the user's prompt."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured. Please enter it in Settings.")

        contents: list = []
        tpl = template_fit.flatten_template(rim_template)
        contents.append(tpl)
        lines = [
            "TASK: Repaint this wheel rim texture sheet.",
            "- Image 1: the rim texture to repaint (keep its exact layout and shading).",
        ]
        if body_livery is not None:
            body = body_livery.convert("RGB")
            body.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            contents.append(body)
            lines.append("- Image 2: the car's body livery. Pick rim colours that complement it (main colour, accent colour or a neutral like gloss black / gunmetal / gold as appropriate).")
        if is_blur:
            lines.append("NOTE: this is the MOTION-BLUR version of the rim shown when the wheel spins fast. Keep it soft and blurred - just recolour it consistently.")
        lines.append("")
        lines.append("--- STYLE REQUEST ---")
        lines.append(prompt.strip() if prompt and prompt.strip() else "Colour the rims to match the livery in a clean, professional motorsport style.")
        lines.append("")
        lines.append("Output ONLY the repainted flat rim texture with an identical layout to Image 1.")
        contents.append("\n".join(lines))

        alpha = rim_template.convert("RGBA").getchannel("A")

        def post(img: Image.Image) -> Image.Image:
            out = self._finalize_image(img, target_resolution)
            out, _ = template_fit.lock_to_template(out, None, previous=tpl, do_clip=False)
            if alpha.getextrema()[0] < 255:  # keep the template's transparent areas
                out.putalpha(alpha.resize(out.size))
            return out

        return self._run_image_request(
            contents,
            self.build_wheel_system_instruction(),
            target_resolution,
            model_name,
            sizes=("2K", "1K", None),
            label="wheel rim",
            postprocess=post,
        )

    @staticmethod
    def generate_mock_wheel(
        rim_template: Image.Image, body_livery: Optional[Image.Image] = None, target_resolution: int = 1024
    ) -> Image.Image:
        """Offline fallback: tints the rim template with the livery's main colour."""
        from src.core.image_processor import ImageProcessor

        color = (40, 40, 44)
        if body_livery is not None:
            color = ImageProcessor.dominant_colors(body_livery, 1)[0]
        tinted = ImageProcessor.tint_texture(rim_template, color, 0.9)
        return tinted.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
