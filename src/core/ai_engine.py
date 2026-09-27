"""
AI Generation Engine integrating Google Gemini multimodal image generation
(Gemini 3.1 Flash Image / Gemini 3 Pro Image).
"""

import io
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw

from src.core import template_fit
from src.core.config import config
from src.utils.constants import AVAILABLE_AI_MODELS, DEFAULT_AI_MODEL, FALLBACK_AI_MODEL


LAYOUT_LOCK_RULES = (
    "GEOMETRY LOCK (most important): Image 1 is the exact canvas. Your output must overlay it pixel-for-pixel: "
    "same framing, same zoom, no added border or margin, no rotation, no mirroring, no cropping. Every UV island "
    "(panel) must stay at exactly the same position, size and outline as in Image 1 - only fill them with paint "
    "and graphics. Do not add, remove, merge or re-arrange panels. Paint must not spill outside the islands."
)

# The texture is wrapped onto a 3D car and lit by the game engine; baked-in light would be lit a second time.
FLAT_COLOUR_RULES = (
    "FLAT UNLIT COLOUR (very important): this is a diffuse/albedo texture - AMS2 adds all lighting, reflections, "
    "gloss and shadows itself when it wraps the texture on the car. So paint every area in flat, uniform, unlit "
    "colour: a panel that is blue must be the SAME blue value edge to edge. Do NOT add any lighting or photo "
    "effects: no shading, no highlights, no specular glints, no reflections, no environment/sky reflections, no "
    "shadows, no ambient occlusion, no vignetting, no gradients from light falloff, no clear-coat sheen, no metallic "
    "flake or grain, no noise, no paper/canvas texture, no 3D bevels or embossing on logos or numbers. "
    "Also ignore any lighting, shadows or reflections visible in the reference images - take only their colours "
    "and graphics. EXCEPTION: if the design instructions explicitly ask for a graphic effect (e.g. a colour fade / "
    "gradient, carbon-fibre weave, camo, halftone dots, brushed-metal look, a painted drop-shadow on a number), "
    "paint that graphic exactly as asked - it is part of the design, not lighting."
)


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
            "3. IN-PLACE WIREFRAME REPLACEMENT: In Image 1, each car body panel is an unrolled 2D UV island (hood/bonnet, roof, front bumper, rear bumper, left side doors/fenders, right side doors/fenders, mirrors, spoilers).\n"
            "   - In the exact positions, orientation, and silhouettes of these islands from Image 1, paint solid opaque automotive body paint, racing stripes, sponsor logos, and racing numbers.\n"
            "   - Completely eliminate and replace all wireframe/grid lines with smooth, solid paint and vinyl graphics. No wireframe lines should remain on the painted panels.\n"
            "4. TRANSFER GRAPHICS FROM REFERENCES: Reference images (Images 2..N) are strictly for transferring color scheme, logos, and styling onto the flat panels. Never draw or paste the 3D car or photo scene from reference images.\n"
            "5. CLEAN BACKGROUND: Outside of the UV panel islands, keep the background solid dark (#121418).\n"
            "6. VECTOR-CRISP GRAPHICS: Render sharp sponsor decals, racing numbers, and crisp racing stripes strictly as specified in the user design prompt and reference images.\n"
            "7. FLAT UNLIT COLOUR: The game engine does all lighting, gloss and reflections. Paint flat, uniform colours like vector artwork or a vinyl-wrap print file - "
            "no baked shading, highlights, reflections, shadows, sheen or surface texture. Only paint gradients or textured effects (fades, carbon weave, camo...) when the design instructions ask for them."
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
        parts.append("- Image 1: 2D UV Wireframe Template (Base layout to be painted in-place).")
        if reference_roles:
            for i, role in enumerate(reference_roles, start=2):
                parts.append(f"- Image {i}: Reference Image [{role}] (Extract colors, logos, and decals from here).")
        else:
            parts.append("- Images 2+: Reference images for color scheme, sponsor decals, and styling.")

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
        parts.append("1. Paint all UV islands (hood, roof, sides, bumpers, mirrors) with solid paint and decals strictly following the design instructions and reference images above.")
        parts.append("   Use flat, even, unlit colours (like a vinyl-wrap print file). No shading, highlights, reflections, shadows, sheen or surface texture - the game adds all lighting. Gradients/patterns only where the design asks for them.")
        parts.append("2. Strictly maintain the exact 1:1 square 2D layout and island coordinates from Image 1.")
        parts.append("3. Output ONLY the flat 2D unwrapped texture map sheet (4K). Zero 3D car perspective, zero wheels/ground.")

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
            effective_prompt += "\n\n" + LAYOUT_LOCK_RULES
        if "FLAT UNLIT COLOUR" not in effective_prompt:
            effective_prompt += "\n\n" + FLAT_COLOUR_RULES
        contents.append(effective_prompt)

        post = None
        if prep is not None:
            w, h = prep.original_size
            out_size = (target_resolution, max(1, round(target_resolution * h / w))) if w >= h else \
                (max(1, round(target_resolution * w / h)), target_resolution)
            post = lambda img: template_fit.restore_geometry(img.convert("RGBA"), prep, out_size)  # noqa: E731

        return self._run_image_request(
            contents, effective_system_instruction, target_resolution, model_name, label="livery",
            aspect_ratio=aspect, postprocess=post,
        )


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
            "5. VECTOR-CRISP GRAPHICS: Render sharp sponsor decals, racing numbers, and crisp racing stripes.\n"
            "6. FLAT UNLIT COLOUR: Keep all paint flat and uniform (the game adds lighting). Never add shading, highlights, reflections, shadows or surface texture. "
            "If the current livery already has baked-in lighting or sheen, flatten it out while applying the changes."
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

        # 1. Wireframe
        if wireframe is not None:
            contents.append(template_fit.prepare_for_ai(wireframe).image)
            prompt_parts.append(f"- Image {len(contents)}: 2D UV Wireframe Template (Structural UV layout).")

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
