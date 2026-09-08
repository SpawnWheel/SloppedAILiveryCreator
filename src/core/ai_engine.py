"""
AI Generation Engine integrating Google Gemini multimodal image generation
(Gemini 3.1 Flash Image / Gemini 3 Pro Image).
"""

import io
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw

from src.core.config import config
from src.utils.constants import DEFAULT_AI_MODEL, FALLBACK_AI_MODEL


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
            "6. VECTOR-CRISP GRAPHICS: Render sharp sponsor decals, racing numbers, and crisp racing stripes strictly as specified in the user design prompt and reference images."
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
            if finish:
                parts.append(f"- Paint Finish: {finish}")
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

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        model = model_name or config.get("ai_model", DEFAULT_AI_MODEL)

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

        # 1. Add Wireframe as Image 1 (primary structural canvas)
        if wireframe is not None:
            wf_copy = wireframe.copy()
            if wf_copy.mode != "RGB":
                wf_copy = wf_copy.convert("RGB")
            # Preserve wireframe up to 4096 for 4K generation
            if max(wf_copy.size) > 4096:
                wf_copy.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            contents.append(wf_copy)

        # 2. Add Reference Images
        for ref_img, _ in references:
            img_copy = ref_img.copy()
            if img_copy.mode != "RGB":
                img_copy = img_copy.convert("RGB")
            if max(img_copy.size) > 1536:
                img_copy.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            contents.append(img_copy)

        # 3. Add User Prompt
        contents.append(effective_prompt)

        # Models with native multimodal image generation support
        models_to_try = [
            model,
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-2.5-flash-image",
            FALLBACK_AI_MODEL,
        ]
        # Deduplicate preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for current_model in models_to_try:
            try:
                print(f"[AIEngine] Generating 4K livery with model: {current_model}...")

                # Try 4K first, fallback to 2K/1:1 if model restricts 4K
                gen_config = None
                for size_opt in ["4K", "2K", None]:
                    try:
                        if size_opt:
                            gen_config = types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                system_instruction=effective_system_instruction,
                                image_config=types.ImageConfig(
                                    aspect_ratio="1:1",
                                    image_size=size_opt,
                                ),
                            )
                        else:
                            gen_config = types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                system_instruction=effective_system_instruction,
                                image_config=types.ImageConfig(aspect_ratio="1:1"),
                            )
                        break
                    except Exception:
                        continue

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=gen_config,
                )

                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        # 1. Try extracting via part.as_image()
                        try:
                            img_obj = part.as_image()
                            if hasattr(img_obj, "image_bytes") and img_obj.image_bytes:
                                pil_img = Image.open(io.BytesIO(img_obj.image_bytes))
                                print(f"[AIEngine] Successfully decoded image ({pil_img.size}) from {current_model}")
                                return self._finalize_image(pil_img, target_resolution)
                            if isinstance(img_obj, Image.Image):
                                return self._finalize_image(img_obj, target_resolution)
                        except Exception:
                            pass

                        # 2. Try extracting via part.inline_data
                        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                            try:
                                pil_img = Image.open(io.BytesIO(part.inline_data.data))
                                print(f"[AIEngine] Successfully decoded inline image ({pil_img.size}) from {current_model}")
                                return self._finalize_image(pil_img, target_resolution)
                            except Exception:
                                pass

            except Exception as e:
                print(f"[AIEngine] Model '{current_model}' failed: {e}")
                last_error = e
                continue

        raise RuntimeError(
            f"Failed to generate livery with Gemini image models. Error: {last_error}"
        )

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
            wf_resized = wireframe.resize(size, Image.LANCZOS).convert("RGBA")
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
            "5. VECTOR-CRISP GRAPHICS: Render sharp sponsor decals, racing numbers, and crisp racing stripes."
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

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        model = model_name or config.get("ai_model", DEFAULT_AI_MODEL)
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
            wf_copy = wireframe.copy().convert("RGB")
            if max(wf_copy.size) > 4096:
                wf_copy.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
            contents.append(wf_copy)
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
        ])

        full_prompt = "\n".join(prompt_parts)
        contents.append(full_prompt)

        models_to_try = [
            model,
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
            "gemini-2.5-flash-image",
            FALLBACK_AI_MODEL,
        ]
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for current_model in models_to_try:
            try:
                print(f"[AIEngine] Refining 4K livery with model: {current_model}...")

                # Try 4K first, fallback to 2K/1:1 if model restricts 4K
                gen_config = None
                for size_opt in ["4K", "2K", None]:
                    try:
                        if size_opt:
                            gen_config = types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                system_instruction=system_instruction,
                                image_config=types.ImageConfig(
                                    aspect_ratio="1:1",
                                    image_size=size_opt,
                                ),
                            )
                        else:
                            gen_config = types.GenerateContentConfig(
                                response_modalities=["IMAGE"],
                                system_instruction=system_instruction,
                                image_config=types.ImageConfig(aspect_ratio="1:1"),
                            )
                        break
                    except Exception:
                        continue

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=gen_config,
                )

                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        try:
                            img_obj = part.as_image()
                            if hasattr(img_obj, "image_bytes") and img_obj.image_bytes:
                                pil_img = Image.open(io.BytesIO(img_obj.image_bytes))
                                print(f"[AIEngine] Successfully refined image ({pil_img.size}) from {current_model}")
                                return self._finalize_image(pil_img, target_resolution)
                            if isinstance(img_obj, Image.Image):
                                return self._finalize_image(img_obj, target_resolution)
                        except Exception:
                            pass

                        if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                            try:
                                pil_img = Image.open(io.BytesIO(part.inline_data.data))
                                print(f"[AIEngine] Successfully refined inline image ({pil_img.size}) from {current_model}")
                                return self._finalize_image(pil_img, target_resolution)
                            except Exception:
                                pass

            except Exception as e:
                print(f"[AIEngine] Model '{current_model}' refinement failed: {e}")
                last_error = e
                continue

        raise RuntimeError(
            f"Failed to refine livery with Gemini image models. Error: {last_error}"
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
