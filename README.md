# AMS2 AI Livery Studio

A desktop application in Python (PySide6) that automates the creation and installation of custom racing liveries for **Automobilista 2 (AMS2)** using Google Gemini multimodal AI (Nano Banana / Gemini Pro Image).

---

## 🌟 Key Features

1. **Wireframe & Template Ingestion**:
   - Accepts `.png`, `.jpg`, `.psd`, and `.zip` archives via drag-and-drop.
   - Direct URL fetching with offline local caching.
   - Auto-extracts the primary UV wireframe from ZIP templates.

2. **Multi-Reference Board & Prompt Studio**:
   - Add 1–5 visual reference images with custom role tagging:
     - *Color Palette / Theme*
     - *Graphic Pattern / Accents*
     - *Sponsor / Decal*
   - Text prompt input for numbers, driver names, and livery themes.

3. **Interactive Markup Canvas & Drawing Tools**:
   - Draw visual notes, arrows, boxes, and text callouts directly onto the livery texture.
   - Color presets, custom palette picker, and adjustable brush sizes.
   - Non-destructive transparent overlay with full Undo and Clear.

4. **Multi-Turn AI Refinement Loop**:
   - Send annotated drawings and feedback notes back to Gemini multimodal AI.
   - Iterative version tracking (`v1`, `v2`, `v3`...) with `< Prev` and `Next >` navigation.
   - Non-deterministic **⟳ Re-roll** for alternative design iterations.
   - **✅ Commit Livery** to lock in the approved revision before installation.

5. **Edit Already Created Liveries**:
   - Load existing custom liveries directly from AMS2 `Overrides/` car configurations.
   - Or open raw `.dds`, `.png`, `.jpg`, `.tga` files from anywhere on disk.
   - Annotate, iterate with AI refinement, and commit changes back into the game.

6. **Custom Preview Thumbnail Upload**:
   - Upload custom 252×64px in-game menu thumbnails (e.g. 3D showroom screenshots or track photos).
   - Seamless 1-click fallback to the auto-generated flatpack crop.

7. **Image Processing & Alpha Reflectivity Injector**:
   - Injects custom Alpha-channel reflectivity masks into the livery DDS:
     - **Full Gloss**: Alpha 255 (High reflection)
     - **Semi-Gloss**: Alpha 210
     - **Satin**: Alpha 170
     - **Matte**: Alpha 80 (Diffuse finish)
     - **Metallic**: Alpha 200
     - **Chrome**: Alpha 255
   - Resizes textures to 4096×4096px (4K).
   - Generates 252×64px AMS2 UI menu thumbnails.

8. **DDS Conversion via Microsoft DirectXTex**:
   - Automatically downloads and bundles `texconv.exe`.
   - Converts textures to **BC3/DXT5** with full mipmaps (`-m 0`).
   - Converts thumbnails to **BC1/DXT1**.
   - Decompresses and imports existing DDS liveries into full-fidelity RGBA images.

9. **Safe, Non-Destructive AMS2 Installation & Backup**:
   - Auto-discovers `%USERPROFILE%/Documents/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides/`.
   - Creates timestamped backups (`<car>_backup_YYYYMMDD_HHMMSS.xml.bak`) before modifying files.
   - Real-time **Livery ID Collision Warning** badge.
   - Surgical XML merging without disturbing existing slots or comments.
   - One-click rollback dialog to restore previous XML backups.
   - **Standalone ZIP Export** for sharing on modding hubs (Overtake.gg).

10. **Direct Pre-Made Livery Installation (No AI Needed)**:
   - Install pre-made `.dds`, `.png`, `.jpg`, `.tga`, or `.bmp` liveries sent by friends or downloaded from mod sites.
   - Preserves original Alpha channel reflectivity on DDS textures; applies chosen finish to PNG/JPG.
   - Optional custom menu preview thumbnail picker with auto flatpack fallback.
   - Automatically handles 4K resizing, mipmap generation, target folder creation, and XML override updates.

---

## 🚀 Getting Started

### 1. Requirements & Installation

```bash
# Clone or navigate to the repository
cd AIDesigner

# Install dependencies
pip install -r requirements.txt
```

Dependencies:
- `PySide6`
- `pyqtdarktheme`
- `google-genai`
- `Pillow`
- `psd-tools`
- `requests`
- `defusedxml`

### 2. Configure Google Gemini API Key

1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. You can either:
   - Set the environment variable: `set GEMINI_API_KEY=AIzaSy...`
   - Or open the app, click **File -> Settings / API Key...**, and paste your key.

### 3. Launch the Application

```bash
python main.py
```

---

## 🏎️ Workflow Guide

1. **Section 1: Ingest Template or Load Existing Livery**:
   - Drag & drop a UV wireframe image or ZIP from your AMS2 template collection.
   - *Or* click **📂 Load Livery to Edit...** to pull an existing car livery from your AMS2 Overrides or an image/DDS file!
2. **Section 2: Add References & Prompt**:
   - Upload reference images and describe your theme, driver name, and numbers.
3. **Section 3: Preview, Draw Annotations & Refine**:
   - Click **⚡ Generate Livery** to create the initial design.
   - Use the **Markup Tools** (Pen, Arrow, Box, Text Callout) to circle areas or point arrows directly on the texture.
   - Type feedback in **Refinement Notes** (e.g. *"Make stripes thinner, move decal to door as drawn"*).
   - Click **🔄 Refine with AI** to iterate as many times as you want across versions (`v1`, `v2`, `v3`...).
   - Use **⟳ Re-roll** if an iteration needs a fresh alternative take.
   - When satisfied, click **✅ Commit Livery**!
   - Optionally upload a custom screenshot/render in the **In-Game Menu Preview** box, or leave it on Auto (flatpack).
4. **Section 4: Configure AMS2 Target**:
   - Select the car folder (e.g. `bmw_m4_gt3`) and Livery ID (e.g. `51`).
   - Enter your Team Name and Driver Name.
5. **Section 5: Deploy or Export**:
   - Click **💾 Install to AMS2 (Auto-Backup)** for instant in-game installation of your generated design.
   - Or click **📥 Quick Install Livery File...** to directly install a friend's `.dds`, `.png`, or `.jpg` livery file into the selected car slot without using AI.
   - Or click **📦 Export as Shareable ZIP** to package a drop-in mod archive.

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
