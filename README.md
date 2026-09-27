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
   - Want a different take? Just press **⚡ Generate Livery** again — every run is a new design (earlier ones stay in the version history).
   - **✅ Commit Livery** to lock in the approved revision before installation.

5. **Edit Already Created Liveries**:
   - Load existing custom liveries directly from AMS2 `Overrides/` car configurations.
   - Or open raw `.dds`, `.png`, `.jpg`, `.tga` files from anywhere on disk.
   - Annotate, iterate with AI refinement, and commit changes back into the game.

6. **Custom Preview Image Upload**:
   - Upload a custom in-game menu preview (AMS2 uses **2048×600** — see any car's `_dist.xml`), e.g. a showroom screenshot.
   - Seamless 1-click fallback to the auto-generated flatpack crop.

7. **Image Processing & Alpha Reflectivity Injector**:
   - Writes the **Gloss** level into the BODY alpha channel (AMS2's "paint fx" mask):
     - **Gloss**: Alpha 255 (full effect of the base paint) — default
     - **Semi-Gloss**: Alpha 210
     - **Satin**: Alpha 170
     - **Matte**: Alpha 80
   - **Gloss vs Base Livery:** the Base Livery (section 4) picks the paint *shader* — normal, Matte, Chrome,
     Metallic… whatever the car offers — plus every texture you don't replace. Gloss only controls how strongly
     that base paint's shine shows. For chrome/metallic, choose that base livery and leave Gloss on *Gloss*.
   - Gloss is never sent to the AI: the AI paints flat, unlit colour and the game supplies all lighting.
   - Resizes textures to 4096×4096px (4K).
   - Generates 2048×600px AMS2 UI menu preview images.

8. **DDS Conversion via Microsoft DirectXTex**:
   - Automatically downloads and bundles `texconv.exe`.
   - Converts textures to **BC3/DXT5** with full mipmaps (`-m 0`).
   - Converts preview images and wheel rims to **BC1/DXT1**, rim blur textures to **BC3/DXT5**.
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

11. **🛞 Wheels / Rims** (new in 1.1):
   - Reads each car's `<car>_dist.xml` to list its real **Base Liveries** (e.g. `Default`, `Matte`, `Chrome`, or
     paint names like `ViperWhite`) and which textures it lets you override.
   - Load the rim texture(s) from the car's template pack, then **Quick recolour** (keeps shading), **✨ Match Livery
     with AI**, or **🖌 Paint by hand**. Installed as `WHEELDIFF` / `WHEELBLURDIFF` next to `body.dds`.

12. **🖌 Paint Editor** (new in 1.1) — a Photoshop-lite window for finishing AI results by hand:
   - Your template shown over the paint (opacity + blend mode) so you always know where the panels are. Never exported.
   - Layers: paint layers, **PNG decals / sponsors** (drag & drop, Ctrl+V), live **text** for numbers & names.
   - Tools: move/scale/rotate/flip, brush, eraser, line, rectangle, ellipse, polygon, **paint bucket that uses the
     template outlines as walls** (click a panel to paint it), eyedropper.
   - Remove white/black background from JPG logos, colourise a logo, undo/redo, zoom/pan.
   - **Apply to Livery** adds a new version (`vN (Hand-painted)`); re-opening that version restores all layers.
     Save/open layered projects as `.ams2paint`.

13. **📋 Copy for Other AI** — use ChatGPT, Gemini web or any other image AI instead of the built-in Gemini call:
   - The button next to **⚡ Generate Livery** (or *File > Copy Prompt + Images for Other AI*, `Ctrl+Shift+C`) writes a
     folder to `Documents/AMS2 Livery Studio/Prompt Packs/<team>_<date>/` with `prompt.txt` (system rules + your brief +
     geometry-lock and flat-colour rules in one pasteable prompt), `01_template.png` (flattened/padded exactly as the
     app sends it) and `02_…png` reference images in the order the prompt names them.
   - Prompt **and** images go to the clipboard — press `Ctrl+V` in the AI site's message box (Chrome/Edge). If a site
     only takes one kind, use **Copy Images** / **Copy Prompt** in the dialog, drag the thumbnails into the browser, or
     right-click a thumbnail > *Copy this image*.
   - **📥 Import AI Result…** loads the downloaded image as a new version (`vN (Other AI)`), removes the padding and
     auto-aligns it to the template. Change the folder with the `prompt_pack_dir` config key.

---

## 🛞 How wheels work in AMS2

A livery override only replaces the textures listed in the car XML. Anything not listed comes from the
`BASELIVERY` — that's why custom liveries normally keep the stock rims. Most cars' `_dist.xml` list:

| Texture name | What it is | Size / format |
|---|---|---|
| `WHEELDIFF` | rim colour (`rim_diff`) | 1024², DXT1 + mips |
| `WHEELBLURDIFF` | rim when spinning fast (`rim_blur_diff`) | 1024², DXT5 + mips |
| `WHEELSPEC`, `WHEELBLURSPEC` | rim reflections (not edited by the studio) | 1024² |

So: either pick a **Base Livery** whose rims you like (section 4), or paint `WHEELDIFF`/`WHEELBLURDIFF`
in the **Wheels / Rims** tab. You need the rim textures from the car's template pack as the starting point.

## 🖌 Paint Editor shortcuts

`V` move · `B` brush · `E` eraser · `L` line · `U` rectangle · `O` ellipse · `P` polygon · `G` bucket · `I` eyedropper ·
`T` text · `H`/Space pan · wheel zoom · `[` `]` brush size · `X` swap colours · `Ctrl+Z`/`Ctrl+Y` undo/redo ·
`Ctrl+0` fit · `Ctrl+J` duplicate layer · `Del` delete layer · arrows nudge a decal (Shift = 10px)

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
- `numpy`, `opencv-python-headless` (fast paint-bucket fill; the editor falls back to a slower fill without OpenCV)

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
   - Press **⚡ Generate Livery** again any time for a completely new alternative.
   - Click **🖌 Paint Editor** to add sponsor PNGs, numbers, lines or paint whole panels by hand.
   - When satisfied, click **✅ Commit Livery**!
   - Optionally upload a custom screenshot/render in the **In-Game Menu Preview** box, or leave it on Auto (flatpack).
4. **Section 4: Configure AMS2 Target**:
   - Select the car folder (e.g. `bmw_m4_gt3`), Livery ID (e.g. `51`) and **Base Livery** (read from the car's `_dist.xml`).
   - Enter your Team Name and Driver Name.
5. **Section 5: Deploy or Export**:
   - Click **💾 Install to AMS2 (Auto-Backup)** for instant in-game installation of your generated design.
   - Or click **📥 Quick Install Livery File...** to directly install a friend's `.dds`, `.png`, or `.jpg` livery file into the selected car slot without using AI.
   - Or click **📦 Export as Shareable ZIP** to package a drop-in mod archive (includes custom rims, and a README with
     the XML block to paste if the receiver already has their own `<car>.xml`).

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 📝 Changelog

### Unreleased
- **📋 Copy for Other AI**: export prompt + template + reference images to a folder and the clipboard, and import the
  other AI's result back as an aligned new version.

### 1.1.0
- **Wheels / Rims tab** with quick recolour, AI match and hand painting; installs `WHEELDIFF` / `WHEELBLURDIFF`.
- **Paint Editor** (layers, decals, text, template-aware bucket fill, project files).
- **Base Livery picker** populated from each car's `_dist.xml` (previously always `Default`, which some cars such as the
  Viper ACR don't have).
- **Preview image is now 2048×600** as documented by AMS2 (was 252×64).
- **Safer XML editing**: Reiza's own `_dist.xml` files are not always valid XML. The old writer then rebuilt the car XML from
  scratch and could silently drop your other liveries. Edits are now spliced into the original text — comments, other
  liveries and textures the studio doesn't manage are kept, and new car XMLs start as a copy of `_dist.xml` like AMS2 asks.
- AI: real 4K → 2K → default size fallback per request, stops immediately on an invalid API key, clearer error messages,
  preview model IDs added to the fallback list.
- Loading an installed livery also restores its rims and base livery, and no longer doubles the "#51" suffix in the team name.
- ZIP export README now points to the game install folder and includes the XML block to paste.
- Compact actions panel, warning before quitting during a running install/generation.
