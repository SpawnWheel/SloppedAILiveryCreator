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

3. **AI Generation Engine (Nano Banana / Gemini Pro Image)**:
   - Multimodal generation combining UV wireframe layout, references, and prompt.
   - Non-deterministic re-roll support.
   - Offline test pattern generator for rapid preview without API charges.

4. **Image Processing & Alpha Reflectivity Injector**:
   - Injects custom Alpha-channel reflectivity masks into the livery DDS:
     - **Full Gloss**: Alpha 255 (High reflection)
     - **Semi-Gloss**: Alpha 210
     - **Satin**: Alpha 170
     - **Matte**: Alpha 80 (Diffuse finish)
     - **Metallic**: Alpha 200
     - **Chrome**: Alpha 255
   - Resizes textures to 4096×4096px (4K).
   - Generates 252×64px AMS2 UI menu thumbnails.

5. **DDS Conversion via Microsoft DirectXTex**:
   - Automatically downloads and bundles `texconv.exe`.
   - Converts textures to **BC3/DXT5** with full mipmaps (`-m 0`).
   - Converts thumbnails to **BC1/DXT1**.

6. **Safe, Non-Destructive AMS2 Installation & Backup**:
   - Auto-discovers `%USERPROFILE%/Documents/Automobilista 2/Vehicles/Textures/CustomLiveries/Overrides/`.
   - Creates timestamped backups (`<car>_backup_YYYYMMDD_HHMMSS.xml.bak`) before modifying files.
   - Real-time **Livery ID Collision Warning** badge.
   - Surgical XML merging without disturbing existing slots or comments.
   - One-click rollback dialog to restore previous XML backups.
   - **Standalone ZIP Export** for sharing on modding hubs (Overtake.gg).

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

1. **Section 1: Ingest Template**:
   - Drag & drop a UV wireframe image or ZIP from your AMS2 template collection, or paste a download URL.
2. **Section 2: Add References & Prompt**:
   - Upload 1–5 reference images and tag their roles (Color, Pattern, Decal).
   - Describe your desired livery, team colors, numbers, and sponsor placement in the prompt box.
3. **Section 3: Preview & Generate**:
   - Select your desired material finish (Matte, Satin, Gloss, etc.).
   - Click **⚡ Generate Livery** (or **⟳ Re-roll** to iterate).
4. **Section 4: Configure AMS2 Target**:
   - Select the car folder (e.g. `bmw_m4_gt3`) and Livery ID (e.g. `51`).
   - Enter your Team Name and Driver Name.
   - Notice the green/orange status badge indicating slot availability.
5. **Section 5: Deploy or Export**:
   - Click **💾 Install to AMS2 (Auto-Backup)** for instant in-game installation.
   - Or click **📦 Export as Shareable ZIP** to package a drop-in mod archive.
   - Use **↺ Restore Previous Backup** if you ever wish to revert changes.

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
