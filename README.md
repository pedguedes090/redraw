# 🎨 Redraw - Unified Image Inpainting Tool

Object removal via inpainting: provide an image and a mask, and the system fills in the masked area to seamlessly remove the object.

Combines **AOT-GAN** (natural-image inpainting) and **MangaInpainting** (manga/comic inpainting) into one tool with both a **Web UI** (draw-to-remove) and a **CLI**.

## Overview

| Feature | AOT-GAN | MangaInpainting |
|---|---|---|
| **Image type** | Natural photos (portraits, landscapes) | Manga / comics (grayscale) |
| **Use case** | Remove objects, restore photos | Remove speech bubbles, text, SFX |
| **Resolution** | 512×512 (auto-resized) | Arbitrary |
| **Input** | RGB image + mask | Grayscale image + structural lines + mask |

## Installation

### CPU or GPU

```bash
# Create a virtual environment (conda or venv)
python -m venv venv && source venv/bin/activate   # or: conda create -n redraw python=3.10 && conda activate redraw

# Install PyTorch — pick the right command for your system:
#   CPU-only:   pip install torch torchvision
#   CUDA 11.8:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   CUDA 12.1:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# (optional) Install test dependencies
pip install pytest
```

> **GPU is optional.** The tool works on CPU (slower but functional). If CUDA is available it will be used automatically.

### Model Checkpoints

Pre-trained weights are included in the repository under `checkpoints/`. If you need to download them manually:

**AOT-GAN:**
- [CelebA-HQ model](https://drive.google.com/drive/folders/1Zks5Hyb9WAEpupbTdBqsCafmb25yqsGJ) — portraits
- [Places2 model](https://drive.google.com/drive/folders/1bSOH-2nB3feFRyDEmiX81CEiWkghss3i) — general scenes
- Place the `.pt` file in `checkpoints/aot_gan/`

**MangaInpainting:**
- [MangaInpainting](https://drive.google.com/file/d/1YeVwaNfchLhy3lAA7jOLBP-W23onjy8S/view) → extract to `checkpoints/mangainpaintor/`
- [ScreenVAE](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view) → extract to `checkpoints/ScreenVAE/`

### Directory Layout

```
redraw/
├── app.py                    ← Web UI (Gradio)
├── inpaint.py                ← CLI tool
├── cli/                      ← `python -m cli.inpaint` entry point
├── setup_models.py           ← Auto-extract model zips
├── requirements.txt
├── tests/                    ← Automated tests
├── checkpoints/
│   ├── aot_gan/
│   │   └── G0000000.pt
│   ├── mangainpaintor/
│   │   ├── config.yml
│   │   ├── SemanticInpaintingModel_gen.pth
│   │   └── MangaInpaintingModel_gen.pth
│   └── ScreenVAE/
│       ├── latest_net_enc.pth
│       └── latest_net_dec.pth
├── aot_gan/                  ← AOT-GAN source
└── manga_inpainting/         ← MangaInpainting source
```

## Usage

### Web UI (recommended)

```bash
python app.py
```

Open `http://localhost:7860` in a browser:

1. **Upload** an image (or paste from clipboard).
2. Select the **Brush** tool and **paint white** over the area you want to remove.
3. Click **Inpaint**.
4. Download the result.

Tabs:
- **Natural Image (AOT-GAN)** — for photos, faces, landscapes, logos.
- **Manga Inpainting** — for manga/comics (grayscale).
- **Settings** — reload models, check status.

### CLI

```bash
# Natural image — remove an object
python -m cli.inpaint --mode aot \
    --image input.jpg --mask mask.png --output result.png \
    --seed 123

# Manga — remove text/speech bubbles
python -m cli.inpaint --mode manga \
    --image manga.png --mask mask.png --line lines.png \
    --output result.png --seed 42

# Batch processing (directory of images)
python -m cli.inpaint --mode aot \
    --image ./images/ --mask ./masks/ --output ./results/
```

Run `python -m cli.inpaint --help` for all options.

### Mask Format

- **White (255) = area to inpaint** (remove).
- **Black (0) = keep unchanged**.
- PNG format recommended.
- Dimensions must match the input image (auto-resized internally for the model).

### Running Tests

```bash
python -m pytest tests/ -v
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch: `pip install torch torchvision` |
| `AOT-GAN model not loaded` | Ensure `checkpoints/aot_gan/G0000000.pt` exists |
| `MangaInpainting model not loaded` | Ensure `checkpoints/mangainpaintor/` contains `.pth` files |
| `CUDA out of memory` | Reduce image size or switch to CPU mode |
| Gradio UI won't start | Check `pip install gradio>=4.0.0` and port 7860 is free |
| Output has artefacts/halos | Use a slightly larger mask covering the full object + edges |

## Credits

- **AOT-GAN**: [Zeng et al., TVCG 2023](https://arxiv.org/abs/2104.01431) — Aggregated Contextual Transformations for High-Resolution Image Inpainting
- **MangaInpainting**: [Xie et al., SIGGRAPH 2021](https://dl.acm.org/doi/10.1145/3450626.3459822) — Seamless Manga Inpainting with Semantics Awareness

## License

- AOT-GAN: Apache 2.0
- MangaInpainting: See manga_inpainting/LICENSE
