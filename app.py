"""
Unified Inpainting Application
Combines AOT-GAN (natural image inpainting) and MangaInpainting (manga-specific inpainting)
into a single Gradio web interface.
"""

import os
import sys
import importlib
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import gradio as gr

# ===== Path setup =====
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AOT_GAN_DIR = os.path.join(ROOT_DIR, "aot_gan", "src")
MANGA_DIR = os.path.join(ROOT_DIR, "manga_inpainting")

# ===== AOT-GAN Model Loader =====
class AOTGANInpainter:
    """AOT-GAN for natural image inpainting (photos, logos, faces, etc.)"""

    def __init__(self, model_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path

    def load(self, model_path=None):
        if model_path:
            self.model_path = model_path
        if self.model_path is None:
            raise ValueError("Model path not specified. Please provide a pre-trained model path.")

        # Add AOT-GAN source to path
        if AOT_GAN_DIR not in sys.path:
            sys.path.insert(0, AOT_GAN_DIR)

        from model.aotgan import InpaintGenerator

        class Args:
            block_num = 8
            rates = [1, 2, 4, 8]

        self.model = InpaintGenerator(Args())
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[AOT-GAN] Model loaded from {self.model_path}")

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint a natural image.

        Args:
            image: RGB image (H, W, 3), uint8
            mask: Binary mask (H, W), uint8 (255 = area to inpaint)

        Returns:
            Inpainted image (H, W, 3), uint8
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        orig_h, orig_w = image.shape[:2]

        # Resize to 512x512 for model
        img_resized = cv2.resize(image, (512, 512))
        mask_resized = cv2.resize(mask, (512, 512))

        # Prepare tensors
        img_tensor = (ToTensor()(img_resized) * 2.0 - 1.0).unsqueeze(0).to(self.device)
        mask_tensor = ToTensor()(mask_resized).unsqueeze(0).to(self.device)
        mask_tensor = (mask_tensor > 0.5).float()

        # Inpaint
        with torch.no_grad():
            masked_tensor = img_tensor * (1 - mask_tensor) + mask_tensor
            pred_tensor = self.model(masked_tensor, mask_tensor)
            comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

        # Post-process
        result = torch.clamp(comp_tensor[0], -1.0, 1.0)
        result = ((result + 1) / 2.0 * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Resize back
        result = cv2.resize(result, (orig_w, orig_h))
        return result


# ===== MangaInpainting Model Loader =====
class MangaInpainter:
    """MangaInpainting for manga/comic image inpainting with semantic awareness"""

    def __init__(self, checkpoint_dir=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint_dir = checkpoint_dir

    def load(self, checkpoint_dir=None):
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory not specified.")

        # Add MangaInpainting source to path
        if MANGA_DIR not in sys.path:
            sys.path.insert(0, MANGA_DIR)

        from src.config import Config
        from src.models import SemanticInpaintingModel, MangaInpaintingModel
        from src.svae import ScreenVAE
        from src.morphology import Dilation2d

        config_path = os.path.join(self.checkpoint_dir, "config.yml")
        if not os.path.exists(config_path):
            # Use default config
            example_config = os.path.join(MANGA_DIR, "config.yml.example")
            if os.path.exists(example_config):
                import shutil
                shutil.copy(example_config, config_path)

        self.config = Config(config_path)
        self.config._dict['DEVICE'] = self.device
        self.config._dict['GPU'] = [0] if torch.cuda.is_available() else []
        self.config._dict['PATH'] = self.checkpoint_dir

        self.semantic_model = SemanticInpaintingModel(self.config).to(self.device)
        self.manga_model = MangaInpaintingModel(self.config).to(self.device)

        # Load ScreenVAE
        svae_dir = os.path.join(ROOT_DIR, "checkpoints", "ScreenVAE")
        if os.path.exists(svae_dir):
            self.svae = ScreenVAE(save_dir=svae_dir).to(self.device)
        else:
            self.svae = ScreenVAE().to(self.device)

        self.semantic_model.load()
        self.manga_model.load()
        self.semantic_model.eval()
        self.manga_model.eval()
        self.svae.eval()

        self.dilate = Dilation2d(1, 1, 3, soft_max=False)
        print(f"[MangaInpainting] Models loaded from {self.checkpoint_dir}")

    def inpaint(self, image: np.ndarray, line: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint a manga image.

        Args:
            image: Grayscale manga image (H, W), uint8
            line: Structural line image (H, W), uint8
            mask: Binary mask (H, W), uint8 (255 = area to inpaint)

        Returns:
            Inpainted manga image (H, W), uint8
        """
        if self.semantic_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        orig_h, orig_w = image.shape[:2]

        # Prepare tensors
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0 * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
        line_tensor = torch.from_numpy(line.astype(np.float32) / 255.0 * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        # Pad to multiple of 128
        def pad_to_multiple(t, multiple=128):
            _, _, h, w = t.shape
            hp = (h + multiple - 1) // multiple * multiple
            wp = (w + multiple - 1) // multiple * multiple
            return torch.nn.functional.pad(t, (0, wp - w, 0, hp - h), mode='constant', value=0 if 'mask' in str(t.device) else 1)

        ih, iw = img_tensor.shape[2], img_tensor.shape[3]
        img_padded = self._npad(img_tensor, value=-1)
        line_padded = self._npad(line_tensor, value=-1)
        mask_padded = self._npad(mask_tensor, value=0)

        mask_padded = self.dilate(mask_padded, iterations=2)

        with torch.no_grad():
            manga_masked = img_padded * (1 - mask_padded) + mask_padded
            lines_masked = line_padded * (1 - mask_padded) + mask_padded

            screen_masked = self.svae(manga_masked, lines_masked, rep=True)

            screenl, linesl, masksl = self.semantic_model.test(screen_masked, lines_masked, mask_padded)

            screen = screenl[-1]
            lines = linesl[-1]

            outputs = self.manga_model(img_padded, torch.cat([screen, lines], 1), mask_padded)
            outputs_merged = (outputs * mask_padded) + (img_padded * (1 - mask_padded))

        # Post-process
        result = outputs_merged[0, 0, :ih, :iw]
        result = (result * 127.5 + 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        return result

    def _npad(self, im, pad=128, value=1):
        h, w = im.shape[-2:]
        hp = (h + pad - 1) // pad * pad
        wp = (w + pad - 1) // pad * pad
        return torch.nn.functional.pad(im, (0, wp - w, 0, hp - h), mode='constant', value=value)


# ===== Global singletons =====
aot_inpainter = AOTGANInpainter()
manga_inpainter = MangaInpainter()


# ===== Gradio Interface Functions =====

def find_aot_model():
    """Auto-detect AOT-GAN model path"""
    search_dirs = [
        os.path.join(ROOT_DIR, "checkpoints", "aot_gan"),
        os.path.join(ROOT_DIR, "aot_gan", "experiments"),
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.endswith(".pt"):
                        return os.path.join(root, f)
    return None


def find_manga_checkpoint():
    """Auto-detect MangaInpainting checkpoint directory"""
    search_dirs = [
        os.path.join(ROOT_DIR, "checkpoints", "mangainpaintor"),
        os.path.join(ROOT_DIR, "checkpoints", "manga_inpainting"),
        os.path.join(ROOT_DIR, "manga_inpainting", "checkpoints", "mangainpaintor"),
    ]
    for d in search_dirs:
        if os.path.isdir(d):
            # Check if it contains model files
            for f in os.listdir(d):
                if f.endswith(".pth"):
                    return d
    return None


def load_models():
    """Load both models on startup"""
    global aot_inpainter, manga_inpainter

    # AOT-GAN
    aot_path = find_aot_model()
    if aot_path:
        try:
            aot_inpainter.load(aot_path)
        except Exception as e:
            print(f"[Warning] Failed to load AOT-GAN: {e}")
    else:
        print("[Info] AOT-GAN model not found. Place .pt file in checkpoints/aot_gan/")

    # MangaInpainting
    manga_path = find_manga_checkpoint()
    if manga_path:
        try:
            manga_inpainter.load(manga_path)
        except Exception as e:
            print(f"[Warning] Failed to load MangaInpainting: {e}")
    else:
        print("[Info] MangaInpainting model not found. Place models in checkpoints/mangainpaintor/")


def extract_mask_from_editor(input_dict):
    """Extract binary mask from ImageEditor output.
    ImageEditor returns dict with 'background', 'layers', 'composite'.
    The drawn brush strokes are in layers as RGBA images - alpha channel shows where was drawn.
    """
    if input_dict is None:
        return None, None

    background = input_dict.get("background")
    layers = input_dict.get("layers", [])
    composite = input_dict.get("composite")

    if background is None:
        return None, None

    # Build mask from all drawn layers
    h, w = background.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, np.ndarray):
            if layer.ndim == 3 and layer.shape[2] == 4:
                # RGBA layer: alpha channel = drawn area
                alpha = layer[:, :, 3]
                mask = np.maximum(mask, alpha)
            elif layer.ndim == 3 and layer.shape[2] == 3:
                # RGB layer: white pixels = drawn area
                gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                mask = np.maximum(mask, gray)
            elif layer.ndim == 2:
                mask = np.maximum(mask, layer)

    # Binarize
    mask_binary = (mask > 10).astype(np.uint8) * 255
    return background, mask_binary


def inpaint_natural_draw(input_dict):
    """Inpaint natural images using AOT-GAN with drawn mask"""
    if aot_inpainter.model is None:
        return None, "AOT-GAN model not loaded! Place .pt file in checkpoints/aot_gan/"

    image, mask = extract_mask_from_editor(input_dict)

    if image is None:
        return None, "Please upload an image first (click the upload icon)"
    if mask is None or mask.max() == 0:
        return None, "Please use the brush to paint the area you want to remove"

    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    result = aot_inpainter.inpaint(image, mask)
    return result, "Inpainting complete!"


def inpaint_natural_files(image, mask):
    """Inpaint natural images from separate uploaded files"""
    if aot_inpainter.model is None:
        return None, "AOT-GAN model not loaded!"
    if image is None or mask is None:
        return None, "Please provide both image and mask"

    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    m = np.array(mask.convert("L"))
    m = (m > 127).astype(np.uint8) * 255

    result = aot_inpainter.inpaint(img, m)
    return Image.fromarray(result), "Inpainting complete!"


def inpaint_manga_draw(input_dict, line_img):
    """Inpaint manga images using drawn mask"""
    if manga_inpainter.semantic_model is None:
        return None, "MangaInpainting model not loaded!"

    image, mask = extract_mask_from_editor(input_dict)

    if image is None:
        return None, "Please upload a manga image first"
    if mask is None or mask.max() == 0:
        return None, "Please use the brush to paint the area you want to remove"

    # Convert to grayscale
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        img_gray = image

    if line_img is not None:
        line_arr = np.array(Image.fromarray(line_img).convert("L")) if len(line_img.shape) == 3 else line_img
    else:
        line_arr = np.ones_like(img_gray) * 255

    result = manga_inpainter.inpaint(img_gray, line_arr, mask)
    return result, "Manga inpainting complete!"


def inpaint_manga_files(image, line, mask):
    """Inpaint manga images from separate uploaded files"""
    if manga_inpainter.semantic_model is None:
        return None, "MangaInpainting model not loaded!"
    if image is None or mask is None:
        return None, "Please provide image and mask"

    img = np.array(image.convert("L"))

    if line is not None:
        line_img = np.array(line.convert("L"))
    else:
        line_img = np.ones_like(img) * 255

    m = np.array(mask.convert("L"))
    m = (m > 127).astype(np.uint8) * 255

    result = manga_inpainter.inpaint(img, line_img, m)
    return Image.fromarray(result, mode="L"), "Manga inpainting complete!"


# ===== Build Gradio UI =====
def create_ui():
    with gr.Blocks(title="Redraw - Unified Inpainting Tool", theme=gr.themes.Soft(),
                   css="""
                   .editor-container { min-height: 500px; }
                   .how-to { background: #f0f7ff; padding: 12px 16px; border-radius: 8px; margin-bottom: 8px; }
                   """) as app:
        gr.Markdown("""
        # 🎨 Redraw - Unified Image Inpainting Tool
        Combines **AOT-GAN** (natural image) and **MangaInpainting** (manga/comic) into one tool.
        """)

        with gr.Tabs():
            # ===== Tab 1: AOT-GAN Natural Image Inpainting =====
            with gr.Tab("🖼️ Natural Image (AOT-GAN)"):
                gr.Markdown("""
                ### AOT-GAN - Xóa vật thể trong ảnh tự nhiên
                Best for: **photos, faces, landscapes, logos, objects**
                """, elem_classes="how-to")

                with gr.Tabs():
                    # --- Sub-tab: Draw mask ---
                    with gr.Tab("✏️ Vẽ mask trực tiếp"):
                        gr.Markdown("""
                        **Hướng dẫn:** Upload ảnh → Dùng **bút vẽ (brush)** tô trắng lên vùng cần xóa → Nhấn **Inpaint**
                        """)
                        with gr.Row():
                            with gr.Column(scale=1):
                                aot_editor = gr.ImageEditor(
                                    label="Upload ảnh rồi tô vùng cần xóa",
                                    type="numpy",
                                    sources=("upload", "clipboard"),
                                    transforms=(),
                                    brush=gr.Brush(
                                        colors=["#FFFFFF"],
                                        color_mode="fixed",
                                        default_size=25,
                                    ),
                                    eraser=gr.Eraser(default_size=25),
                                    height=500,
                                )
                                with gr.Row():
                                    aot_paint_btn = gr.Button("🎨 Inpaint - Xóa vùng đã tô", variant="primary", size="lg")
                            with gr.Column(scale=1):
                                aot_paint_output = gr.Image(label="Kết quả", height=500)
                                aot_paint_status = gr.Textbox(label="Trạng thái")

                        aot_paint_btn.click(
                            fn=inpaint_natural_draw,
                            inputs=[aot_editor],
                            outputs=[aot_paint_output, aot_paint_status]
                        )

                    # --- Sub-tab: Upload files ---
                    with gr.Tab("📁 Upload ảnh + mask riêng"):
                        gr.Markdown("**Upload ảnh gốc và file mask (trắng = vùng xóa) riêng biệt**")
                        with gr.Row():
                            with gr.Column():
                                aot_image = gr.Image(label="Ảnh gốc", type="pil")
                                aot_mask = gr.Image(label="Mask (trắng = vùng cần xóa)", type="pil")
                                aot_btn = gr.Button("🎨 Inpaint", variant="primary", size="lg")
                            with gr.Column():
                                aot_output = gr.Image(label="Kết quả")
                                aot_file_status = gr.Textbox(label="Trạng thái")

                        aot_btn.click(
                            fn=inpaint_natural_files,
                            inputs=[aot_image, aot_mask],
                            outputs=[aot_output, aot_file_status]
                        )

            # ===== Tab 2: Manga Inpainting =====
            with gr.Tab("📖 Manga Inpainting"):
                gr.Markdown("""
                ### MangaInpainting - Xóa text/bóng thoại trong manga
                Best for: **manga, comics, black-and-white illustrations**
                """, elem_classes="how-to")

                with gr.Tabs():
                    # --- Sub-tab: Draw mask ---
                    with gr.Tab("✏️ Vẽ mask trực tiếp"):
                        gr.Markdown("""
                        **Hướng dẫn:** Upload ảnh manga → Tô trắng lên vùng text/bóng thoại cần xóa → Nhấn **Inpaint**
                        """)
                        with gr.Row():
                            with gr.Column(scale=1):
                                manga_editor = gr.ImageEditor(
                                    label="Upload ảnh manga rồi tô vùng cần xóa",
                                    type="numpy",
                                    sources=("upload", "clipboard"),
                                    transforms=(),
                                    brush=gr.Brush(
                                        colors=["#FFFFFF"],
                                        color_mode="fixed",
                                        default_size=30,
                                    ),
                                    eraser=gr.Eraser(default_size=25),
                                    height=500,
                                )
                                manga_line_draw = gr.Image(
                                    label="Structural Lines (tùy chọn - bỏ trống nếu không có)",
                                    type="numpy", height=150,
                                )
                                with gr.Row():
                                    manga_paint_btn = gr.Button("🎨 Inpaint Manga", variant="primary", size="lg")
                            with gr.Column(scale=1):
                                manga_paint_output = gr.Image(label="Kết quả", height=500)
                                manga_paint_status = gr.Textbox(label="Trạng thái")

                        manga_paint_btn.click(
                            fn=inpaint_manga_draw,
                            inputs=[manga_editor, manga_line_draw],
                            outputs=[manga_paint_output, manga_paint_status]
                        )

                    # --- Sub-tab: Upload files ---
                    with gr.Tab("📁 Upload ảnh + mask riêng"):
                        gr.Markdown("**Upload ảnh manga, file lines (tùy chọn), và file mask riêng biệt**")
                        with gr.Row():
                            with gr.Column():
                                manga_image = gr.Image(label="Ảnh Manga (grayscale)", type="pil")
                                manga_line = gr.Image(label="Structural Lines (tùy chọn)", type="pil")
                                manga_mask = gr.Image(label="Mask (trắng = vùng cần xóa)", type="pil")
                                manga_btn = gr.Button("🎨 Inpaint Manga", variant="primary", size="lg")
                            with gr.Column():
                                manga_output = gr.Image(label="Kết quả")
                                manga_file_status = gr.Textbox(label="Trạng thái")

                        manga_btn.click(
                            fn=inpaint_manga_files,
                            inputs=[manga_image, manga_line, manga_mask],
                            outputs=[manga_output, manga_file_status]
                        )

            # ===== Tab 3: Info & Settings =====
            with gr.Tab("⚙️ Cài đặt"):
                gr.Markdown("""
                ### Trạng thái Model & Hướng dẫn cài đặt

                #### AOT-GAN (Ảnh tự nhiên)
                1. Tải pre-trained models:
                   - [CelebA-HQ (chân dung)](https://drive.google.com/drive/folders/1Zks5Hyb9WAEpupbTdBqsCafmb25yqsGJ)
                   - [Places2 (phong cảnh)](https://drive.google.com/drive/folders/1bSOH-2nB3feFRyDEmiX81CEiWkghss3i)
                2. Đặt file `.pt` vào `checkpoints/aot_gan/`

                #### MangaInpainting (Manga/Comic)
                1. Tải pre-trained models:
                   - [MangaInpainting](https://drive.google.com/file/d/1YeVwaNfchLhy3lAA7jOLBP-W23onjy8S/view)
                   - [ScreenVAE](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view)
                2. Giải nén MangaInpainting vào `checkpoints/mangainpaintor/`
                3. Giải nén ScreenVAE vào `checkpoints/ScreenVAE/`

                #### Cấu trúc thư mục
                ```
                redraw/
                ├── checkpoints/
                │   ├── aot_gan/           ← Đặt AOT-GAN .pt model vào đây
                │   ├── mangainpaintor/    ← Giải nén MangaInpainting vào đây
                │   │   ├── config.yml
                │   │   ├── SemanticInpaintingModel_gen.pth
                │   │   └── MangaInpaintingModel_gen.pth
                │   └── ScreenVAE/         ← Giải nén ScreenVAE vào đây
                │       ├── latest_net_enc.pth
                │       └── latest_net_dec.pth
                ├── app.py                 ← Web UI này
                ├── inpaint.py             ← CLI tool
                └── ...
                ```
                """)

                with gr.Row():
                    reload_btn = gr.Button("🔄 Tải lại Models", variant="secondary")
                    reload_status = gr.Textbox(label="Trạng thái")

                def reload_models():
                    try:
                        load_models()
                        s_aot = "✅ Đã tải" if aot_inpainter.model else "❌ Chưa có"
                        s_manga = "✅ Đã tải" if manga_inpainter.semantic_model else "❌ Chưa có"
                        return f"AOT-GAN: {s_aot} | MangaInpainting: {s_manga}"
                    except Exception as e:
                        return f"Lỗi: {str(e)}"

                reload_btn.click(fn=reload_models, outputs=[reload_status])

    return app


if __name__ == "__main__":
    print("=" * 60)
    print("  Redraw - Unified Image Inpainting Tool")
    print("  AOT-GAN + MangaInpainting")
    print("=" * 60)

    # Create checkpoint directories
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints", "aot_gan"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints", "mangainpaintor"), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints", "ScreenVAE"), exist_ok=True)

    # Try to load models
    load_models()

    # Launch UI
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
