"""
Setup script to extract pre-trained models from zip files and organize checkpoints.
Run this after downloading model files.
"""

import os
import sys
import zipfile
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_DIR = os.path.join(ROOT_DIR, "checkpoints")


def setup_directories():
    """Create checkpoint directories"""
    dirs = [
        os.path.join(CHECKPOINTS_DIR, "aot_gan"),
        os.path.join(CHECKPOINTS_DIR, "mangainpaintor"),
        os.path.join(CHECKPOINTS_DIR, "ScreenVAE"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  [OK] {d}")


def extract_zip(zip_path, target_dir):
    """Extract a zip file to target directory"""
    if not os.path.exists(zip_path):
        print(f"  [SKIP] {zip_path} not found")
        return False

    print(f"  Extracting {os.path.basename(zip_path)} -> {target_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(target_dir)
    print(f"  [OK] Extracted successfully")
    return True


def setup_manga_models():
    """Extract MangaInpainting models"""
    manga_zip = os.path.join(ROOT_DIR, "mangainpaintor.zip")
    manga_dir = os.path.join(CHECKPOINTS_DIR, "mangainpaintor")

    if os.path.exists(manga_zip):
        extract_zip(manga_zip, CHECKPOINTS_DIR)
        # Check if files are in a subdirectory
        for root, dirs, files in os.walk(CHECKPOINTS_DIR):
            for f in files:
                if f.endswith("_gen.pth") and root != manga_dir:
                    src = os.path.join(root, f)
                    dst = os.path.join(manga_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                        print(f"  Moved {f} -> {manga_dir}")
    else:
        print(f"  [SKIP] mangainpaintor.zip not found in {ROOT_DIR}")


def setup_screenvae():
    """Extract ScreenVAE model"""
    svae_zip = os.path.join(ROOT_DIR, "ScreenVAE.zip")
    svae_dir = os.path.join(CHECKPOINTS_DIR, "ScreenVAE")

    if os.path.exists(svae_zip):
        extract_zip(svae_zip, CHECKPOINTS_DIR)
        # Check if files are in a subdirectory
        for root, dirs, files in os.walk(CHECKPOINTS_DIR):
            for f in files:
                if f.startswith("latest_net_") and root != svae_dir:
                    src = os.path.join(root, f)
                    dst = os.path.join(svae_dir, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                        print(f"  Moved {f} -> {svae_dir}")
    else:
        print(f"  [SKIP] ScreenVAE.zip not found in {ROOT_DIR}")


def check_aot_models():
    """Check for AOT-GAN models in experiments folder"""
    aot_exp = os.path.join(ROOT_DIR, "aot_gan", "experiments")
    aot_ckpt = os.path.join(CHECKPOINTS_DIR, "aot_gan")

    found = False
    if os.path.isdir(aot_exp):
        for root, dirs, files in os.walk(aot_exp):
            for f in files:
                if f.startswith("G") and f.endswith(".pt"):
                    src = os.path.join(root, f)
                    dst = os.path.join(aot_ckpt, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        print(f"  Copied {f} -> {aot_ckpt}")
                    found = True

    if not found:
        print(f"  [INFO] No AOT-GAN models found.")
        print(f"         Download from: https://drive.google.com/drive/folders/1bSOH-2nB3feFRyDEmiX81CEiWkghss3i")
        print(f"         Place .pt file in: {aot_ckpt}")


def verify_setup():
    """Verify model files are in place"""
    print("\n" + "=" * 50)
    print("Verification:")
    print("=" * 50)

    # AOT-GAN
    aot_dir = os.path.join(CHECKPOINTS_DIR, "aot_gan")
    aot_models = [f for f in os.listdir(aot_dir) if f.endswith(".pt")] if os.path.isdir(aot_dir) else []
    if aot_models:
        print(f"  ✅ AOT-GAN: {len(aot_models)} model(s) found")
        for m in aot_models:
            print(f"     - {m}")
    else:
        print(f"  ❌ AOT-GAN: No models found in {aot_dir}")

    # MangaInpainting
    manga_dir = os.path.join(CHECKPOINTS_DIR, "mangainpaintor")
    manga_models = [f for f in os.listdir(manga_dir) if f.endswith(".pth")] if os.path.isdir(manga_dir) else []
    if manga_models:
        print(f"  ✅ MangaInpainting: {len(manga_models)} model(s) found")
        for m in manga_models:
            print(f"     - {m}")
    else:
        print(f"  ❌ MangaInpainting: No models found in {manga_dir}")

    # ScreenVAE
    svae_dir = os.path.join(CHECKPOINTS_DIR, "ScreenVAE")
    svae_models = [f for f in os.listdir(svae_dir) if f.endswith(".pth")] if os.path.isdir(svae_dir) else []
    if svae_models:
        print(f"  ✅ ScreenVAE: {len(svae_models)} model(s) found")
        for m in svae_models:
            print(f"     - {m}")
    else:
        print(f"  ❌ ScreenVAE: No models found in {svae_dir}")


def main():
    print("=" * 50)
    print("  Redraw - Model Setup Script")
    print("=" * 50)

    print("\n1. Creating directories...")
    setup_directories()

    print("\n2. Setting up MangaInpainting models...")
    setup_manga_models()

    print("\n3. Setting up ScreenVAE models...")
    setup_screenvae()

    print("\n4. Checking AOT-GAN models...")
    check_aot_models()

    verify_setup()

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("Run 'python app.py' to start the web UI")
    print("=" * 50)


if __name__ == "__main__":
    main()
