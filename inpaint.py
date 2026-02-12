"""
Unified CLI Inpainting Tool
Supports both AOT-GAN (natural images) and MangaInpainting (manga/comics).

Usage examples:
    # Natural image inpainting with AOT-GAN
    python inpaint.py --mode aot --image input.jpg --mask mask.png --output result.png --model checkpoints/aot_gan/G0000000.pt

    # Manga inpainting
    python inpaint.py --mode manga --image manga.png --mask mask.png --line lines.png --output result.png --checkpoint checkpoints/mangainpaintor

    # Batch processing (folder of images)
    python inpaint.py --mode aot --image ./images/ --mask ./masks/ --output ./results/ --model checkpoints/aot_gan/G0000000.pt
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
from PIL import Image
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_image_paths(path):
    """Get image file paths from a file or directory"""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            paths.extend(glob.glob(os.path.join(path, ext)))
        return sorted(paths)
    return []


def run_aot_inpainting(args):
    """Run AOT-GAN inpainting"""
    sys.path.insert(0, os.path.join(ROOT_DIR, "aot_gan", "src"))

    from app import AOTGANInpainter

    inpainter = AOTGANInpainter(model_path=args.model)
    inpainter.load()

    image_paths = get_image_paths(args.image)
    mask_paths = get_image_paths(args.mask)

    if len(mask_paths) == 1 and len(image_paths) > 1:
        mask_paths = mask_paths * len(image_paths)

    os.makedirs(args.output if os.path.isdir(args.output) or not args.output.endswith(('.png', '.jpg'))
                else os.path.dirname(args.output) or '.', exist_ok=True)

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        print(f"[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}...")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8) * 255

        result = inpainter.inpaint(image, mask)

        # Determine output path
        if os.path.isdir(args.output) or len(image_paths) > 1:
            out_dir = args.output if os.path.isdir(args.output) else args.output
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(img_path))
        else:
            out_path = args.output

        Image.fromarray(result).save(out_path)
        print(f"  Saved to {out_path}")

    print(f"\nDone! Processed {len(image_paths)} images.")


def run_manga_inpainting(args):
    """Run MangaInpainting"""
    sys.path.insert(0, os.path.join(ROOT_DIR, "manga_inpainting"))

    from app import MangaInpainter

    inpainter = MangaInpainter(checkpoint_dir=args.checkpoint)
    inpainter.load()

    image_paths = get_image_paths(args.image)
    mask_paths = get_image_paths(args.mask)
    line_paths = get_image_paths(args.line) if args.line else [None] * len(image_paths)

    if len(mask_paths) == 1 and len(image_paths) > 1:
        mask_paths = mask_paths * len(image_paths)
    if len(line_paths) == 1 and len(image_paths) > 1:
        line_paths = line_paths * len(image_paths)

    os.makedirs(args.output if os.path.isdir(args.output) or not args.output.endswith(('.png', '.jpg'))
                else os.path.dirname(args.output) or '.', exist_ok=True)

    for i, (img_path, mask_path, line_path) in enumerate(zip(image_paths, mask_paths, line_paths)):
        print(f"[{i+1}/{len(image_paths)}] Processing {os.path.basename(img_path)}...")

        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8) * 255

        if line_path:
            line = np.array(Image.open(line_path).convert("L"))
        else:
            line = np.ones_like(image) * 255

        result = inpainter.inpaint(image, line, mask)

        # Determine output path
        if os.path.isdir(args.output) or len(image_paths) > 1:
            out_dir = args.output if os.path.isdir(args.output) else args.output
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(img_path))
        else:
            out_path = args.output

        Image.fromarray(result).save(out_path)
        print(f"  Saved to {out_path}")

    print(f"\nDone! Processed {len(image_paths)} images.")


def main():
    parser = argparse.ArgumentParser(
        description="Redraw - Unified Inpainting CLI (AOT-GAN + MangaInpainting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Natural image inpainting (AOT-GAN)
  python inpaint.py --mode aot --image photo.jpg --mask mask.png --output result.png --model checkpoints/aot_gan/G0000000.pt

  # Manga inpainting
  python inpaint.py --mode manga --image manga.png --mask mask.png --line lines.png --output result.png --checkpoint checkpoints/mangainpaintor

  # Batch processing
  python inpaint.py --mode aot --image ./images/ --mask ./masks/ --output ./results/ --model checkpoints/aot_gan/G0000000.pt
        """
    )

    parser.add_argument("--mode", type=str, required=True, choices=["aot", "manga"],
                        help="Inpainting mode: 'aot' for natural images, 'manga' for manga/comics")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to mask image or directory (white = inpaint area)")
    parser.add_argument("--line", type=str, default=None,
                        help="[Manga mode] Path to structural line image or directory")
    parser.add_argument("--output", type=str, default="./output",
                        help="Path to output image or directory (default: ./output)")
    parser.add_argument("--model", type=str, default=None,
                        help="[AOT mode] Path to pre-trained AOT-GAN model (.pt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[Manga mode] Path to MangaInpainting checkpoint directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible results")

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.mode == "aot":
        if args.model is None:
            # Auto-detect
            from app import find_aot_model
            args.model = find_aot_model()
            if args.model is None:
                print("Error: AOT-GAN model not found. Use --model to specify path.")
                sys.exit(1)
        run_aot_inpainting(args)

    elif args.mode == "manga":
        if args.checkpoint is None:
            from app import find_manga_checkpoint
            args.checkpoint = find_manga_checkpoint()
            if args.checkpoint is None:
                print("Error: MangaInpainting checkpoint not found. Use --checkpoint to specify path.")
                sys.exit(1)
        run_manga_inpainting(args)


if __name__ == "__main__":
    main()
