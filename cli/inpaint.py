"""
CLI entrypoint for inpainting.
Usage: python -m cli.inpaint --image input.jpg --mask mask.png --out output.png --seed 123
"""
import os
import sys

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from inpaint import main

if __name__ == "__main__":
    main()
