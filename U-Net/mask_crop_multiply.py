# mask_crop_multiply.py
# Usage:
# python mask_crop_multiply.py --orig <path_to_original> --mask <path_to_mask> --out <output_path> [--pad 10]
#
# Function:
#   1. Loads the original X-ray and predicted mask.
#   2. Multiplies them (mask applied to original).
#   3. Finds bounding box of non-zero region.
#   4. Crops and saves final image for CNN.

import argparse
import os
from PIL import Image
import numpy as np

def find_bbox(mask_arr):
    ys, xs = np.where(mask_arr > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pad", type=int, default=10)
    args = parser.parse_args()

    orig = Image.open(args.orig).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    orig_arr = np.array(orig, dtype=np.uint8)
    mask_arr = np.array(mask, dtype=np.uint8)
    mask_arr = (mask_arr > 0).astype(np.uint8)  # binarize 0/1

    multiplied = orig_arr * mask_arr[:, :, None]

    bbox = find_bbox(mask_arr)
    if bbox is None:
        print(f"[WARN] No mask region found. Saving full masked image.")
        Image.fromarray(multiplied).save(args.out)
        return

    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - args.pad)
    y_min = max(0, y_min - args.pad)
    x_max = min(orig_arr.shape[1] - 1, x_max + args.pad)
    y_max = min(orig_arr.shape[0] - 1, y_max + args.pad)

    cropped = multiplied[y_min:y_max + 1, x_min:x_max + 1, :]
    cropped_img = Image.fromarray(cropped)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cropped_img.save(args.out)
    print(f"[INFO] Cropped masked image saved to {args.out}")

if __name__ == "__main__":
    main()