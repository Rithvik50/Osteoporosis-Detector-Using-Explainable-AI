import os
import sys
import subprocess
import platform
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
UNET_DIR = BASE_DIR / "U-Net"
CNN_DIR = BASE_DIR / "CNN"
PRED_DIR = UNET_DIR / "predictions"

UNET_MODEL = UNET_DIR / "checkpoints" / "best.pt"
CNN_MODEL = CNN_DIR / "data" / "runs" / "train" / "best.pt"

# ==========================================================
# INPUT VALIDATION
# ==========================================================
if len(sys.argv) < 2:
    print("[ERROR] No input image path provided.")
    print("Usage: python pipeline.py path/to/image.jpg")
    sys.exit(1)

input_image = Path(sys.argv[1]).resolve()
if not input_image.exists():
    print(f"[ERROR] Input image not found: {input_image}")
    sys.exit(1)

print(f"[INFO] Starting pipeline for image: {input_image}")

# ==========================================================
# STEP 1: Run UNET inference
# ==========================================================
print("[INFO] Running UNET model...")
try:
    subprocess.run([
        sys.executable, str(UNET_DIR / "infer_crop.py"),
        "--model_path", str(UNET_MODEL),
        "--image_path", str(input_image),
        "--save_overlay"
    ], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] UNET inference failed.")
    sys.exit(1)

print("[INFO] UNET inference completed.")

# ==========================================================
# STEP 2: Locate generated mask
# ==========================================================
basename = input_image.stem
mask_file = PRED_DIR / f"{basename}_predicted_mask.png"

if not mask_file.exists():
    print(f"[ERROR] Mask not found: {mask_file}")
    sys.exit(1)

print(f"[INFO] Found mask: {mask_file}")

# ==========================================================
# STEP 3: Multiply mask with original + crop for CNN
# ==========================================================
cropped_output = CNN_DIR / "data" / "infer" / f"{basename}_masked_cropped.png"

print("[INFO] Generating masked & cropped image...")
try:
    subprocess.run([
        sys.executable, str(UNET_DIR  / "mask_crop_multiply.py"),
        "--orig", str(input_image),
        "--mask", str(mask_file),
        "--out", str(cropped_output),
        "--pad", "10"
    ], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] Mask crop multiply failed.")
    sys.exit(1)

print(f"[INFO] Cropped masked image saved to {cropped_output}")

# ==========================================================
# STEP 4: Run CNN inference
# ==========================================================
print("[INFO] Running CNN inference on cropped image...")
try:
    subprocess.run([
        sys.executable, str(CNN_DIR / "data" / "inference.py"),
        "--image_path", str(cropped_output),
        "--model_path", str(CNN_MODEL)
    ], check=True)
except subprocess.CalledProcessError:
    print("[ERROR] CNN inference failed.")
    sys.exit(1)

print("[SUCCESS] CNN inference completed. Pipeline finished.")
