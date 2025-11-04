import os
import sys
import glob
import subprocess
import shutil
from pathlib import Path

def run_command(command, cwd=None):
    """Run a shell command safely and print progress."""
    print(f"[INFO] Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        sys.exit(1)

def ensure_dir(path: Path):
    """Ensure directory exists, create if missing."""
    if not path.exists():
        print(f"[WARN] Directory {path} not found. Creating...")
        path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[INFO] Directory verified: {path}")

def main():
    # --- Check input argument ---
    if len(sys.argv) < 2:
        print("[ERROR] Usage: python pipeline.py <image_path>")
        sys.exit(1)

    input_image = Path(sys.argv[1]).resolve()
    if not input_image.exists():
        print(f"[ERROR] Input image not found: {input_image}")
        sys.exit(1)

    # --- Define directories ---
    BASE_DIR = Path(__file__).resolve().parent
    unet_dir = BASE_DIR / "U-Net"
    cnn_dir = BASE_DIR / "CNN" / "data"

    unet_model = unet_dir / "checkpoints" / "best.pt"
    cnn_model = cnn_dir / "runs" / "train" / "best.pt"
    predictions_dir = unet_dir / "predictions"
    infer_dir = cnn_dir / "infer"

    # --- Verify and create directories ---
    for d in [unet_dir, cnn_dir, predictions_dir, infer_dir]:
        ensure_dir(d)

    # --- Check model files ---
    for model in [unet_model, cnn_model]:
        if not model.exists():
            print(f"[ERROR] Model file not found: {model}")
            sys.exit(1)

    print(f"[INFO] Starting pipeline for image: {input_image}")

    # --- Run UNet inference ---
    print("[INFO] Running UNet model...")
    run_command([
        "python",
        str(unet_dir / "infer_crop.py"),
        "--model_path", str(unet_model),
        "--image_path", str(input_image),
        "--save_overlay"
    ])
    print("[INFO] UNet inference completed.")

    # --- Get latest predicted mask ---
    print("[INFO] Searching for latest cropped mask...")
    predicted_masks = sorted(
        predictions_dir.glob("*_predicted_mask.png"),
        key=os.path.getmtime,
        reverse=True
    )

    if not predicted_masks:
        print(f"[ERROR] No cropped image found in {predictions_dir}")
        sys.exit(1)

    cropped_image = predicted_masks[0]
    print(f"[INFO] Latest cropped image found: {cropped_image.name}")

    # --- Copy to CNN infer folder ---
    dest_path = infer_dir / cropped_image.name
    shutil.copy2(cropped_image, dest_path)
    print(f"[INFO] Cropped image copied to {dest_path}")

    # --- Run CNN inference ---
    print("[INFO] Running CNN inference on cropped image...")
    run_command([
        "python",
        str(cnn_dir / "inference.py"),
        "--image_path", str(dest_path),
        "--model_path", str(cnn_model)
    ])

    print("[SUCCESS] CNN inference completed. Pipeline finished successfully.")

if __name__ == "__main__":
    main()
