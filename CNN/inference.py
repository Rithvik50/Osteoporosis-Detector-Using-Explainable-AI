import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ==============================================================
# UTF-8 Safe Print (prevents UnicodeEncodeError under Streamlit)
# ==============================================================
def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding, errors="replace").decode())


# ==============================================================
# MODEL DEFINITIONS (copied from training)
# ==============================================================
class Conv(nn.Module):
    """Standard convolution with activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELAN(nn.Module):
    """ELAN block from YOLOv7"""
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c1, c3, 1, 1)
        self.cv3 = nn.Sequential(Conv(c3, c4, 3, 1), Conv(c4, c4, 3, 1))
        self.cv4 = nn.Sequential(Conv(c3, c4, 3, 1), Conv(c4, c4, 3, 1))
        self.cv5 = Conv(c3 * 2 + c4 * 2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        return self.cv5(torch.cat([x1, x2, x3, x4], dim=1))


class YOLOv7Classifier(nn.Module):
    """YOLOv7-inspired classifier"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.stem = nn.Sequential(
            Conv(3, 32, 3, 1),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 1)
        )
        self.stage1 = nn.Sequential(
            Conv(64, 128, 3, 2),
            ELAN(128, 256, 64, 64)
        )
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ELAN(256, 512, 128, 128)
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ELAN(512, 512, 256, 256)
        )
        self.stage4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ELAN(512, 512, 256, 256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ==============================================================
# LOAD MODEL
# ==============================================================
def load_model(model_path, device):
    """
    Loads YOLOv7Classifier with weights from a .pt or .pth checkpoint.
    """
    safe_print("Loading CNN model...")
    checkpoint = torch.load(model_path, map_location=device)

    model = YOLOv7Classifier(num_classes=6)
    model = model.to(device)

    # Handle both state_dict-only and checkpoint dict formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    safe_print("Model loaded successfully.")
    return model


# ==============================================================
# IMAGE PREPROCESSING
# ==============================================================
def preprocess_image(image_path, img_size=640):
    safe_print("Preprocessing input image...")
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    safe_print("Image preprocessing complete.")
    return input_tensor

# ==============================================================
# GRAD-CAM IMPLEMENTATION
# ==============================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        cam /= cam.max()
        return cam
    
def overlay_heatmap(image_path, cam_map, output_filename="gradcam_result.jpg", alpha=0.5):
    output_dir = Path(__file__).resolve().parent / "gradcam_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_filename

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cam_resized = cv2.resize(cam_map, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img)

    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"🧭 Grad-CAM heatmap saved to: {output_path}")
    return str(output_path)

# ==============================================================
# PREDICTION
# ==============================================================
def predict(image_path, model_path, device, img_size=640, gradcam=True):
    model = load_model(model_path, device)
    input_tensor = preprocess_image(image_path, img_size).to(device)

    print("🚀 Running inference...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = probs.argmax().item()
        confidence = probs.max().item() * 100

    singh_grade = pred_class + 1
    print(f"🦴 Predicted Singh Index Grade: {singh_grade}")
    print(f"📊 Confidence: {confidence:.2f}%")

    # Grad-CAM visualization
    if gradcam:
        print("🧠 Generating Grad-CAM heatmap...")
        target_layer = model.stage4[-1].cv5  # Deepest conv block
        cam = GradCAM(model, target_layer)
        cam_map = cam.generate(input_tensor, pred_class)
        overlay_path = overlay_heatmap(image_path, cam_map, output_filename="gradcam_output.jpg")
        print(f"✅ Grad-CAM visualization saved to {overlay_path}")

    return singh_grade, confidence


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Singh Index CNN Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to cropped hip X-ray image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to CNN weights file (.pt or .pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"

    predict(args.image_path, args.model_path, args.device, args.img_size)
