import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ========================================================================
# MODEL DEFINITION (must match the one used during training)
# ========================================================================
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p if p is not None else k // 2,
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ELAN(nn.Module):
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
    def __init__(self, num_classes=6):
        super().__init__()
        self.stem = nn.Sequential(
            Conv(3, 32, 3, 1),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 1)
        )
        self.stage1 = nn.Sequential(Conv(64, 128, 3, 2), ELAN(128, 256, 64, 64))
        self.stage2 = nn.Sequential(nn.MaxPool2d(2, 2), ELAN(256, 512, 128, 128))
        self.stage3 = nn.Sequential(nn.MaxPool2d(2, 2), ELAN(512, 512, 256, 256))
        self.stage4 = nn.Sequential(nn.MaxPool2d(2, 2), ELAN(512, 512, 256, 256))
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
        x = self.classifier(x)
        return x

# ========================================================================
# INFERENCE FUNCTION
# ========================================================================
def predict(image_path, model_path, device, img_size=640):
    # Load model
    model = YOLOv7Classifier(num_classes=6)
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint types (best.pt or state_dict)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs.max().item() * 100

    # Map label to Singh Index grade (1–6)
    singh_grade = pred_class + 1
    print(f"🦴 Predicted Singh Index Grade: {singh_grade}")
    print(f"📊 Confidence: {confidence:.2f}%")
    return singh_grade, confidence

# ========================================================================
# MAIN ENTRY POINT
# ========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv7 Hip X-ray Inference")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model_path", type=str, default="runs/train/best.pt",
                        help="Path to trained model weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")

    print(f"\n{'='*60}")
    print(f"Running inference on: {args.image_path}")
    print(f"Using model: {args.model_path}")
    print(f"Device: {args.device.upper()}")
    print(f"{'='*60}\n")

    predict(args.image_path, args.model_path, args.device)
