import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import argparse

# ----- Load Models -----
def load_models(unet_path, resnet_path, device="cpu"):
    unet = torch.load(unet_path, map_location=device)
    unet.eval()

    resnet = torch.load(resnet_path, map_location=device)
    resnet.eval()

    return unet, resnet

# ----- Preprocessing -----
transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),  # match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

def preprocess_xray(img_path):
    """Loads X-ray as PIL and tensor."""
    img = Image.open(img_path).convert("RGB")
    return img

def apply_unet_mask(unet, img_pil, device="cpu"):
    """Runs UNet on input X-ray and returns masked image."""
    img = np.array(img_pil)

    # UNet expects tensor input
    inp = transforms.ToTensor()(img).unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        mask_pred = unet(inp)  # shape (1,1,H,W) or (1,num_classes,H,W)
        mask_pred = torch.sigmoid(mask_pred)  # binary mask
        mask_np = (mask_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

    # apply mask
    masked = cv2.bitwise_and(img, img, mask=mask_np)
    masked_pil = Image.fromarray(masked)
    return masked_pil

def classify_resnet(resnet, masked_pil, device="cpu"):
    """Classify masked image into Singh Index grade."""
    inp = transform_resnet(masked_pil).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        logits = resnet(inp)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class + 1, probs.squeeze().cpu().tolist()  # Singh index 1–6

# ----- Full Pipeline -----
def xray_pipeline(unet, resnet, img_path, device="cpu"):
    img = preprocess_xray(img_path)
    masked_img = apply_unet_mask(unet, img, device)
    grade, probs = classify_resnet(resnet, masked_img, device)
    return grade, probs, masked_img

# ----- CLI -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Osteoporosis Pipeline: UNet + ResNet Singh Index Classifier")
    parser.add_argument("--img", type=str, required=True, help="Path to input X-ray image")
    parser.add_argument("--unet", type=str, required=True, help="Path to trained UNet .pth model")
    parser.add_argument("--resnet", type=str, required=True, help="Path to trained ResNet .pth model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()

    # Load models
    unet, resnet = load_models(args.unet, args.resnet, args.device)

    # Run pipeline
    grade, probs, masked_img = xray_pipeline(unet, resnet, args.img, args.device)

    print(f"\n✅ Predicted Singh Index Grade: {grade}")
    print("🔎 Class Probabilities:", probs)

    # Show masked X-ray
    masked_img.show()
