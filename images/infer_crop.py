import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path

# Copy the UNet model definition from training script
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.enc1 = conv_block(in_channels, base); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2);    self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4);self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base * 4, base * 8);self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)

def letterbox_resize(img, size):
    """Resize image to square while maintaining aspect ratio"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((size, size), np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img_r
    
    return canvas, scale, (top, left, nh, nw)

def unletterbox_mask(mask, original_shape, scale, crop_info):
    """Convert letterboxed mask back to original image dimensions"""
    top, left, nh, nw = crop_info
    oh, ow = original_shape
    
    # Extract the resized portion from letterbox
    resized_mask = mask[top:top+nh, left:left+nw]
    
    # Resize back to original dimensions
    original_mask = cv2.resize(resized_mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    
    return original_mask

def predict_image(model, image_path, img_size=512, device='cpu', threshold=0.5):
    """Predict mask for a single image"""
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = img.shape
    print(f"Original image shape: {original_shape}")
    
    # Preprocess (same as training)
    img_resized, scale, crop_info = letterbox_resize(img, img_size)
    
    # Convert to tensor
    x = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0  # 1x1xHxW
    x = x.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()
    
    # Convert back to numpy
    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
    probs_np = probs.squeeze().cpu().numpy()
    
    # Resize both mask and probability map back to original dimensions
    final_mask = unletterbox_mask(pred_mask_np, original_shape, scale, crop_info)
    final_probs = unletterbox_mask((probs_np * 255).astype(np.uint8), original_shape, scale, crop_info) / 255.0
    
    return final_mask, final_probs

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained U-Net model")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--image_path", required=True, help="Path to input image")
    parser.add_argument("--output_dir", default="predictions", help="Directory to save predictions")
    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument("--save_overlay", action="store_true", help="Save overlay of mask on original image")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(in_channels=1, out_channels=1, base=32)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run prediction
    image_path = Path(args.image_path)
    print(f"Processing: {image_path}")
    
    try:
        pred_mask, prob_map = predict_image(model, image_path, args.img_size, device, args.threshold)
        
        # Save binary mask
        mask_name = f"{image_path.stem}_predicted_mask.png"
        cv2.imwrite(str(output_dir / mask_name), pred_mask)
        
        # Save probability map (0-255)
        prob_name = f"{image_path.stem}_probability_map.png"
        prob_img = (prob_map * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / prob_name), prob_img)
        
        print(f"Saved binary mask: {output_dir / mask_name}")
        print(f"Saved probability map: {output_dir / prob_name}")
        
        # Optional: Create overlay
        if args.save_overlay:
            original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            # Convert grayscale to 3-channel
            overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            
            # Add red mask overlay
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 2] = pred_mask  # Red channel
            
            # Blend
            alpha = 0.3
            blended = cv2.addWeighted(overlay, 1-alpha, mask_colored, alpha, 0)
            
            overlay_name = f"{image_path.stem}_overlay.png"
            cv2.imwrite(str(output_dir / overlay_name), blended)
            print(f"Saved overlay: {output_dir / overlay_name}")
        
        # Print some statistics
        mask_area = np.sum(pred_mask > 0)
        total_area = pred_mask.size
        coverage = (mask_area / total_area) * 100
        
        print(f"Mask coverage: {coverage:.2f}%")
        print(f"Max probability: {np.max(prob_map):.3f}")
        print(f"Mean probability in masked region: {np.mean(prob_map[pred_mask > 0]):.3f}" if mask_area > 0 else "No masked region detected")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()