# train_unet.py
import os, argparse, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------- Dataset ----------------
class PairSegDataset(Dataset):
    def __init__(self, img_dir, img_size=512, split="train", val_fraction=0.2, augment=True):
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.augment = augment and (split == "train")

        imgs = sorted(
            [p for p in self.img_dir.iterdir()
             if p.suffix.lower() in IMG_EXTS and not p.name.endswith("_mask.png")],
            key=lambda p: p.name
        )
        pairs = []
        for im in imgs:
            m = im.with_name(im.stem + "_mask.png")
            if m.exists():
                pairs.append((im, m))

        print(f"Found {len(pairs)} image-mask pairs")
        
        # simple deterministic split
        n = len(pairs)
        if n == 0:
            raise ValueError("No image-mask pairs found! Check your directory structure.")
            
        val_count = max(1, int(round(n * val_fraction)))
        val_indices = set(range(0, n, max(1, n // val_count)))
        
        if split == "val":
            self.samples = [pairs[i] for i in val_indices]
        else:  # train
            self.samples = [pairs[i] for i in range(n) if i not in val_indices]
            
        print(f"Split: {split}, samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _letterbox(self, img, size):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_r = cv2.resize(
            img, (nw, nh),
            interpolation=cv2.INTER_AREA if img.dtype != np.uint8 else cv2.INTER_LINEAR
        )
        canvas = np.zeros((size, size), np.uint8)
        top = (size - nh) // 2
        left = (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = img_r
        return canvas

    def __getitem__(self, i):
        im_path, m_path = self.samples[i]
        img = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"bad pair: {im_path} {m_path}")

        # binarize mask robustly
        mask = (mask > 127).astype(np.uint8) * 255

        # minimal augmentation - FIXED VERSION
        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                mask = np.ascontiguousarray(np.fliplr(mask))
            if random.random() < 0.2:
                ang = random.uniform(-5, 5)
                h, w = img.shape  # Fixed: img.shape is (height, width) for grayscale
                cx, cy = w / 2.0, h / 2.0  # center x, center y
                M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
                img = cv2.warpAffine(
                    img, M, (w, h),  # Fixed: (width, height) order for cv2.warpAffine
                    flags=cv2.INTER_LINEAR, borderValue=0
                )
                mask = cv2.warpAffine(
                    mask, M, (w, h),  # Fixed: (width, height) order for cv2.warpAffine
                    flags=cv2.INTER_NEAREST, borderValue=0
                )
            if random.random() < 0.3:
                a = random.uniform(0.9, 1.1)
                b = random.randint(-10, 10)
                img = np.clip(a * img + b, 0, 255).astype(np.uint8)

        # resize to square (letterbox)
        img = self._letterbox(img, self.img_size)
        mask = self._letterbox(mask, self.img_size)

        # to tensors
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # 1xHxW
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)  # 1xHxW
        return x, y

# ---------------- Model (U-Net) ----------------
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
        return self.head(d1)  # logits

# ---------------- Loss ----------------
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2, 3)) + self.eps
        return (1 - num / den).mean()

def bce_dice(logits, targets, w=0.5):
    b = F.binary_cross_entropy_with_logits(logits, targets)
    d = DiceLoss()(logits, targets)
    return w * b + (1 - w) * d

# ---------------- Train / Val ----------------
def run_epoch(model, loader, opt, device, train=True):
    model.train(train)
    tot = 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: opt.zero_grad()
            logits = model(x)
            loss = bce_dice(logits, y)
            if train:
                loss.backward(); opt.step()
            tot += loss.item() * x.size(0)
    return tot / len(loader.dataset)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="folder with originals and *_mask.png")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = PairSegDataset(args.images_dir, args.img_size, "train", args.val_frac, augment=True)
    val_ds   = PairSegDataset(args.images_dir, args.img_size, "val",   args.val_frac, augment=False)

    # CPU-friendly loader defaults
    cpu_only = (device.type == "cpu")
    num_workers = 0 if cpu_only else 2
    pin_memory = False if cpu_only else True

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory)

    model = UNet(in_channels=1, out_channels=1, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    for e in range(1, args.epochs + 1):
        tr = run_epoch(model, train_ld, opt, device, train=True)
        va = run_epoch(model, val_ld,   opt, device, train=False)
        print(f"Epoch {e:03d} | train {tr:.4f} | val {va:.4f}")
        torch.save({"epoch": e, "model": model.state_dict()}, Path(args.out) / "last.pt")
        if va < best:
            best = va
            torch.save({"epoch": e, "model": model.state_dict()}, Path(args.out) / "best.pt")

if __name__ == "__main__":
    main()