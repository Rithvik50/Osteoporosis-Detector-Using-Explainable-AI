import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET CLASS
# ============================================================================
class HipXrayDataset(Dataset):
    """Dataset for Hip X-ray Classification"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from class_1 to class_6 folders
        for class_idx in range(1, 7):
            class_folder = self.root_dir / f'class_{class_idx}'
            if class_folder.exists():
                for img_path in class_folder.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.images.append(str(img_path))
                        self.labels.append(class_idx - 1)  # 0-indexed
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# YOLOV7 BACKBONE + CLASSIFICATION HEAD
# ============================================================================
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
        self.cv3 = nn.Sequential(
            Conv(c3, c4, 3, 1),
            Conv(c4, c4, 3, 1)
        )
        self.cv4 = nn.Sequential(
            Conv(c3, c4, 3, 1),
            Conv(c4, c4, 3, 1)
        )
        self.cv5 = Conv(c3 * 2 + c4 * 2, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        return self.cv5(torch.cat([x1, x2, x3, x4], dim=1))

class YOLOv7Classifier(nn.Module):
    """YOLOv7-inspired architecture for classification"""
    
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            Conv(3, 32, 3, 1),
            Conv(32, 64, 3, 2),
            Conv(64, 64, 3, 1)
        )
        
        # Backbone stages
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
        
        # Classification head
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

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Val]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return running_loss / len(loader), 100. * correct / total

# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
def main():
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv7 for Hip X-ray Classification')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--train-dir', type=str, default='dataset/train', help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='dataset/val', help='Validation data directory')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'test_dir': 'dataset/test',
        'num_classes': 6,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 0.0005,
        'img_size': args.img_size,
        'device': args.device if torch.cuda.is_available() else 'cpu',
        'save_dir': args.save_dir
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = HipXrayDataset(config['train_dir'], transform=train_transform)
    val_dataset = HipXrayDataset(config['val_dir'], transform=val_transform)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print(f"\n{'='*50}")
    print(f"Initializing YOLOv7 Classifier for Singh Index")
    print(f"{'='*50}")
    model = YOLOv7Classifier(num_classes=config['num_classes']).to(config['device'])
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting Training on {config['device'].upper()}")
    print(f"{'='*50}\n")
    
    for epoch in range(config['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device'], epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config['device'], epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(config['save_dir'], 'best.pt'))
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(config['save_dir'], f'epoch_{epoch+1}.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'final.pt'))
    
    # Plot training curves
    plot_training_curves(history, config['save_dir'])
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Models saved in: {config['save_dir']}")
    print(f"{'='*50}\n")

def plot_training_curves(history, save_dir):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    print(f"Training curves saved to {save_dir}/training_curves.png")

if __name__ == '__main__':
    main()