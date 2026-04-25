# train_brisc.py
import os
from glob import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

# Import the model
from model_brisc import AttentionUNet

# ==========================================
# 1. BRISC Dataset & Augmentations
# ==========================================
class BRISCSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read JPG RGB image and PNG Grayscale mask
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask  = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        
        # Binarize mask (tumor > 127)
        mask  = (mask > 127).astype(np.float32)

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]

        # Ensure mask has a channel dimension: (1, H, W)
        mask = mask.unsqueeze(0) if isinstance(mask, torch.Tensor) else torch.tensor(mask).unsqueeze(0)
        return image, mask

def get_transforms(img_size):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform

# ==========================================
# 2. Metrics & Loss Functions
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        preds   = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter   = (preds * targets).sum(dim=1)
        dice    = (2.0 * inter + self.smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

def calc_dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds   = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter   = (preds * targets).sum(dim=1)
    return ((2.0 * inter + smooth) / (preds.sum(dim=1) + targets.sum(dim=1) + smooth)).mean().item()

# ==========================================
# 3. Training Logic
# ==========================================
def train(data_dir):
    wandb.init(
        project="BRISC-Attention-UNet",
        config={
            "learning_rate": 2e-4,
            "epochs": 40,
            "batch_size": 32, # Splits perfectly across 2-4 GPUs
            "img_size": 256,
            "val_split": 0.15,
            "architecture": "Attention U-Net"
        }
    )
    config = wandb.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Splitting
    train_img_dir = os.path.join(data_dir, "segmentation_task/train/images")
    train_mask_dir = os.path.join(data_dir, "segmentation_task/train/masks")
    
    all_images = sorted(glob(os.path.join(train_img_dir, "*.jpg")))
    all_masks  = sorted(glob(os.path.join(train_mask_dir, "*.png")))
    
    assert len(all_images) == len(all_masks), "Image/mask count mismatch!"
    
    tr_imgs, vl_imgs, tr_msks, vl_msks = train_test_split(
        all_images, all_masks, test_size=config.val_split, random_state=42
    )

    train_tf, val_tf = get_transforms(config.img_size)
    
    train_loader = DataLoader(BRISCSegDataset(tr_imgs, tr_msks, transform=train_tf), 
                              batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(BRISCSegDataset(vl_imgs, vl_msks, transform=val_tf), 
                              batch_size=config.batch_size, shuffle=False, num_workers=8)

    # Multi-GPU Setup
    model = AttentionUNet(in_channels=3, num_classes=1).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    wandb.watch(model, log="all", log_freq=10)
    
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_val_dice = 0.0

    print("Starting training...")
    for epoch in range(1, config.epochs + 1):
        # TRAIN
        model.train()
        tr_loss, tr_dice = 0.0, 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            tr_loss += loss.item()
            tr_dice += calc_dice_score(logits.detach(), masks)
            
        tr_loss /= len(train_loader)
        tr_dice /= len(train_loader)
        scheduler.step()
        
        # VALIDATION
        model.eval()
        vl_loss, vl_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                
                vl_loss += criterion(logits, masks).item()
                vl_dice += calc_dice_score(logits, masks)
                
        vl_loss /= len(val_loader)
        vl_dice /= len(val_loader)

        wandb.log({
            "train_loss": tr_loss, "train_dice": tr_dice,
            "val_loss": vl_loss, "val_dice": vl_dice,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch
        })

        # Save Best Model
        tag = ""
        if vl_dice > best_val_dice:
            best_val_dice = vl_dice
            save_path = f"attention_unet_best.pth"
            
            # Extract state dict out of DataParallel to avoid loading errors later
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            tag = " [SAVED BEST]"

        print(f"Epoch {epoch:02d} | Tr Loss: {tr_loss:.4f} | Tr Dice: {tr_dice:.4f} | Vl Loss: {vl_loss:.4f} | Vl Dice: {vl_dice:.4f}{tag}")

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    # UPDATE THIS PATH TO MATCH YOUR SERVER'S UNZIPPED DIRECTORY
    DATA_DIR = "./data/brisc2025" 
    train(DATA_DIR)