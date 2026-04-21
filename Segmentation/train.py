# train.py
import os
import glob
import re
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

# Import the model from our other script
from model import UNet2D

# ==========================================
# 2.5D HDF5 DataLoader 
# ==========================================
class BraTS25DDataset(Dataset):
    # ADDED: allowed_vols parameter
    def __init__(self, data_dir, allowed_vols=None):
        self.data_dir = data_dir
        self.file_paths = glob.glob(os.path.join(data_dir, "*.h5"))
        self.volumes = {}
        self.samples = []
        
        pattern = re.compile(r'volume_(\d+)_slice_(\d+)')
        for path in self.file_paths:
            filename = os.path.basename(path)
            match = pattern.search(filename)
            if match:
                vol_id = int(match.group(1))
                
                # ADDED: Skip this file if it belongs to a patient not in our split
                if allowed_vols is not None and vol_id not in allowed_vols:
                    continue
                
                slice_idx = int(match.group(2))
                if vol_id not in self.volumes:
                    self.volumes[vol_id] = {}
                self.volumes[vol_id][slice_idx] = path
                self.samples.append((vol_id, slice_idx))

    def __len__(self):
        return len(self.samples)

    def _load_h5(self, filepath):       
        with h5py.File(filepath, 'r') as h5f:
            image = h5f['image'][:]
            mask = h5f['mask'][:]   
            
            # Convert image from (H, W, 4) to (4, H, W)
            if len(image.shape) == 3 and image.shape[-1] == 4:
                image = np.transpose(image, (2, 0, 1))
                
            # Convert mask from (H, W, 3) to (3, H, W)
            if len(mask.shape) == 3 and mask.shape[-1] == 3:
                mask = np.transpose(mask, (2, 0, 1))
                
        return image, mask

    def __getitem__(self, idx):
        vol_id, center_slice_idx = self.samples[idx]
        vol_dict = self.volumes[vol_id]
        
        indices_to_load = [center_slice_idx - 1, center_slice_idx, center_slice_idx + 1]
        images_25d = []
        center_mask = None
        
        for s_idx in indices_to_load:
            target_idx = s_idx if s_idx in vol_dict else center_slice_idx
            filepath = vol_dict[target_idx]
            
            img, msk = self._load_h5(filepath)
            images_25d.append(img)
            
            if s_idx == center_slice_idx:
                center_mask = msk
                
        image_tensor = torch.tensor(np.concatenate(images_25d, axis=0), dtype=torch.float32)
        mask_tensor = torch.tensor(center_mask, dtype=torch.float32)
        
        return image_tensor, mask_tensor

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1.0 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
        return bce_loss + dice_loss

# ==========================================
# Training Setup and Loop
# ==========================================
def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================
    # DATA SPLIT LOGIC
    # ==========================================
    data_dir = './data/brats_unzipped/BraTS2020_training_data/content/data'
    
    # Find all unique patient volumes
    all_paths = glob.glob(os.path.join(data_dir, "*.h5"))
    all_vols = set()
    for p in all_paths:
        match = re.search(r'volume_(\d+)_slice', p)
        if match: 
            all_vols.add(int(match.group(1)))
            
    all_vols = list(all_vols)
    np.random.seed(42) # Consistent shuffle
    np.random.shuffle(all_vols)
    
    # 80/20 Train/Val Split
    split_idx = int(len(all_vols) - 15)
    train_vols = all_vols[:split_idx]
    val_vols = all_vols[split_idx:]
    
    print(f"Training on {len(train_vols)} volumes. Validating on {len(val_vols)} volumes.")

    train_dataset = BraTS25DDataset(data_dir, allowed_vols=train_vols)
    val_dataset = BraTS25DDataset(data_dir, allowed_vols=val_vols)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # ==========================================
    # MODEL & MULTI-GPU LOGIC
    # ==========================================
    model = UNet2D(in_channels=config.in_channels, out_channels=config.out_classes).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    wandb.watch(model, log="all", log_freq=10)

    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # ==========================================
    # TRAINING & VALIDATION LOOP
    # ==========================================
    print("Starting training...")
    for epoch in range(config.epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                intersection = (preds * masks).sum()
                dice_score = (2. * intersection + 1e-5) / (preds.sum() + masks.sum() + 1e-5)

            wandb.log({
                "train_batch_loss": loss.item(), 
                "train_batch_dice": dice_score.item(),
                "epoch": epoch + 1
            })

        avg_train_loss = train_epoch_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval() # Set model to evaluation mode
        val_epoch_loss = 0.0
        val_epoch_dice = 0.0
        
        with torch.no_grad(): # Disable gradient calculation for testing
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_epoch_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                intersection = (preds * masks).sum()
                dice = (2. * intersection + 1e-5) / (preds.sum() + masks.sum() + 1e-5)
                val_epoch_dice += dice.item()
                
        avg_val_loss = val_epoch_loss / len(val_loader)
        avg_val_dice = val_epoch_dice / len(val_loader)

        # Log epoch-level metrics
        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": avg_val_loss,
            "epoch_val_dice": avg_val_dice,
            "epoch": epoch + 1
        })
        
        print(f"--- Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} ---")
        
        # Save model checkpoint
        checkpoint_path = f"./segmentation/checkpoints/unet_25d_epoch_{epoch+1}.pth"
        
        # NOTE: When saving a DataParallel model, you must use model.module.state_dict()
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    wandb.init(
        project="BraTS-25D-Segmentation",
        config={
            "learning_rate": 1e-4,
            "epochs": 10,
            "batch_size": 16, # Note: PyTorch will automatically split this batch across GPUs
            "architecture": "2.5D U-Net",
            "dataset": "BraTS 2020",
            "in_channels": 12, 
            "out_classes": 3
        }
    )
    config = wandb.config
    train(config)