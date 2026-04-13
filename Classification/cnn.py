"""
CNN Image Classification Testing Functions

Authors: Cole Kerkemeyer, Phillip Kornberg
Date: 4.1.26
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models, datasets
import pandas as pd
from PIL import Image

# Setting Global Paths and CNN Specification
csvPath = 'Data/manifest.csv'
dataRoot = 'Data'
batchSize = 64
epocs = 25
lr = 1e-4
numWorkers = 4
savePath = 'brain_tumor_model.pth'
valSplit = 0.2

# Loading Brain Tumor Images and Labels from CSV
class BrainTumorDataset(Dataset):

    def __init__(self, df, dataRoot, transform = None):
        """
        Function to Initialize Brain Tumor Dataset and Create Label Mapping

        : param DataFrame df - DataFrame containing image paths and labels
        : param str dataRoot - Root directory where images are stored
        : param transform - Optional torchvision transforms applied to images

        : return None
        """

        # Storing Data Frame in Root Directory
        self.df = df.reset_index(drop = True)
        self.data_root = dataRoot
        self.transform = transform

        # Converting Tumor Names to Integer Labels
        self.classes = sorted(self.df['tumor_label'].unique().tolist())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        """
        Function to Get Total Number of Samples in Dataset

        : return int length - Number of images in dataset
        """
         
        return len(self.df)

    def __getitem__(self, idx):
        """
        Function to Load and Transform a Single Image and Its Label

        : param int idx - Index of sample to retrieve

        : return tensor image - Transformed image tensor
        : return int label - Integer class label for image
        """

        row = self.df.iloc[idx]

        # Setting Relative Path
        rel = row['relative_path'].replace('\\', os.sep)
        rel = os.path.join(*rel.split(os.sep)[1:])

        # Setting Image Path and Transformning
        imgPath = os.path.join(self.data_root, rel)
        image = Image.open(imgPath).convert('RGB')
        label = self.class_to_idx[row['tumor_label']]

        if self.transform:
            image = self.transform(image)

        return image, label

def printSummary(name, df):
    """
    Prints a dataset summary including total sample count and class distribution.

    : param str name - Name of the dataset split (e.g., 'train', 'val')
    : param DataFrame df - Pandas DataFrame containing at least a 'tumor_label' column

    : return None - Prints summary statistics to console
    """

    classes = sorted(df['tumor_label'].unique().tolist())
    print(f"[{name}] {len(df)} samples")
    print("  " + " | ".join(f"{c}: {(df['tumor_label'] == c).sum()}" for c in classes))

def getValues():
    """
    Function to Compute Mean and Standard Deviation of Training Dataset

    : return tuple(torch.Tensor, torch.Tensor)
        mean - Tensor of shape (3,) representing RGB mean values
        std - Tensor of shape (3,) representing RGB standard deviation values
    """
    dataset = datasets.ImageFolder(
        root='Data/train',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std  = torch.zeros(3)
    total = 0

    for images, _ in loader:
        mean += images.mean(dim=[0, 2, 3]) * images.size(0)
        std  += images.std(dim=[0, 2, 3])  * images.size(0)
        total += images.size(0)

    mean /= total
    std  /= total

    return mean, std

# Finding Mean and Std
mean, std = getValues()

# Training Transformations and Adjusting Images (Resize, Brighten, Normalize) 
trainTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Setting Validation Transformations
evalTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def runEpoch(model, loader, criterion, optimizer, device, training = True):
    """
    Function to Run One Full Epoch of Training or Validation

    : param model - CNN model being trained or evaluated
    : param DataLoader loader - DataLoader for dataset split
    : param loss criterion - Loss function (e.g., CrossEntropyLoss)
    : param optimizer - Optimizer used for training
    : param torch.device device - CPU or GPU device
    : param bool training - Whether to run in training mode or evaluation mode

    : return float avgLoss - Average loss over epoch
    : return float accuracy - Accuracy over epoch
    """

    # Settng Model Mode and Initializing Variables
    model.train() if training else model.eval()
    totalLoss, correct = 0, 0

    # Enabling Gradient Descent
    with torch.set_grad_enabled(training):

        # Moving Data to GPU and Running
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            # Calculating Total Loss and Correct Labeling
            totalLoss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    return totalLoss / len(loader), correct / len(loader.dataset)

def main():
    """
    Main Function to Run Full CNN Training Pipeline

    Steps:
    - Load dataset from CSV
    - Split into training and validation sets
    - Apply preprocessing and augmentation
    - Load ResNet50 pretrained model
    - Freeze layers and replace classifier
    - Train model over multiple epochs
    - Save best performing model

    : return None
    """

    # Seeing If Cuda is Available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Loading Full CSV and Removing Mask
    df = pd.read_csv(csvPath)
    df = df[df['is_mask'] == False].copy()

    # Splitting Data 
    trainDf = df[df['split'] == 'train'].copy()
    valDf   = trainDf.sample(frac = valSplit, random_state = 42)
    trainDf = trainDf.drop(valDf.index)
    
    # Printing Summary
    printSummary('train', trainDf)
    printSummary('val', valDf)
    print()

    # Create PyTorch Dataset
    trainDataset = BrainTumorDataset(trainDf, dataRoot, transform = trainTransform)
    valDataset   = BrainTumorDataset(valDf,   dataRoot, transform = evalTransform)

    # Making Sure Class Mapping is Consistent
    classes = trainDataset.classes
    classestoIndex = trainDataset.class_to_idx
    valDataset.classes = classes
    valDataset.class_to_idx = classestoIndex

    # Creating Data Loaders
    trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle = True,  num_workers = numWorkers, pin_memory = True)
    valLoader   = DataLoader(valDataset,   batch_size = batchSize, shuffle = False, num_workers = numWorkers, pin_memory = True)

    # Running Model
    numClasses = len(classes)
    model = models.resnet50(pretrained = True)

    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replacing Final Classification Layer
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, numClasses))
    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)
    scheduler = StepLR(optimizer, step_size = 5, gamma = 0.5)

    # Training Model
    bestVal = 0.0
    print(f"Training for {epocs} epochs...\n")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10}  {'':>8}")
    print("-" * 64)

    # Running Model
    for epoch in range(1, epocs + 1):
        trainLoss, trainAcc = runEpoch(model, trainLoader, criterion, optimizer, device, training = True)
        valLoss, valAcc = runEpoch(model, valLoader, criterion, optimizer, device, training = False)
        scheduler.step()

        saved = ""
        if valAcc > bestVal:
            bestVal = valAcc
            torch.save({'model_state_dict': model.state_dict(), 'class_to_idx': classestoIndex, 'classes': classes, 'num_classes':numClasses,}, savePath)
            saved = "saved"

        # Printing EPOCH
        print(f"{epoch:>6}   {trainLoss:>9.4f}   {trainAcc:>9.4f}   "
              f"{valLoss:>9.4f}   {valAcc:>9.4f}  {saved}")

    # Printing Summary
    print(f"\nBest val accuracy: {bestVal:.4f}")
    print(f"Model saved to '{savePath}'")

# Running CNN
if __name__ == '__main__':
    main()
