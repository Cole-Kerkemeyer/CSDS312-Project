"""
CNN Image Classification Testing Functions

Authors: Cole Kerkemeyer, Phillip Kornberg
Date: 4.1.26
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm


def getImagePathsAndLabels(baseDir):
    """
    Function to Collect Tumor Type and Image Path from Test Folder
    
    : param str baseDir - Path of Base Directory
    
    : return arr imagePaths - Path of All Images
    : return arr labels - Labels of All Images
    """

    # Initiating Arrays
    imagePaths = []
    labels = []

    # Iterating Over Each Subfolder
    for className in os.listdir(baseDir):
        classPath = os.path.join(baseDir, className)
        if not os.path.isdir(classPath):
            continue

        # Collecting Images in Each Subfolder
        for imgFile in os.listdir(classPath):
            if imgFile.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                imagePaths.append(os.path.join(classPath, imgFile))
                labels.append(className)

    return imagePaths, labels

def classifyImages(imgPaths, labels):
    """
    Function to Test Image Classification Using CNN Modle
    
    : param arr imgPaths - Path of Images to Test
    : param arr labels - Type of Tumor for Accuracy Calculation
    
    : return float accuracy - Accuracy of Predictions
    """
    # Specifying Device Usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setting Mapping for Tumor Types
    idxToClass = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
    classToIdx = {v: k for k, v in idxToClass.items()}

    # Load Model, Regularization, and Setting Output Layer to 4 Classes
    model = models.resnet50()
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, 4))

    # Loading Trained Model Weights
    checkpoint = torch.load('brain_tumor_model.pth', map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    mean, std  = [0.1735, 0.1735, 0.1735], [0.1771, 0.1771, 0.1771]

    # Transforming Image and Normalizing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Initializing Accuracy Count
    correctCount = 0

    # Loop through all images
    for imgPath, label in tqdm(zip(imgPaths, labels), total=len(imgPaths), desc="Classifying Images"):
        # Loading Image
        image = Image.open(imgPath).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Predicting Image and Printing Probabilities
        with torch.no_grad():
            probs = torch.softmax(model(image), dim=1)[0]
            predIdx = probs.argmax().item()

        # Checking If Prediction Is Correct
        trueIdx = classToIdx[label]
        if predIdx == trueIdx:
            correctCount += 1

    # Calculating and Returning Accuracy
    accuracy = correctCount / len(imgPaths)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    return accuracy

# Classifying Test Images
imagePaths, labels = getImagePathsAndLabels("Data/test")
classifyImages(imagePaths, labels)