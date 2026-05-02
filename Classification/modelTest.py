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
    Function To Test Image Classification Using CNN Model

    : param arr imgPaths - Path Of Images To Test
    : param arr labels - Type Of Tumor For Accuracy Calculation

    : return float accuracy - Overall Accuracy Of Predictions
    : return dict classAccuracies - Per-Class Accuracy Of Predictions
    """

    import numpy as np

    # Specifying Device Usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setting Mapping For Tumor Types
    idxToClass = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
    classToIdx = {v: k for k, v in idxToClass.items()}

    # Loading Model Architecture
    model = models.resnet50()
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, 4))

    # Loading Trained Model Weights
    checkpoint = torch.load(
        'brain_tumor_model.pth',
        map_location=device,
        weights_only=False
    )

    stateDict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(stateDict)

    model.to(device)
    model.eval()

    # Normalization Values
    mean, std = [0.1735, 0.1735, 0.1735], [0.1771, 0.1771, 0.1771]

    # Defining Image Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Initializing Metrics
    correctCount = 0
    classCorrect = {cls: 0 for cls in classToIdx}
    classTotal = {cls: 0 for cls in classToIdx}

    # Storing Predictions For Advanced Metrics
    yTrue = []
    yPred = []

    # Looping Through Images
    for imgPath, label in tqdm(zip(imgPaths, labels), total=len(imgPaths), desc="Classifying Images"):
        # Loading Image
        image = Image.open(imgPath).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Running Model Inference
        with torch.no_grad():
            probs = torch.softmax(model(image), dim=1)[0]
            predIdx = probs.argmax().item()

        trueIdx = classToIdx[label]

        # Saving Predictions
        yTrue.append(trueIdx)
        yPred.append(predIdx)

        classTotal[label] += 1

        # Checking Correct Prediction
        if predIdx == trueIdx:
            correctCount += 1
            classCorrect[label] += 1

    # Overall Accuracy
    accuracy = correctCount / len(imgPaths)
    print(f"\nOverall Accuracy: {accuracy:.2%}")

    # Per-Class Accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 30)

    classAccuracies = {}

    for cls in classToIdx:
        classAccuracy = classCorrect[cls] / classTotal[cls] if classTotal[cls] > 0 else 0.0
        classAccuracies[cls] = classAccuracy
        print(f"  {cls:<15} {classAccuracy:.2%}  ({classCorrect[cls]}/{classTotal[cls]})")

    # Confusion Matrix
    numClasses = len(classToIdx)
    confusionMatrix = np.zeros((numClasses, numClasses), dtype=int)

    for t, p in zip(yTrue, yPred):
        confusionMatrix[t][p] += 1

    print("\nConfusion Matrix:")
    print(confusionMatrix)

    # Precision, Recall, F1 Score
    print("\nPrecision / Recall / F1 Per Class:")
    print("-" * 40)

    precision = {}
    recall = {}
    f1Score = {}

    for i in range(numClasses):
        truePositive = confusionMatrix[i][i]
        falsePositive = confusionMatrix[:, i].sum() - truePositive
        falseNegative = confusionMatrix[i].sum() - truePositive

        prec = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) > 0 else 0.0
        rec = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        className = idxToClass[i]

        precision[className] = prec
        recall[className] = rec
        f1Score[className] = f1

        print(f"{className:<15} {prec:.3f}  {rec:.3f}  {f1:.3f}")

    return accuracy, classAccuracies, precision, recall, f1Score


# Classifying Test Images
imagePaths, labels = getImagePathsAndLabels("Data/test")
classifyImages(imagePaths, labels)