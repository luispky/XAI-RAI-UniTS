# Test the accuracy of resnet50 on the test set of Imagenette

import sys

import test
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

from pathlib import Path

# Add src directory to Python path
PARENT_DIRECTORY = str(Path(__file__).resolve().parent.parent)
sys.path.append(PARENT_DIRECTORY + "/src")

from utils import download_imagenette, load_imagenette

def compute_proportion_dataloder(loader):
    labels = np.concatenate([y.numpy() for _, y in loader])
    unique, counts = np.unique(labels, return_counts=True)
    return unique, np.round(counts/len(labels),2)

def main():
    # Load Imagenette test set
    download_imagenette()
    _, test_loader = load_imagenette()
    print("Test dataset size:", len(test_loader.dataset))
    
    # Load a pretrained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    
    num_classes = 1000  # Imagenette has 10 classes
    correct_classifications = np.zeros(num_classes, dtype=np.int64)
    total_classifications = np.zeros(num_classes, dtype=np.int64)
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Update counts for each class
            for label in range(num_classes):
                mask = (labels == label)
                correct_classifications[label] += (predicted[mask] == labels[mask]).sum().item()
                total_classifications[label] += mask.sum().item()
    
    # Compute overall and per-class accuracy
    overall_accuracy = correct_classifications.sum() / total_classifications.sum()
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    
    # Avoid division by zero for per-class accuracy
    per_class_accuracy = np.zeros(num_classes, dtype=np.float32)
    non_zero_mask = total_classifications > 0
    per_class_accuracy[non_zero_mask] = correct_classifications[non_zero_mask] / total_classifications[non_zero_mask]
    
    # Print the accuracies only for the non-zero entries
    for label, accuracy in enumerate(per_class_accuracy):
        if total_classifications[label] > 0:
            print(f"Class {label}: {accuracy:.2%}")

if __name__ == "__main__":
    main()

    