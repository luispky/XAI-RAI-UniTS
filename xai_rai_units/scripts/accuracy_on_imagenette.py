"""
Test the accuracy of resnet50 on the test set of Imagenette
"""
import torch
from sklearn.metrics import accuracy_score

from xai_rai_units.src.utils import download_imagenette, load_imagenette
from xai_rai_units.src.utils import IDX_TO_CLASS_IMAGENET
from xai_rai_units.src.utils import transform_imagenette_to_imagenet_indices
from xai_rai_units.src.utils import load_model
from xai_rai_units.src.paths import RESULTS_DIR


def main():
    """
    Test the accuracy of imagenette, a smaller
    version of ImageNet, with only 10 classes.
    """
    # Load Imagenette test set
    download_imagenette()
    _, test_loader = load_imagenette()
    
    # Load a pretrained classifier model
    model_name = "alexnet"
    model = load_model(model_name)
    model.eval()
    
    # Evaluate the model on the test set of Imagenette
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    # Convert the true and predicted labels to class names
    y_true = transform_imagenette_to_imagenet_indices(torch.tensor(y_true))
    y_pred = torch.tensor(y_pred)
    
    # Global accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Global accuracy: {accuracy:.2f}")
    
    # Per-class accuracy
    y_unique = torch.unique(y_true)
    per_class_accuracy = {}
    
    for y in y_unique:
        idx = y_true == y
        per_class_accuracy[IDX_TO_CLASS_IMAGENET[y.item()]] = accuracy_score(y_true[idx], y_pred[idx])
    
    print("\nPer-class accuracy:")
    for class_name, acc in per_class_accuracy.items():
        print(f"{class_name}: {acc:.2f}")
    
    # Output the results to a txt file in the RESULTS_DIR directory
    results_file = RESULTS_DIR / f"{model_name}_imagenette_classification_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Global accuracy: {accuracy:.2f}\n\n")
        f.write("Per-class accuracy:\n")
        for class_name, acc in per_class_accuracy.items():
            f.write(f"{class_name}: {acc:.2f}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()