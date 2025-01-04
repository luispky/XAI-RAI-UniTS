import torch
import torch.nn as nn
from typing import Union, List, Optional, Callable, Tuple
import numpy as np

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from xai_rai_units.src.utils import IDX_TO_CLASS_IMAGENET
from xai_rai_units.src.utils import preprocess_class_label, overlay_heatmaps

# Supported Grad-CAM methods
METHODS = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "HiResCAM": HiResCAM,
}


def gradcam_explanations_classifier_series(
        model: nn.Module,
        perturbed_images: torch.Tensor,
        class_label_imagenet: str = "chain_saw",
        target_layers: Optional[List[Union[nn.Module, nn.Sequential]]] = None,
        method: str = "GradCAM",
        predicted_labels: bool = False,
        reshape_transform: Optional[Callable] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[str]]]]:
    """
    Produces GradCAM explanations for a series of perturbed images of a given ImageNet class.

    :param model: A CNN model for which to produce explanations.
    :param perturbed_images: A batch of perturbed images as a Tensor.
    :param class_label_imagenet: The ImageNet class label (e.g., "chain_saw").
    :param target_layers: List of model layers to use for generating Grad-CAM explanations.
    :param method: The GradCAM method to use (e.g., "GradCAM", "GradCAM++").
    :param predicted_labels: Whether to return predicted labels.
    :param reshape_transform: Optional reshape function for transformers.
    
    :return: A Tensor of images with GradCAM explanations overlayed
             and optionally predicted labels as a list of strings.
    """
    # Validate the method
    if method not in METHODS:
        raise ValueError(f"Invalid method '{method}'. Choose from: {list(METHODS.keys())}")

    # Preprocess the class label
    class_idx = preprocess_class_label(class_label_imagenet)
    
    targets = [ClassifierOutputTarget(class_idx) for _ in range(len(perturbed_images))]

    # Get the Grad-CAM method
    gradcam_method = METHODS[method]

    # Check target layers
    if target_layers is None or not isinstance(target_layers, list):
        raise ValueError("`target_layers` must be a list of model layers (e.g., Bottleneck, Sequential).")

    # Generate Grad-CAM explanations
    print(f"Generating Grad-CAM explanations using method '{method}' for class '{class_label_imagenet}'...")
    with gradcam_method(model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform) as cam:
        # Compute Grad-CAM heatmaps for the perturbed images: (B, H, W)
        grayscale_cam = cam(input_tensor=perturbed_images, targets=targets)
        
        # Overlay heatmaps on the perturbed images
        explanations = overlay_heatmaps(grayscale_cam, perturbed_images)

        # Check for predicted labels
        pred_labels = None
        if predicted_labels:
            outputs = cam.outputs
            predictions = outputs.argmax(dim=1).to(torch.int32)
            pred_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]

    return explanations, pred_labels
