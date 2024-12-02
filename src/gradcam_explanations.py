#!/usr/bin/env python

import os
import torch
import torchvision
import torch.nn as nn
from typing import Dict, Union, List, Optional
import numpy as np

from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    XGradCAM,
    EigenCAM,
    HiResCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils import normalize_images, get_imagenet_idx_to_class
from torchvision.models import resnet50, ResNet50_Weights

# Supported Grad-CAM methods
METHODS = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "HiResCAM": HiResCAM,
}

# Class-to-Index and Index-to-Class mappings for ImageNet
IDX_TO_CLASS_IMAGENET = get_imagenet_idx_to_class()
CLASS_TO_IDX_IMAGENET = {v: k for k, v in IDX_TO_CLASS_IMAGENET.items()}


def gradcam_explanations_classifier_series(
    model: nn.Module,
    perturbed_images: torch.Tensor,
    class_label_imagenet: str = "chain_saw",
    target_layers: List[Union[nn.Sequential, nn.Module]] = None,
    method: str = "GradCAM",
    predicted_labels: bool = False,
) -> Union[torch.Tensor, List[str]]:
    """
    Produces GradCAM explanations for a series of perturbed images of a given ImageNet class.

    :param model: A CNN model for which to produce explanations.
    :param perturbed_images: A batch of perturbed images as a Tensor.
    :param class_label_imagenet: The ImageNet class label (e.g., "chain_saw").
    :param target_layers: List of model layers to use for generating Grad-CAM explanations.
    :param method: The GradCAM method to use (e.g., "GradCAM", "GradCAM++").
    
    :return: A Tensor of images with GradCAM explanations overlayed.
    :raises ValueError: If invalid method or class label is provided.
    """
    # Validate the method
    if method not in METHODS:
        raise ValueError(f"Invalid method '{method}'. Choose from: {list(METHODS.keys())}")

    # Validate class label
    if class_label_imagenet not in CLASS_TO_IDX_IMAGENET:
        raise ValueError(f"Invalid class label '{class_label_imagenet}'. Please provide a valid ImageNet class label.")

    # Fetch class index and initialize Grad-CAM targets
    class_idx = CLASS_TO_IDX_IMAGENET[class_label_imagenet]
    targets = [ClassifierOutputTarget(class_idx) for _ in range(len(perturbed_images))]

    # Normalize input images to range [0, 1]
    imgs_normalized = normalize_images(perturbed_images)

    # Get the Grad-CAM method
    gradcam_method = METHODS[method]

    # Check target layers
    if target_layers is None or not isinstance(target_layers, list):
        raise ValueError("`target_layers` must be a list of model layers (e.g., Bottleneck, Sequential).")

    # Generate Grad-CAM explanations
    print(f"Generating Grad-CAM explanations using method '{method}' for class '{class_label_imagenet}'...")
    with gradcam_method(model=model, target_layers=target_layers) as cam:
        # Compute Grad-CAM heatmaps
        grayscale_cam = cam(input_tensor=perturbed_images, targets=targets)

        # Overlay heatmaps onto normalized images
        imgs_overlay_cam = torch.tensor(
            np.array([show_cam_on_image(img, cam) for img, cam in zip(imgs_normalized, grayscale_cam)])
        )
        
        # Return predicted labels if requested
        if predicted_labels:
            predictions = cam.outputs.argmax(dim=1).to(torch.int32)
            pred_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]    
            
            return imgs_overlay_cam, pred_labels

    return imgs_overlay_cam