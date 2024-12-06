# todo why #!/usr/bin/env python ?
#!/usr/bin/env python
import torch
import torch.nn as nn
from typing import Union, List, Optional, Callable, Tuple
import numpy as np

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from xai_rai_units.src.utils import normalize_images, CLASS_TO_IDX_IMAGENET, IDX_TO_CLASS_IMAGENET


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
    class_label_imagenet = class_label_imagenet.lower().split(".")[0]

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
    with gradcam_method(model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform) as cam:
        # Compute Grad-CAM heatmaps
        grayscale_cam = cam(input_tensor=perturbed_images, targets=targets)

        # Overlay heatmaps onto normalized images
        imgs_overlay_cam = torch.Tensor(
            np.array(
                [show_cam_on_image(img, cam, use_rgb=True) for img, cam in zip(imgs_normalized, grayscale_cam)]
            )
        )

        # Check for predicted labels
        pred_labels = None
        if predicted_labels:
            # Ensure Grad-CAM outputs predictions (forward pass required)
            outputs = cam.outputs  # Check if `outputs` exists in the GradCAM implementation
            if outputs is None:
                raise ValueError("Grad-CAM outputs are not available for label prediction.")
            predictions = outputs.argmax(dim=1).to(torch.int32)
            pred_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]

    return imgs_overlay_cam, pred_labels
