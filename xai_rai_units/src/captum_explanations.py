import torch
import torch.nn as nn
from typing import Union, List, Optional, Tuple
from captum.attr import (
    LayerGradCam,
    LayerAttribution,
    LayerDeepLift,
    LayerConductance,
    GuidedGradCam,
    DeepLift,
)
from xai_rai_units.src.utils import IDX_TO_CLASS_IMAGENET
from xai_rai_units.src.utils import preprocess_class_label, overlay_heatmaps


METHODS = {
    "GradCam": LayerGradCam,
    "GuidedGradCam": GuidedGradCam,
    "DeepLift": DeepLift,
    "LayerDeepLift": LayerDeepLift,
    "LayerConductance": LayerConductance,
}

def captum_explanations_classifier_series(
    model: nn.Module,
    perturbed_images: torch.Tensor,
    class_label_imagenet: str,
    target_layers: nn.Module,
    method: str = "GradCam",
    predicted_labels: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[str]]]]:
    """
    Produces feature attribution explanations for a series of perturbed images of a given ImageNet class
    using the feature attribution classes from the Captum library.

    :param model: A CNN model for which to produce explanations.
    :param perturbed_images: A batch of perturbed images from a given ImageNet class as a Tensor.
    :param class_label_imagenet: The ImageNet class label (e.g., "chain_saw").
    :param target_layers: List of model layers to use for generating Grad-CAM explanations.
    :param predicted_labels: Whether to return predicted labels.

    :return: A Tensor of images with feature attribution explanations overlayed
             and optionally predicted labels as a list of strings.
    """
    # Preprocess the class label
    class_idx = preprocess_class_label(class_label_imagenet)

    # Initialize the explanation method
    explanation_method = METHODS[method]
    explainer = explanation_method(model) if method in {"GuidedGradCam", "DeepLift"} else explanation_method(model, target_layers[0])

    # Compute the feature attributions
    attr = explainer.attribute(perturbed_images, target=torch.tensor([class_idx] * len(perturbed_images)))

    # Generate explanations
    if method == "GradCam" or method in {"LayerDeepLift", "LayerConductance"}:
        if method in {"LayerDeepLift", "LayerConductance"}:
            attr = attr.mean(dim=1, keepdim=True)  # Average across channels
        upsampled_attr = LayerAttribution.interpolate(attr, (224, 224)).squeeze(1).detach().numpy()
        explanations = overlay_heatmaps(upsampled_attr, perturbed_images)
    else:
        explanations = attr
    
    # Get predicted labels if requested
    pred_labels = None
    if predicted_labels:
        model.eval()
        with torch.no_grad():
            outputs = model(perturbed_images)
            predictions = outputs.argmax(dim=1).to(torch.int32)
            pred_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]

    return (explanations, pred_labels) if predicted_labels else explanations