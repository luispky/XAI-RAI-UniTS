import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Optional, Tuple, Callable
from captum.attr import (
    LayerGradCam,
    LayerAttribution,
    LayerDeepLift,
    LayerConductance,
    GuidedGradCam,
    DeepLift,
)
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    XGradCAM,
    EigenCAM,
    HiResCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from xai_rai_units.src.utils import (
    IDX_TO_CLASS_IMAGENET,
    preprocess_class_label,
    overlay_heatmaps,
)

# Constants for methods
GRADCAM_METHODS = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "HiResCAM": HiResCAM,
}

CAPTUM_METHODS = {
    "LayerGradCam": LayerGradCam,
    "LayerDeepLift": LayerDeepLift,
    "LayerConductance": LayerConductance,
    "GuidedGradCam": GuidedGradCam,
    "DeepLift": DeepLift,
}

SUPPORTED_LIBRARIES = {"gradcam", "captum"}

class ExplanationGenerator:
    """
    Interface to generate explanations for image classification models using Grad-CAM or Captum library methods.
    The class is tailed to work with models trained on the ImageNet dataset and generate explanations for a Tensor
    of perturbed images of a specific class label.
    The Grad-CAM methods can work with both convolutional and transformer-based architectures.
    The Captum methods are limited to convolutional architectures.
    """
    def __init__(self, model: nn.Module, library: str, method: str):
        """
        Initialize the explanation generator.

        :param model: A PyTorch model for which explanations will be generated. If the library is 'captum', only
                        convolutional models are supported. If the library is 'gradcam', both convolutional and
                        transformer-based models are supported.
        :param library: The library to use for explanations ('gradcam' or 'captum').
        :param method: The specific explanation method to use (e.g., 'GradCAM', 'LayerGradCam').
        """
        self.model = model
        self.library = library.lower()
        self.method = method

        if self.library not in SUPPORTED_LIBRARIES:
            raise ValueError(f"Unsupported library '{self.library}'. Choose from {SUPPORTED_LIBRARIES}.")

    def generate_explanations(
        self,
        perturbed_images: torch.Tensor,
        class_label_imagenet: str,
        target_layers: Optional[List[Union[nn.Module, nn.Sequential]]] = None,
        reshape_transform: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, List[str], np.ndarray, Optional[torch.Tensor]]:
        """
        Generate explanations for a batch of perturbed images using the specified library and method.
        The explanations returned only correspond to the images where the predicted label changed from the original
        image. The fraction of noise that caused a change in the predicted label is also returned.

        :param perturbed_images: A batch of perturbed input images from a ImageNet class.
        :param class_label_imagenet: The ImageNet class label for which explanations are generated.
        :param target_layers: The target layers for which explanations are generated.
        :param reshape_transform: A transformation function to reshape the input images when using Grad-CAM methods.

        :return: A tuple containing the explanation heatmaps, predicted labels, an array with the fraction of noise
                that caused a change in the predicted label, and the attributions (if available).
        """
        # Preprocess the class label
        class_idx = preprocess_class_label(class_label_imagenet)

        print(f"Generating explanations using the {self.method} method"
              f" from the {self.library} library for the class {class_label_imagenet}"
              f" from the ImageNet dataset.")
        
        explanations, predicted_labels, attributions = None, None, None
        if self.library == "gradcam":
            explanations, predicted_labels, attributions = self._generate_gradcam_explanations(
                perturbed_images, class_idx, target_layers, reshape_transform
            )
        elif self.library == "captum":
            explanations, predicted_labels, attributions = self._generate_captum_explanations(perturbed_images, class_idx, target_layers)
        
        n_images = len(perturbed_images)
        # Compute the fraction of noise that caused a change in the predicted label
        changes = np.array([i for i in range(1, n_images) if predicted_labels[i] != predicted_labels[i-1]])
        changes = np.insert(changes, 0, 0) # Include the first image with 0 noise
        noise_fraction_changes = changes / n_images
        
        explanations_changes = explanations[changes]        
        predicted_labels_changes = [predicted_labels[i] for i in changes]
        
        if attributions is not None:
            return explanations_changes, predicted_labels_changes, noise_fraction_changes, attributions[changes]
        else:
            return explanations_changes, predicted_labels_changes, noise_fraction_changes, None

    def _generate_gradcam_explanations(
        self,
        perturbed_images: torch.Tensor,
        class_idx: int,
        target_layers: List[Union[nn.Module, nn.Sequential]],
        reshape_transform: Optional[Callable],
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Generate explanations using Grad-CAM.
        """
        if self.method not in GRADCAM_METHODS:
            raise ValueError(f"Invalid Grad-CAM method '{self.method}'. Choose from {list(GRADCAM_METHODS.keys())}")

        explainer = GRADCAM_METHODS[self.method]

        if not target_layers:
            raise ValueError("`target_layers` must be provided for Grad-CAM methods.")

        with explainer(model=self.model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
            attributions = cam(input_tensor=perturbed_images,
                                targets=[ClassifierOutputTarget(class_idx)] * len(perturbed_images))
            explanations = overlay_heatmaps(attributions, perturbed_images)

            outputs = cam.outputs
            predictions = outputs.argmax(dim=1).to(torch.int32)
            predicted_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]

            return explanations, predicted_labels, torch.tensor(attributions)

    def _generate_captum_explanations(
        self,
        perturbed_images: torch.Tensor,
        class_idx: int,
        target_layers: List[Union[nn.Module, nn.Sequential]],
    ) -> Tuple[torch.Tensor, List[str], Optional[torch.Tensor]]:
        """
        Generate explanations using Captum.
        """
        if self.method not in CAPTUM_METHODS:
            raise ValueError(f"Invalid Captum method '{self.method}'. Choose from {list(CAPTUM_METHODS.keys())}")

        explanation_method = CAPTUM_METHODS[self.method]
        explainer = (
            explanation_method(self.model)
            if self.method == "DeepLift"
            else explanation_method(self.model, target_layers[0])
        )

        # Generate attributions
        attributions = explainer.attribute(perturbed_images, target=torch.tensor([class_idx] * len(perturbed_images)))

        # Generate the predicted labels
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(perturbed_images)
            predictions = outputs.argmax(dim=1).to(torch.int32)
            predicted_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]
        
        if self.method in {"LayerGradCam", "LayerDeepLift", "LayerConductance"}:
            if self.method in {"LayerDeepLift", "LayerConductance"}:
                attributions = attributions.mean(dim=1, keepdim=True)
            attributions = LayerAttribution.interpolate(attributions, (224, 224)).squeeze(1).detach().numpy()
            explanations = overlay_heatmaps(attributions, perturbed_images)
            return explanations, predicted_labels, torch.tensor(attributions)
        else:
            return attributions, predicted_labels, None
