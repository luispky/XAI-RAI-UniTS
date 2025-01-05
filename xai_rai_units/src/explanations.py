import torch
import torch.nn as nn
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
    "GuidedGradCam": GuidedGradCam,
    "DeepLift": DeepLift,
    "LayerDeepLift": LayerDeepLift,
    "LayerConductance": LayerConductance,
}

SUPPORTED_LIBRARIES = {"gradcam", "captum"}


class ExplanationGenerator:
    def __init__(self, model: nn.Module, library: str, method: str):
        """
        Initialize the explanation generator for the given library and method.
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
        predicted_labels: bool = False,
        reshape_transform: Optional[Callable] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[str]]]]:
        """
        Generate explanations for perturbed images using the selected library and method.
        """
        # Preprocess the class label
        class_idx = preprocess_class_label(class_label_imagenet)

        print(f"Generating explanations using the {self.method} method"
              f" from the {self.library} library for the class {class_label_imagenet}"
              f" from the ImageNet dataset.")

        if self.library == "gradcam":
            return self._generate_gradcam_explanations(
                perturbed_images, class_idx, target_layers, predicted_labels, reshape_transform
            )
        elif self.library == "captum":
            return self._generate_captum_explanations(perturbed_images, class_idx, target_layers, predicted_labels)

    def _generate_gradcam_explanations(
        self,
        perturbed_images: torch.Tensor,
        class_idx: int,
        target_layers: List[Union[nn.Module, nn.Sequential]],
        predicted_labels: bool,
        reshape_transform: Optional[Callable],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[str]]]]:
        """
        Generate explanations using Grad-CAM.
        """
        if self.method not in GRADCAM_METHODS:
            raise ValueError(f"Invalid Grad-CAM method '{self.method}'. Choose from {list(GRADCAM_METHODS.keys())}")

        explainer = GRADCAM_METHODS[self.method]

        if not target_layers:
            raise ValueError("`target_layers` must be provided for Grad-CAM methods.")

        with explainer(model=self.model, target_layers=target_layers, reshape_transform=reshape_transform) as cam:
            grayscale_cam = cam(input_tensor=perturbed_images, targets=[ClassifierOutputTarget(class_idx)] * len(perturbed_images))
            explanations = overlay_heatmaps(grayscale_cam, perturbed_images)

            pred_labels = None
            if predicted_labels:
                outputs = cam.outputs
                predictions = outputs.argmax(dim=1).to(torch.int32)
                pred_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]

            return explanations, pred_labels

    def _generate_captum_explanations(
        self,
        perturbed_images: torch.Tensor,
        class_idx: int,
        target_layers: List[Union[nn.Module, nn.Sequential]],
        predicted_labels: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[str]]]]:
        """
        Generate explanations using Captum.
        """
        if self.method not in CAPTUM_METHODS:
            raise ValueError(f"Invalid Captum method '{self.method}'. Choose from {list(CAPTUM_METHODS.keys())}")

        explanation_method = CAPTUM_METHODS[self.method]
        explainer = (
            explanation_method(self.model)
            if self.method in {"GuidedGradCam", "DeepLift"}
            else explanation_method(self.model, target_layers[0])
        )

        attr = explainer.attribute(perturbed_images, target=torch.tensor([class_idx] * len(perturbed_images)))

        if self.method in {"GradCam", "LayerDeepLift", "LayerConductance"}:
            if self.method in {"LayerDeepLift", "LayerConductance"}:
                attr = attr.mean(dim=1, keepdim=True)
            upsampled_attr = LayerAttribution.interpolate(attr, (224, 224)).squeeze(1).detach().numpy()
            explanations = overlay_heatmaps(upsampled_attr, perturbed_images)
        else:
            explanations = attr

        pred_labels = None
        if predicted_labels:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(perturbed_images)
                predictions = outputs.argmax(dim=1).to(torch.int32)
                pred_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]

        return explanations, pred_labels if predicted_labels else explanations