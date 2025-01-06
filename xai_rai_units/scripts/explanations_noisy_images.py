from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src.perturbations import noisy_image_linspace
from xai_rai_units.src.utils import (
    load_local_images,
    set_seed,
    show_images,
    setup_model_and_layers
)

import sys
import numpy as np


def main(library="gradcam", method="GradCAM", model_name="alexnet", n_images=16, magnitude=0.1, seed=42):
    """
    Main function to generate visual explanations for a given image using Grad-CAM or Captum methods.

    :param library: The library to use for generating explanations ('gradcam' or 'captum').
    :param method: The specific explanation method to use (e.g., 'GradCAM', 'LayerGradCam').
    :param model_name: The name of the pre-trained model to use (e.g., 'alexnet', 'resnet50').
    :param n_images: The number of noisy images to generate for explanation.
    :param magnitude: The maximum noise magnitude for generating perturbed images.
    :param seed: The random seed for reproducibility.
    """
    # Set the random seed for reproducibility
    set_seed(seed)

    # Define the image filename (assumes the image is stored locally)
    filename = "llama"
    # Load and preprocess the image
    preprocessed_image = load_local_images(filename)

    # Generate a sequence of noisy images for perturbation analysis
    noisy_images = noisy_image_linspace(preprocessed_image, magnitude, n_images)

    # Configure the model, target layers, and reshape transformation based on the model architecture
    model, target_layers, reshape_transform = setup_model_and_layers(model_name)

    # Initialize the explanation generator with the specified library and method
    generator = ExplanationGenerator(model, library, method)

    # Generate explanations and optionally retrieve predicted labels
    explanations, pred_labels = generator.generate_explanations(
        noisy_images,
        filename,
        target_layers,
        True,
        reshape_transform
    )

    # Visualize and save the generated explanations
    show_images(
        explanations,
        labels=pred_labels,
        save_fig=True,
        filename=f"{library}_{filename.split('.')[0]}_{model_name}"
    )

if __name__ == "__main__":
    main(library="gradcam", method="GradCAM", model_name="alexnet", magnitude=0.5, seed=42)
