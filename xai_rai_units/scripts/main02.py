"""
The purpose of this project is to find out whether XAI
tools are robust to various image perturbations. The
project will use multiple models (AlexNet, ResNet50, etc.)
to classify images, perturb them, and then generate
explanations for the classification.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from xai_rai_units.src.paths import LABELS_PATH
from xai_rai_units.src.alexnet import load_labels
from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src import perturbations as pert
from xai_rai_units.src.utils import (
    setup_model_and_layers,
    load_local_images,
    sample_filenames,
    set_seed,
)

# Font configuration dictionary for easy customization
FONT_SIZES = {
    "suptitle": 14,
    "title": 12,
    "label": 11,
    "ticks": 11,
    "text_box": 10,
}

# -- SPACING CONFIGURATION --
# You can adjust these to tighten or loosen the subplots.
SPACING_TOP = 0.92     # reduce to bring the suptitle closer to the first row
SPACING_BOTTOM = 0.06
SPACING_LEFT = 0.04
SPACING_RIGHT = 0.96
SPACING_H = 0.02       # vertical spacing between rows
SPACING_W = 0.02       # horizontal spacing between columns

# Tuple of all the perturbation functions to apply
PERTURBATIONS = (
    pert.identity_perturbation_linspace,
    pert.gaussian_perturbation_linspace,
    pert.blur_perturbation_linspace,
    pert.occlusion_perturbation_linspace,
    pert.void_perturbation_linspace,
    pert.inv_grad_perturbation_linspace,
)

# Tuple of model names to evaluate
MODEL_NAMES = (
    "alexnet",
    "resnet50",
    "swin_transformer",
    "vit",
)

def plot_model_classification(model, images, labels, magnitude=0.01, n_classes=4, do_yticks=True):
    """
    Plot the classification probabilities of the model for a range of perturbations.
    The horizontal axis corresponds to noise fractions from 0 to 1.
    """
    model.eval()
    out_init = model(images[0].unsqueeze(0))
    indices = np.argsort(out_init.detach().numpy().flatten())[-n_classes:]

    # This array is used internally but not explicitly shown on the final x-axis
    _ = np.linspace(0, 1, len(images)) ** 2 * magnitude

    out = model(images)
    out = torch.softmax(out, dim=1)
    proba = out.detach().numpy()[:, list(reversed(indices[:n_classes]))]

    # x-values for plotting: from 0 to 1, representing noise fraction
    noise_fraction_array = np.linspace(0, 1, len(images))

    # Plot probabilities vs. noise fraction
    plt.plot(noise_fraction_array, proba, label=[labels[i] for i in list(reversed(indices))])

    # Decide if we show the y-axis ticks and label
    if do_yticks:
        plt.ylabel("Probability", fontsize=FONT_SIZES["label"])
    else:
        plt.yticks([])
        plt.ylabel("")

    plt.ylim(0, 1.05)
    plt.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])
    plt.legend(fontsize=FONT_SIZES["ticks"])

def main(n_images=30, library="gradcam", method="GradCAM", magnitude=0.1, seed=42):
    """
    Main function to run the perturbation and explanation generation pipeline:
      1. Load labels (constant path from LABELS_PATH).
      2. Loop over all models in MODEL_NAMES.
      3. For each model and each sampled image:
         a. Generate perturbed images (row 1).
         b. Generate explanations (row 2).
         c. Plot classification probabilities vs. noise fraction (row 3).
    """
    # Fix the random seed for reproducibility
    set_seed(seed)
    
    # Always use LABELS_PATH directly
    labels = load_labels(str(LABELS_PATH))

    # Load images (sample_filenames picks random images if no explicit list is given)
    filenames = sample_filenames(n=5)
    images = load_local_images(filenames)

    for model_name in MODEL_NAMES:
        model, target_layers, reshape_transform = setup_model_and_layers(model_name)
        generator = ExplanationGenerator(model, library, method)

        for filename, image in zip(filenames, images):
            # <-- Only change here: pass gridspec_kw={'height_ratios': [1, 1, 0.85]}
            fig, ax = plt.subplots(
                nrows=3,
                ncols=len(PERTURBATIONS),
                figsize=(12, 8),
                gridspec_kw={'height_ratios': [1, 1, 0.82]}  # last row is a bit smaller
            )

            # Suptitle for model name, image file, and magnitude, placed closer
            plt.suptitle(
                f"Model: {model_name} | Image: {filename} | Magnitude: {magnitude}",
                fontsize=FONT_SIZES["suptitle"]
            )

            for i_pert, perturbation in enumerate(PERTURBATIONS):
                # Row 1: Display the last perturbed image
                plt.sca(ax[0, i_pert])
                name = (
                    perturbation.__name__
                    .replace("_perturbation", "")
                    .replace("_linspace", "")
                    .replace("_", "\n")
                )
                plt.title(name, fontsize=FONT_SIZES["title"])

                perturbed_images = perturbation(
                    image, magnitude=magnitude, n=n_images, model=model
                )
                # Clamp image data to [0,1] to avoid matplotlib clipping messages
                img_disp = np.clip(perturbed_images[-1].permute(1, 2, 0).numpy(), 0, 1)
                plt.imshow(img_disp)
                plt.axis("off")

                # Row 2: Noisy explanations
                plt.sca(ax[1, i_pert])
                explanations, pred_labels, noise_fraction_changes, attributions = generator.generate_explanations(
                    perturbed_images, filename, target_layers, reshape_transform
                )

                # Pick the second explanation if available, else the first
                j = 1 if len(attributions) >= 2 else 0
                noisy_explanation = explanations[j].clone()

                # Normalize explanation to [0,1] and clamp
                _min, _max = noisy_explanation.min(), noisy_explanation.max()
                denom = max(_max - _min, 1e-8)
                noisy_explanation = (noisy_explanation - _min) / denom
                noisy_explanation = torch.clamp(noisy_explanation, 0, 1)

                plt.imshow(noisy_explanation.numpy())
                plt.axis("off")

                # Text box above explanation
                label_text = f"{pred_labels[j]}:\n{noise_fraction_changes[j]:.2%}"
                ax[1, i_pert].text(
                    0.5, 0.98,
                    label_text,
                    color="white",
                    fontsize=FONT_SIZES["text_box"],
                    ha="center",
                    va="bottom",
                    transform=ax[1, i_pert].transAxes,
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
                )

                # Row 3: Classification probabilities vs. noise fraction
                plt.sca(ax[2, i_pert])
                # Show y-axis ticks only on first column to reduce duplication
                do_yticks = (i_pert == 0)
                plot_model_classification(model, perturbed_images, labels, magnitude, do_yticks=do_yticks)

            # In the last row, set x-limits and ticks to [0, 0.25, 0.5, 0.75, 1.0]
            for col_idx in range(len(PERTURBATIONS)):
                ax[2, col_idx].set_xlim(0, 1)
                ax[2, col_idx].set_xticks([0, 0.2, 0.4, 0.6, 0.8])
                ax[2, col_idx].tick_params(axis="x", labelsize=FONT_SIZES["ticks"])

            # A single x-label for the entire figure
            if hasattr(fig, "supxlabel"):  # Matplotlib 3.4+ feature
                fig.supxlabel("Noise fraction", fontsize=FONT_SIZES["label"], y=0.001)
            else:
                plt.xlabel("Noise fraction", fontsize=FONT_SIZES["label"])

            # Adjust subplot spacing so all rows are equally spaced and subplots are tight.
            # Tweak SPACING_* variables to your taste.
            fig.subplots_adjust(
                top=SPACING_TOP,
                bottom=SPACING_BOTTOM,
                left=SPACING_LEFT,
                right=SPACING_RIGHT,
                hspace=SPACING_H,
                wspace=SPACING_W
            )

            plt.show()

def parse_args():
    """
    Parse command line arguments for controlling various parameters:
      - n_images
      - library
      - method
      - magnitude
    LABELS_PATH is kept constant as per the request.
    """
    parser = argparse.ArgumentParser(description="Run XAI-RAI pipeline with various parameters.")
    parser.add_argument(
        "--n_images",
        type=int,
        default=30,
        help="Number of images used for perturbation (default: 30)."
    )
    parser.add_argument(
        "--library",
        type=str,
        default="gradcam",
        help="Library to use for generating explanations (e.g., gradcam)."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="GradCAM",
        help="XAI method to use (e.g., GradCAM, LayerCAM, etc.)."
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=0.1,
        help="Magnitude for perturbations (default: 0.1)."
    )
    # parser for the seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        n_images=args.n_images,
        library=args.library,
        method=args.method,
        magnitude=args.magnitude,
        seed=args.seed
    )
