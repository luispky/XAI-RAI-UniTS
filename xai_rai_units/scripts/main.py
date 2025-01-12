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
from typing import List, Dict, Callable, Optional

from xai_rai_units.src.paths import LABELS_PATH, FIGURES_DIR
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
FONT_SIZES: Dict[str, int] = {
    "suptitle": 14,
    "title": 12,
    "label": 11,
    "ticks": 10,
    "text_box": 10,
    "legend": 8,
}

# -- SPACING CONFIGURATION --
# You can adjust these to tighten or loosen the subplots.
SPACING_TOP: float = 0.93  # increase to bring the suptitle closer to the first row
SPACING_BOTTOM: float = 0.06
SPACING_LEFT: float = 0.04
SPACING_RIGHT: float = 0.96
SPACING_H: float = -0.12  # vertical spacing between rows
SPACING_W: float = 0.07  # horizontal spacing between columns


# Dictionary of all the perturbation functions to apply
PERTURBATIONS: Dict[str, Callable] = {
    "Identity": pert.identity_perturbation_linspace,
    "Gaussian": pert.gaussian_perturbation_linspace,
    "Blur": pert.blur_perturbation_linspace,
    "Occlusion": pert.occlusion_perturbation_linspace,
    "Void": pert.void_perturbation_linspace,
    "InvGrad": pert.inv_grad_perturbation_linspace,
    "Contrast": pert.increase_contrast_linspace,
}

# Tuple of model names to evaluate (used if model_name == "all")
MODEL_NAMES = (
    "alexnet",
    "resnet50",
    "swin_transformer",
    "vit",
)


def plot_model_classification(
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: List[str],
        magnitude: float = 0.01,
        n_classes: int = 4,
        do_yticks: bool = True,
        noise_prediction_change: float = 0.0,
        do_legend: bool = True):
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

    # Add vertical line at noise_prediction_change
    plt.axvline(noise_prediction_change, color="black", linestyle="--", linewidth=1.0)
    plt.ylim(-0.05, 1.05)
    plt.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

    if do_legend:
        plt.legend(fontsize=FONT_SIZES["ticks"], prop={'size': FONT_SIZES["legend"]})


def process_image_for_model(
        filename: str,
        image: torch.Tensor,
        model: torch.nn.Module,
        target_layers,
        reshape_transform,
        generator: ExplanationGenerator,
        labels: List[str],
        n_perturbations: int,
        magnitude: float,
        show_figures: bool = True):
    """
    Generate perturbed images, produce explanations, and plot:
      1. The last perturbed image (row 1).
      2. First explanation of the perturbed image (row 2).
      3. Classification probabilities vs. noise fraction (row 3).
    
    Args:
        filename (str): The filename (or identifier) of the image being processed.
        image (torch.Tensor): The original unperturbed image, shape [channels, height, width].
        model (torch.nn.Module): The PyTorch model used for predictions.
        target_layers: The layers in the model targeted by the XAI method.
        reshape_transform: Optional function to reshape intermediate results for XAI.
        generator (ExplanationGenerator): An object to generate explanations for perturbed images.
        labels (List[str]): List of class labels for the model's output indices.
        n_perturbations (int): Number of images used for perturbation (linspace).
        magnitude (float): Magnitude of the perturbation.
        show_figures (bool): If True (default), display figures; if False, saves to FIGURES_DIR/robustness.
    
    Returns:
        None
    """
    if show_figures:
        # Show the figure interactively, do not save
        output_filepath = None
    else:
        # Save the figure to the "robustness" subdirectory under FIGURES_DIR
        output_filename = (
            f"{model.__class__.__name__}_mag{magnitude}_n-pert{n_perturbations}_{filename}"
        )
        robustness_dir = FIGURES_DIR / "robustness"
        robustness_dir.mkdir(parents=True, exist_ok=True)
        output_filepath = robustness_dir / f"{output_filename}.png"
        # If the output file already exists, skip processing
        if output_filepath.exists():
            print(f"Skipping {filename}: '{output_filepath}' already exists.")
            return

    # Create the figure and axes for subplots
    fig, ax = plt.subplots(
        nrows=3,
        ncols=len(PERTURBATIONS),
        figsize=(12, 8),
        gridspec_kw={'height_ratios': [1, 1, 0.82]}  # last row is a bit smaller
    )

    # Suptitle for model name, image file, and magnitude, placed closer
    plt.suptitle(
        f"Model: {model.__class__.__name__} | Image: {filename} | Magnitude: {magnitude}",
        fontsize=FONT_SIZES["suptitle"]
    )

    # Enumerate over the dictionary to get both the key (string) and the function
    for i_pert, (pert_key, perturbation) in enumerate(PERTURBATIONS.items()):
        # Row 1: Display the last perturbed image
        plt.sca(ax[0, i_pert])
        plt.title(pert_key, fontsize=FONT_SIZES["title"])

        perturbed_images = perturbation(
            image, magnitude=magnitude, n=n_perturbations, model=model
        )
        # Clamp image data to [0,1] to avoid matplotlib clipping messages
        img_disp = np.clip(perturbed_images[-1].permute(1, 2, 0).numpy(), 0, 1)
        plt.imshow(img_disp)
        plt.axis("off")

        # Row 2: Noisy explanations
        plt.sca(ax[1, i_pert])
        explanations, pred_labels, noise_fraction_changes, _ = generator.generate_explanations(
            perturbed_images, filename, target_layers, reshape_transform, verbose=False
        )

        # Pick the second explanation if available, else the first
        j = 1 if len(explanations) >= 2 else 0

        noisy_explanation = explanations[j].clone()

        # Normalize explanation to [0,1] and clamp
        _min, _max = noisy_explanation.min(), noisy_explanation.max()
        denom = max(_max - _min, 1e-8)
        noisy_explanation = (noisy_explanation - _min) / denom
        noisy_explanation = torch.clamp(noisy_explanation, 0, 1)

        plt.imshow(noisy_explanation.numpy())
        plt.axis("off")

        # Text box above explanation
        label_text = f"{pred_labels[j]}"
        if noise_fraction_changes[j] > 0:
            label_text += f'\n(mag={noise_fraction_changes[j]:.1%})'
        ax[1, i_pert].text(
            0.5, 0.99,
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
        do_yticks = (i_pert == 0)
        plot_model_classification(
            model=model,
            images=perturbed_images,
            labels=labels,
            magnitude=magnitude,
            do_yticks=do_yticks,
            noise_prediction_change=float(noise_fraction_changes[j]),
            do_legend=(i_pert == 0),
        )

    # In the last row, set x-limits and ticks to [0, 0.25, 0.5, 0.75, 1.0]
    for col_idx in range(len(PERTURBATIONS)):
        ax[2, col_idx].set_xlim(0, 1)
        ax[2, col_idx].set_xticks(np.linspace(0, 1, 5)[:-1])
        ax[2, col_idx].tick_params(axis="x", labelsize=FONT_SIZES["ticks"])

    # A single x-label for the entire figure
    if hasattr(fig, "supxlabel"):  # Matplotlib 3.4+ feature
        fig.supxlabel("Noise fraction", fontsize=FONT_SIZES["label"], y=0.001)
    else:
        plt.xlabel("Noise fraction", fontsize=FONT_SIZES["label"])

    fig.subplots_adjust(
        top=SPACING_TOP,
        bottom=SPACING_BOTTOM,
        left=SPACING_LEFT,
        right=SPACING_RIGHT,
        hspace=SPACING_H,
        wspace=SPACING_W
    )

    # Show or save the figure
    if show_figures:
        plt.show()
        plt.close(fig)
    else:
        assert output_filepath is not None, "Output path should not be None when show_figures=False."
        plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {output_filepath}")
        plt.close(fig)


def main(
        library: str = "gradcam",
        method: str = "GradCAM",
        sample_images: int = 5,
        n_perturbations: int = 30,
        magnitude: float = 0.1,
        seed: int = 42,
        model_name: Optional[str] = "alexnet",
        show_figures: bool = True,
) -> None:
    """
    Main function to run the perturbation and explanation generation pipeline:
      1. Load labels (constant path from LABELS_PATH).
      2. Loop over either the specified model (if model_name != "all")
         or all models in MODEL_NAMES (if model_name == "all").
      3. For each model and each sampled image:
         a. Generate perturbed images (row 1).
         b. Generate explanations (row 2).
         c. Plot classification probabilities vs. noise fraction (row 3).
    
    Args:
        library (str, optional): Library to use for generating explanations (e.g., gradcam).
        method (str, optional): XAI method to use (e.g., GradCAM, LayerCAM, etc.).
        sample_images (int, optional): Number of local images to sample for demonstration (default: 5).
        n_perturbations (int, optional): Number of images used for perturbation (default: 30).
        magnitude (float, optional): Magnitude for perturbations (default: 0.1).
        seed (int, optional): Random seed for reproducibility (default: 42).
        model_name (str, optional): Defaults to "alexnet". If set to "all", runs all models in MODEL_NAMES.
            Otherwise, runs only the specified model.
        show_figures (bool, optional): If True (default), display figures; if False,
        saves them to FIGURES_DIR/robustness.
    
    Returns:
        None
    """
    # Fix the random seed for reproducibility
    set_seed(seed)

    # Always use LABELS_PATH directly
    labels = load_labels(str(LABELS_PATH))

    # Load images (sample_filenames picks random images if no explicit list is given)
    filenames = sample_filenames(n=sample_images)
    images = load_local_images(filenames)

    # Determine which models to run
    if model_name == "all":
        models_to_run = MODEL_NAMES
    else:
        models_to_run = (model_name,)

    for m_name in models_to_run:
        model, target_layers, reshape_transform = setup_model_and_layers(m_name)
        generator = ExplanationGenerator(model, library, method)

        for filename, image in zip(filenames, images):
            print(f"\nProcessing image {filename} with model {m_name}...")

            process_image_for_model(
                filename=filename,
                image=image,
                model=model,
                target_layers=target_layers,
                reshape_transform=reshape_transform,
                generator=generator,
                labels=labels,
                n_perturbations=n_perturbations,
                magnitude=magnitude,
                show_figures=show_figures
            )


if __name__ == "__main__":
    """
    Parse command line arguments for controlling various parameters:
      - n_perturbations
      - library
      - method
      - magnitude
      - seed
      - sample_images
      - model_name (defaults to 'alexnet'; use 'all' to run on all models)
      - show_figures
    """
    parser = argparse.ArgumentParser(description="Run XAI-RAI pipeline with various parameters.")
    parser.add_argument(
        "--n_perturbations",
        type=int,
        default=40,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--sample_images",
        type=int,
        default=-1,
        help="Number of local images to sample for demonstration (default: 5)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="alexnet",
        help="Model to run (default: 'alexnet'). If 'all', runs all models in MODEL_NAMES."
    )
    parser.add_argument(
        "--show_figures",
        action="store_false",
        default=True,
        help=(
            "By default, figures are shown on screen. "
            "If specified, figures are saved to FIGURES_DIR/robustness instead of shown interactively."
        ),
    )

    args = parser.parse_args()
    main(
        library=args.library,
        method=args.method,
        sample_images=args.sample_images,
        n_perturbations=args.n_perturbations,
        magnitude=args.magnitude,
        seed=args.seed,
        model_name=args.model_name,
        show_figures=args.show_figures
    )
