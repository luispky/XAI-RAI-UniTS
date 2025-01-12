import yaml
from typing import Dict, Any

from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src.utils import (
    load_local_images,
    set_seed,
    show_images,
    setup_model_and_layers,
    sample_filenames,
)
from xai_rai_units.src.paths import FIGURES_DIR
from xai_rai_units.src import perturbations as pert


# Mapping of perturbation names to their corresponding functions
PERTURBATIONS = {
    "Identity": pert.identity_perturbation_linspace,
    "Gaussian": pert.gaussian_perturbation_linspace,
    "Blur": pert.blur_perturbation_linspace,
    "Occlusion": pert.occlusion_perturbation_linspace,
    "Void": pert.void_perturbation_linspace,
    "InvGrad": pert.inv_grad_perturbation_linspace,
}


def process_images(config: Dict[str, Any]) -> None:
    """
    Process images by applying specified perturbations and generating explanations
    based on the parameters in the provided configuration dictionary.

    Steps:
      1. Set random seed for reproducibility.
      2. Sample or load the specified number of images.
      3. For each model in 'model_names':
         a. For each perturbation function in PERTURBATIONS (filtered by 'perturbation_names' in the config):
            i. Generate a sequence of perturbed images.
            ii. Create an ExplanationGenerator and generate explanations for the perturbed images.
            iii. Display and/or save the explanation images with their predicted labels.
            iv. Skip if the output file already exists.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing keys such as
            'seed', 'sample_images', 'model_names', 'magnitude', 'n_perturbations',
            'library', 'method', and 'perturbation_names'.
    """
    # 1. Set random seed for reproducibility
    set_seed(config["seed"])

    # 2. Load images (sample_filenames picks random images if no explicit list is given)
    filenames = sample_filenames(n=config["sample_images"])
    images = load_local_images(filenames)

    # 3. Process each model and perturbation
    for model_name in config["model_names"]:
        model, target_layers, reshape_transform = setup_model_and_layers(model_name)

        for perturbation_name, perturbation_func in PERTURBATIONS.items():
            # Only use perturbations specified in the config
            if perturbation_name not in config["perturbation_names"]:
                continue

            print(f"\nProcessing: Model={model_name}, Perturbation={perturbation_name}")

            for filename, image in zip(filenames, images):
                # Generate a directory for saving explanation images
                explanations_dir = FIGURES_DIR / "explanations"
                explanations_dir.mkdir(parents=True, exist_ok=True)

                # Generate the output filename
                output_filename = (
                    f"{config['library']}_{config['method']}_{model_name}_"
                    f"{perturbation_name}_mag-{config['magnitude']}_{filename}"
                )
                output_path = explanations_dir / output_filename

                # If the file already exists, skip
                if output_path.exists():
                    print(f"Skipping existing file: {output_path}")
                    continue

                try:
                    # i. Generate a sequence of perturbed images
                    perturbed_images = perturbation_func(
                        image,
                        magnitude=config["magnitude"],
                        n=config["n_perturbations"],
                        model=model,
                    )

                    # ii. Create an ExplanationGenerator to generate explanations
                    generator = ExplanationGenerator(
                        model=model,
                        library=config["library"],
                        method=config["method"],
                    )

                    # iii. Generate explanations and predicted labels
                    explanations, pred_labels, noise_fraction_changes, _ = generator.generate_explanations(
                        perturbed_images=perturbed_images,
                        class_label_filename_imagenet=filename,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform,
                        resnet50_likely_class=True,
                    )

                    print(f"\n{'Label':<20} | {'Percentage of Noise':<20}")
                    print('-' * 40)
                    for label, noise in zip(pred_labels, noise_fraction_changes):
                        print(f"{label:<20} | {noise:.2f}")

                    # iv. Save the generated explanations (no interactive display)
                    show_images(
                        images=explanations,
                        labels=pred_labels,
                        save_fig=True,
                        filename=str(output_path),  # pass as string
                        show=False,
                        proportions=noise_fraction_changes,
                    )

                except Exception as err:
                    print(f"\nAn error occurred: {err}")
                    continue


def main() -> None:
    """
    Main entry point to load configuration and process images.
    Expects a YAML config file named 'explanations_config.yaml'.
    """
    config_file = "explanations_config.yaml"
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Process images based on the configuration
    process_images(config)


if __name__ == "__main__":
    """
    Entrypoint for running the script directly. Loads the config from a YAML file
    and calls process_images() to apply perturbations and generate explanations.
    """
    main()
