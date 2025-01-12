NOW ENSURE THIS OTHER CODE IS ALSO SAVING THE PLOTS PROPERLY WITHOUT MODIFYING ITS FUNCTIONALITY!!! 
```python
import os
import yaml
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

# Mapping of perturbation names to functions
PERTURBATIONS = {
    "Identity": pert.identity_perturbation_linspace,
    "Gaussian": pert.gaussian_perturbation_linspace,
    "Blur": pert.blur_perturbation_linspace,
    "Occlusion": pert.occlusion_perturbation_linspace,
    "Void": pert.void_perturbation_linspace,
    "InvGrad": pert.inv_grad_perturbation_linspace,
}

def process_images(config):
    """
    Process images across different models and perturbation methods based on the provided config.
    """
    set_seed(config["seed"])
    filenames = sample_filenames(n=config["sample_images"])
    images = load_local_images(filenames)

    for model_name in config["model_names"]:
        model, target_layers, reshape_transform = setup_model_and_layers(model_name)
        for perturbation_name, perturbation_func in PERTURBATIONS.items():
            if perturbation_name not in config["perturbation_names"]:
                continue  # Skip perturbations not listed in the config

            print(f"\nProcessing: Model={model_name}, Perturbation={perturbation_name}")
            for filename, image in zip(filenames, images):
                # Generate the output filename
                explanations_dir = FIGURES_DIR / "explanations"
                explanations_dir.mkdir(parents=True, exist_ok=True)
                output_filename = f"{config['library']}_{config['method']}_{model_name}_{perturbation_name}_{filename}"
                output_path = os.path.join(explanations_dir, output_filename)

                # Check if the file already exists
                if os.path.exists(output_path):
                    print(f"Skipping existing file: {output_path}")
                    continue

                try:
                    # Generate a sequence of perturbed images
                    perturbed_images = perturbation_func(
                        image, magnitude=config["magnitude"], n=config["n_perturbations"], model=model
                    )

                    # Initialize the explanation generator
                    generator = ExplanationGenerator(model, config["library"], config["method"])

                    # Generate explanations and predicted labels
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
                        print(f"{str(label):<20} | {noise:.2f}")

                    # Save the generated explanations
                    show_images(
                        explanations,
                        labels=pred_labels,
                        save_fig=True,
                        filename=output_path,
                        show=False,
                    )

                except Exception as e:
                    print(f"\n{e}")
                    continue

if __name__ == "__main__":
    # Load the configuration from the YAML file
    config_file = "explanations_config.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Process images based on the configuration
    process_images(config)
```
ADD DOCUMENTATION, IMPROVE READABILITY, AND ENSURE THE CODE FOLLOW THE BEST PRACTICES.