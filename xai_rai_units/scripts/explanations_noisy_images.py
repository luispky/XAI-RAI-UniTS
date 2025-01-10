from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src.perturbations import gaussian_perturbation_linspace
from xai_rai_units.src.utils import (
    load_local_images,
    set_seed,
    show_images,
    setup_model_and_layers, 
    sample_filenames
)


def main(library="gradcam", method="GradCAM", model_name="alexnet",
         filename="monarch", n_images=16, magnitude=0.1, seed=42):
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
    # set_seed(seed)

    # Define the image filename (assumes the image is stored locally)
    filenames = sample_filenames()
    images = load_local_images(filenames)
    
    # Configure the model, target layers, and reshape transformation based on the model architecture
    model, target_layers, reshape_transform = setup_model_and_layers(model_name)
    
    for filename, image in zip(filenames, images):
        try:
            # Load and preprocess the image

            # Generate a sequence of noisy images for perturbation analysis
            noisy_images = gaussian_perturbation_linspace(image, magnitude, n_images)

            # Initialize the explanation generator with the specified library and method
            generator = ExplanationGenerator(model, library, method)

            # Generate explanations and optionally retrieve predicted labels
            explanations, pred_labels, noise_fraction_changes, _   = generator.generate_explanations(
                noisy_images,
                filename,
                target_layers,
                reshape_transform
            )
            
            # Print header with proper alignment
            print(f"{'Label':<15} | {'Percentage of Noise':<20}")
            print('-' * 40)  # Separator line for better readability

            # Print each label and noise percentage
            for label, noise in zip(pred_labels, noise_fraction_changes):
                print(f"{str(label):<15} | {noise:.2f}")

            # Visualize and save the generated explanations
            show_images(
                explanations,
                labels=pred_labels,
                save_fig=True,
                filename=f"{library}_{filename.split('.')[0]}_{model_name}"
            )
        except Exception as e:
            print(f"\n ❌ Error processing image {filename}: {e}")
            continue

if __name__ == "__main__":
    main(library="gradcam", method="GradCAM", model_name="resnet50",
         magnitude=0.5, seed=42)
