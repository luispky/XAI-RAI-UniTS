import argparse
from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src.utils import (
    load_local_images,
    set_seed,
    show_images,
    setup_model_and_layers,
    sample_filenames,
)
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

def main(args):
    """
    Main function to generate visual explanations for a given image using Grad-CAM or Captum methods.
    """
    # Validate the selected perturbation method
    if args.perturbation_name not in PERTURBATIONS:
        raise ValueError(f"Invalid perturbation name '{args.perturbation_name}'. "
                         f"Choose from: {', '.join(PERTURBATIONS.keys())}")

    # Retrieve the corresponding perturbation function
    perturbation = PERTURBATIONS[args.perturbation_name]
    
    print(f"\n Perturbation method: {args.perturbation_name}, Model: {args.model_name}")

    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Sample image filenames
    filenames = sample_filenames(n=args.sample_images)
    
    # Load and preprocess the images
    images = load_local_images(filenames)
    
    # Configure the model, target layers, and reshape transformation based on the model architecture
    model, target_layers, reshape_transform = setup_model_and_layers(args.model_name)
    
    for filename, image in zip(filenames, images):
        try:
            # Generate a sequence of perturbed images for perturbation analysis
            perturbed_images = perturbation(image, magnitude=args.magnitude, n=args.n_perturbations, model=model)
            
            # Initialize the explanation generator with the specified library and method
            generator = ExplanationGenerator(model, args.library, args.method)

            # Generate explanations and optionally retrieve predicted labels
            explanations, pred_labels, noise_fraction_changes, _ = generator.generate_explanations(
                perturbed_images=perturbed_images,
                class_label_filename_imagenet=filename,
                target_layers=target_layers,
                reshape_transform=reshape_transform,
                resnet50_likely_class=True
            )
            
            # Print header with proper alignment
            print(f"\n{'Label':<20} | {'Percentage of Noise':<20}")
            print('-' * 40)  # Separator line for better readability

            # Print each label and noise percentage
            for label, noise in zip(pred_labels, noise_fraction_changes):
                print(f"{str(label):<20} | {noise:.2f}")

            # Visualize and save the generated explanations
            show_images(
                explanations,
                labels=pred_labels,
                save_fig=False,
            )
        
        except Exception as e:
            print(f"\n{e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual explanations for images using XAI techniques.")
    parser.add_argument("--library", type=str, default="gradcam", choices=["gradcam", "captum"],
                        help="Library to use for generating explanations.")
    parser.add_argument("--method", type=str, default="GradCAM",
                        help="Explanation method to use (e.g., 'GradCAM', 'LayerGradCam').")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Name of the pre-trained model to use (e.g., 'alexnet', 'resnet50').")
    parser.add_argument("--sample_images", type=int, default=5,
                        help="Number of sample images to process.")
    parser.add_argument("--perturbation_name", type=str, default="Gaussian", 
                        choices=list(PERTURBATIONS.keys()),
                        help="Name of the perturbation method to use.")
    parser.add_argument("--n_perturbations", type=int, default=5,
                        help="Number of perturbed images to generate for perturbation analysis.")
    parser.add_argument("--magnitude", type=float, default=0.2,
                        help="Maximum noise magnitude for generating perturbed images.")
    parser.add_argument("--seed", type=int, default=24,
                        help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    main(args)
