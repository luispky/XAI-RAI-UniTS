from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src.perturbations import (
    noisy_image_linspace, 
    generate_noisy_images
)
from xai_rai_units.src.utils import (
    load_local_images,
    set_seed,
    show_images, 
    setup_model_and_layers
)

def main(library="gradcam", method="GradCAM", model_name="alexnet", n_images=16, magnitude=0.1, seed=42):
    set_seed(seed)

    filename = "llama"
    preprocessed_image = load_local_images(filename)

    noisy_images = generate_noisy_images(preprocessed_image, magnitude, n_images)

    # Configure model, target layers, and reshape transformation
    model, target_layers, reshape_transform = setup_model_and_layers(model_name)

    generator = ExplanationGenerator(model, library, method)
    explanations, pred_labels = generator.generate_explanations(
        noisy_images, filename, target_layers, True, reshape_transform
    )

    show_images(explanations, labels=pred_labels, save_fig=True,
                      filename=f"{library}_{filename.split('.')[0]}_{model_name}")

if __name__ == "__main__":
    main(library="gradcam", method="GradCAM", model_name="resnet50", magnitude=0.8, seed=42)