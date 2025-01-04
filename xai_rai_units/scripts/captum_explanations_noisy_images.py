"""
Use gradcam_explanations_classifier on a sequence of increasingly noisy images.
"""
from xai_rai_units.src.captum_explanations import captum_explanations_classifier_series
from xai_rai_units.src.paths import IMAGE_DIR
from xai_rai_units.src.perturbations import noisy_image_linspace
from xai_rai_units.src import utils


def main(n_images=16, magnitude=.1, seed=42):
    """
    Use gradcam_explanations_classifier on a sequence of increasingly noisy images.
    """
    # Set seed for reproducibility
    utils.set_seed(seed)

    # Load test image
    filename = "cassette_player.jpg"
    image_path = f"{str(IMAGE_DIR)}/{filename}"
    preprocessed_image = utils.load_local_images(image_path)

    # Generate a sequence of noisy images
    noisy_images = noisy_image_linspace(preprocessed_image, magnitude, n_images)

    # utils.show_images(noisy_images, save_fig=True, filename=f"noisy_{filename.split('.')[0]}.png")

    # Load model
    model_name = 'alexnet'
    model = utils.load_model(model_name)
    
    target_layers = None
    if model_name == 'alexnet':
        target_layers = model.features[10]
    elif model_name == 'resnet50':
        target_layers = model.layer4[-1].conv3

    # Grad-CAM explanations
    explanations, pred_labels = captum_explanations_classifier_series(
        model=model,
        perturbed_images=noisy_images,
        class_label_imagenet=filename,  # Internally preprocesses the label
        target_layers=target_layers,
        method = 'LayerConductance',
        predicted_labels=True,
    )

    utils.show_images(explanations, labels=pred_labels)

if __name__ == "__main__":
    main(magnitude=.8)
