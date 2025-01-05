"""
Use gradcam_explanations_classifier on a sequence of increasingly noisy images.
"""
from xai_rai_units.src.gradcam_explanations import gradcam_explanations_classifier_series
from xai_rai_units.src import utils
from xai_rai_units.src.perturbations import noisy_image_linspace, generate_noisy_images


def main(n_images=16, magnitude=.1, seed=42):
    """
    Use gradcam_explanations_classifier on a sequence of increasingly noisy images.
    """
    # Set seed for reproducibility
    utils.set_seed(seed)

    # Load test image
    filename = "llama"
    preprocessed_image = utils.load_local_images(filename)

    # Generate a sequence of noisy images
    # noisy_images = noisy_image_linspace(preprocessed_image, magnitude, n_images)
    noisy_images = generate_noisy_images(preprocessed_image, magnitude, n_images)

    # utils.show_images(noisy_images, save_fig=True, filename=f"noisy_{filename.split('.')[0]}.png")

    # Load model
    model_name = 'resnet50'
    model = utils.load_model(model_name)
    
    # Target layers for Grad-CAM
    reshape_transform = None
    target_layers = []
    # ! NOTICE: The target layer for the model should be selected according to the model architecture
    if model_name == 'alexnet':
        target_layers = [model.features[10]]
    elif model_name == 'resnet50':
        target_layers = [model.layer4[-1].conv3]
    # ! NOTICE: Vision transformer models don't have Convolutional layers
    elif model_name == 'swin_transformer':
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = utils.reshape_transform_swin_transformer
    elif model_name == 'vit':
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = utils.reshape_transform_vit

    # Grad-CAM explanations
    explanations, pred_labels = gradcam_explanations_classifier_series(
        model=model,
        perturbed_images=noisy_images,
        class_label_imagenet=filename,  # Internally preprocesses the label
        target_layers=target_layers,
        method="GradCAM",
        predicted_labels=True,
        reshape_transform=reshape_transform
    )

    utils.show_images(explanations, labels=pred_labels, save_fig=True,
                      filename=f"gradcam_{filename.split('.')[0]}_{model_name}")


if __name__ == "__main__":
    main(magnitude=.8)
