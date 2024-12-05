import sys

from src.gradcam_explanations import gradcam_explanations_classifier_series
from src.utils import load_local_images, load_model, show_images
from src.utils import generate_noisy_images, noisy_image_linspace
from src.utils import set_seed
from src.paths import IMAGE_DIR
from src.utils import reshape_transform_swin_transformer, reshape_transform_vit

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Load test image
    filename="cassette_player.jpg" 
    image_path = f"{str(IMAGE_DIR)}/{filename}"
    preprocessed_image = load_local_images(image_path)

    # Generate a sequence of noisy images
    n_images = 16
    magnitude = .1
    
    noisy_images = noisy_image_linspace(preprocessed_image, magnitude, n_images)

    show_images(noisy_images, save_fig=True, filename=f"noisy_{filename.split(".")[0]}.png")

    # Load model
    model_name='alexnet' 
    model = load_model(model_name)
    
    # Target layers for Grad-CAM
    reshape_transform = None
    target_layers = []
    if model_name == 'alexnet':
        target_layers = [model.features]
    elif model_name == 'resnet50':
        target_layers = [model.layer4[-1]]
    elif model_name == 'swin_transformer':
        target_layers = [model.layers[-1].blocks[-1].norm2]
        reshape_transform = reshape_transform_swin_transformer
    elif model_name == 'vit':
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
    
    img_overlay_cam, pred_labels  = gradcam_explanations_classifier_series(
        model=model,
        perturbed_images=noisy_images,
        class_label_imagenet=filename, # Internally preprocesses the label
        target_layers=target_layers,
        method="GradCAM",
        predicted_labels=True,
        reshape_transform=reshape_transform
    )
    
    show_images(img_overlay_cam, labels=pred_labels, save_fig=True, filename=f"gradcam_{filename.split(".")[0]}_{model_name}.png")

if __name__ == "__main__":
    main()