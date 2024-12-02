import sys
from torchvision.models import resnet50, ResNet50_Weights

from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from gradcam_explanations import gradcam_explanations_classifier_series
from gradcam_explanations import CLASS_TO_IDX_IMAGENET, IDX_TO_CLASS_IMAGENET
from utils import load_local_images, show_images, generate_noisy_images

def main():

    # Load test image
    image_path = "../test_images/llama.jpeg"
    preprocessed_image = load_local_images(image_path)
    class_label = "llama"

    # Generate a sequence of noisy images
    n_images = 16
    magnitude = 0.5
    noisy_images = generate_noisy_images(preprocessed_image, n_images, magnitude)

    show_images(noisy_images, save_fig=True, filename="noisy_llama.png")

    # Load a pretrained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Target layers for Grad-CAM
    target_layers = [model.layer4[-1]]
    
    img_overlay_cam = gradcam_explanations_classifier_series(
        model=model,
        perturbed_images=noisy_images,
        class_label_imagenet=class_label,
        target_layers=target_layers,
        method="GradCAM",
    )
    
    show_images(img_overlay_cam, save_fig=True, filename="gradcam_llama.png")

if __name__ == "__main__":
    main()