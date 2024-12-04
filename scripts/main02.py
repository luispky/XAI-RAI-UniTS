from email.mime import image
import sys
from torchvision.models import resnet50, ResNet50_Weights

from pathlib import Path

# Add src directory to Python path
PARENT_DIRECTORY = str(Path(__file__).resolve().parent.parent)
sys.path.append(PARENT_DIRECTORY + "/src")

from gradcam_explanations import gradcam_explanations_classifier_series
from gradcam_explanations import CLASS_TO_IDX_IMAGENET, IDX_TO_CLASS_IMAGENET
from utils import load_local_images, show_images, generate_noisy_images
from utils import set_seed
from alexnet import download_alexnet
from paths import MODELS_DIR

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Load test image
    image_path = PARENT_DIRECTORY + "/test_images/llama.jpeg"
    preprocessed_image = load_local_images(image_path)
    class_label = "llama"

    # Generate a sequence of noisy images
    n_images = 16
    magnitude = 0.5
    noisy_images = generate_noisy_images(preprocessed_image, n_images, magnitude)

    # show_images(noisy_images, save_fig=True, filename="noisy_llama.png")

    # Load AlexNet model
    # model_path = f"{str(MODELS_DIR)}/alexnet_weights.pth"
    # model = download_alexnet(model_path)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Target layers for Grad-CAM
    # target_layers = [model.features]
    target_layers = [model.layer4[-1]]
    
    img_overlay_cam, pred_labels  = gradcam_explanations_classifier_series(
        model=model,
        perturbed_images=noisy_images,
        class_label_imagenet=class_label,
        target_layers=target_layers,
        method="GradCAM",
        predicted_labels=True,
    )
    
    show_images(img_overlay_cam, labels=pred_labels, save_fig=True, filename="gradcam_llama_resnet50.png")

if __name__ == "__main__":
    main()