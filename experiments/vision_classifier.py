from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
import json
import urllib.request

# Function to load ImageNet class labels
def load_imagenet_classes():
    # URL for ImageNet class labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = urllib.request.urlopen(url)
    class_labels = json.loads(response.read())
    return class_labels

# Function to display images with annotations
def plot_images_with_predictions(images, predictions, confidences, num_rows=2, num_cols=3, save_fig=False, fig_name="predictions.png"):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx].numpy()
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{predictions[idx]} ({confidences[idx]:.2f}%)", fontsize=10)
        else:
            ax.axis("off")
    if save_fig:
        plt.savefig(fig_name)
        
    plt.tight_layout()
    plt.show()

def main():
    # Define the data path and model input size
    data_path = './images'  # Ensure this folder has images
    img_size = 224
    
    # Define transformations
    transforms_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalize for pretrained models
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load the dataset
    dataset = ImageFolder(root=data_path, transform=transforms_pipeline)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)  # Adjust batch size if needed
    
    # Load ImageNet class labels
    imagenet_classes = load_imagenet_classes()

    # Load a batch of images
    images, _ = next(iter(dataloader))
    
    # Load the ViT-Lite model (e.g., 'vit_tiny_patch16_224')
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.eval()  # Set the model to evaluation mode
    
    # Perform inference
    with torch.no_grad():
        outputs = model(images)  # Get predictions for the batch
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
        top_predictions = torch.argmax(probabilities, dim=1)  # Get the top class indices
    
    # Map predictions to class labels and extract confidence scores
    predicted_labels = [imagenet_classes[idx] for idx in top_predictions.numpy()]
    confidence_scores = [probabilities[i, top_predictions[i]].item() * 100 for i in range(len(top_predictions))]
    
    # Plot the images with predicted labels and confidence scores
    plot_images_with_predictions(images, predicted_labels, confidence_scores, num_rows=3, num_cols=4, save_fig=True, fig_name="to_my_friend_Omaru-sensei.png")

if __name__ == '__main__':
    main()
