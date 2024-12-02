import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Dict, Union
from pathlib import Path

# Base directory for utils.py
BASE_DIR = Path(__file__).resolve().parent

# Paths for saving results
FIGURES_DIR = BASE_DIR.parent / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def get_imagenet_idx_to_class() -> Dict[int, str]:
    import requests
    import json
    
    # URL for the ImageNet class-to-index mapping JSON file
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

    # Send a GET request to download the file
    response = requests.get(url)
    
    # Check for successful request
    response.raise_for_status()

    # Parse the JSON content
    class_idx = response.json()

    # Invert the mapping to get index-to-class
    idx_to_class = {int(k): v[1].lower() for k, v in class_idx.items()}

    return idx_to_class

def normalize_images(images: torch.Tensor, to_numpy: bool = True) -> torch.Tensor | np.ndarray:
    """
    Normalizes a Torch image or a batch of images to be in the range [0, 1] 
    and optionally converts to a NumPy array.

    :param images: Torch tensor of shape (batch, channels, height, width) or (channels, height, width).
    :param to_numpy: Whether to convert the result to a NumPy array. Defaults to True.
    :returns: Normalized images as a Torch tensor or a NumPy array.
    """
    if not torch.is_tensor(images):
        raise TypeError("Input must be a Torch tensor.")
    
    if images.dim() not in [3, 4]:
        raise ValueError("Input tensor must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

    # Normalize each image in the batch
    if images.dim() == 4:  # Batch of images
        images = images - images.view(images.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
        images = images / images.view(images.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    else:  # Single image
        images = images - images.min()
        images = images / images.max()

    # Permute dimensions to (batch, height, width, channels) for compatibility
    if images.dim() == 4:  # Batch of images
        images = images.permute(0, 2, 3, 1)  # (B, H, W, C)
    elif images.dim() == 3:  # Single image
        images = images.permute(1, 2, 0)  # (H, W, C)

    if to_numpy:
        return images.cpu().numpy()  # Ensure conversion is performed on CPU for efficiency
    else:
        return images
    
def show_images(
    images: torch.Tensor,
    labels: Union[list[str], None] = None,
    correct_match: Union[list[bool], None] = None, 
    save_fig: bool = False,
    filename: str = "images",
):
    """
    Displays a batch of images in a grid, optionally with labels and match indicators.

    :param images: Tensor of images with shape (batch_size, channels, height, width).
    :param labels: List of labels for each image, or None if no labels are provided.
    :param correct_match: List of booleans indicating match correctness, or None.
    """
    images = images.float()

    # Ensure the image tensor is in the correct shape
    if images.size(3) == 3:  # If channel is last, move it to first
        images = images.permute(0, 3, 1, 2)

    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True)
    npimg = grid_img.permute(1, 2, 0).numpy()  # Rearrange to (H, W, C) format

    # Plot the images
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(npimg)
    ax.axis("off")

    if labels is not None:
        n_images = images.size(0)  # Number of images in the batch
        nrow = 8  # Number of images per row in the grid
        img_size = images.size(2) + 2  # Adjust based on padding

        for i in range(n_images):
            row = i // nrow
            col = i % nrow
            x_pos = col * img_size + img_size // 2 + 1
            y_pos = row * img_size

            label_text = labels[i] if labels else "No Label"
            color = (
                "white" if correct_match is None
                else ("green" if correct_match[i] else "red")
            )

            # Display label above each image
            ax.text(
                x_pos, y_pos, label_text,
                color=color,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
            )

    plt.tight_layout()
    filepath = FIGURES_DIR / f"{filename}.png"
    plt.savefig(filepath)
    plt.close()
    

def download_imagenette():
    """
    Downloads and extracts the Imagenette dataset safely.
    """
    
    import requests
    import tarfile
    
    # URL for the 320px version of Imagenette
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    filename = "imagenette2-320.tgz"
    extract_dir = "imagenette2-320"

    if os.path.exists(extract_dir):
        print(f"{extract_dir} already exists. Skipping download.")
        return

    # Download the dataset
    print("Downloading Imagenette...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    # Safely extract the dataset
    print("Extracting Imagenette...")
    with tarfile.open(filename, "r:gz") as tar:
        # Check for directory traversal
        for member in tar.getmembers():
            member_path = os.path.join(".", member.name)
            abs_directory = os.path.abspath(".")
            abs_target = os.path.abspath(member_path)
            if not os.path.commonpath([abs_directory, abs_target]) == abs_directory:
                raise Exception(f"Attempted Path Traversal in Tar File: {member.name}")
        
        # Extract files
        tar.extractall(path=".")

    # Remove the tar file after extraction
    os.remove(filename)
    print("Imagenette is ready!")


def load_imagenette():

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load train and validation datasets
    train_dataset = datasets.ImageFolder(root='imagenette2-320/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='imagenette2-320/val', transform=transform)

    # WordNet ID (WNID) to human-readable class name mapping for ImageNet
    wnid_to_class = {
        "n01440764": "tench",
        "n02102040": "english_springer",
        "n02979186": "cassette_player",
        "n03000684": "chain_saw",
        "n03028079": "church",
        "n03394916": "french_horn",
        "n03417042": "garbage_truck",
        "n03425413": "gas_pump",
        "n03445777": "golf_ball",
        "n03888257": "parachute",
    }

    # Access the class-to-index mapping
    class_to_idx = train_dataset.class_to_idx

    # Transform WNIDs to human-readable labels
    train_dataset.class_to_idx = {wnid_to_class[wnid]: idx for wnid, idx in class_to_idx.items()}

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader

def plot_images(images):
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.show()

def plot_class_examples(dataloader, save_fig=False, filename="class_examples"):
    # Extract the class-to-index mapping
    class_to_idx = dataloader.dataset.class_to_idx
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    # Dictionary to store one example per class
    class_examples = {class_name: None for class_name in class_names}

    # Iterate through the DataLoader to find one example per class
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            class_name = idx_to_class[label.item()]  # Get class name
            # Store the first example of each class
            if class_examples[class_name] is None:
                # Convert image to channel-last format (H, W, C)
                image_np = img.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Scale to [0, 1]
                class_examples[class_name] = image_np
            
            # Stop iteration if all classes have been captured
            if all(example is not None for example in class_examples.values()):
                break
        if all(example is not None for example in class_examples.values()):
            break

    # Plot the examples
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 3))
    for ax, (class_name, image) in zip(axes, class_examples.items()):
        ax.imshow(image)  # Plot the image
        ax.set_title(f"{class_to_idx[class_name]}: {class_name}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    filepath = FIGURES_DIR / f"{filename}.png"
    plt.savefig(filepath)
    plt.close()