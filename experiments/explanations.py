from locale import normalize
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def download_imagenette():
    import requests
    import tarfile

    # URL for the 320px version of Imagenette
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    filename = "imagenette2-320.tgz"

    # Download the dataset
    print("Downloading Imagenette...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    # Extract the dataset
    print("Extracting Imagenette...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()

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
    
def imshow_with_labels(images, labels):
    # Add a channel dimension if missing to make the grid
    images = images.float()

    # Make a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True)
    npimg = grid_img.permute(1, 2, 0).numpy() # Rearrange to (H, W, C) format and convert to numpy

    # Plot the images with the specified colormap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(npimg)
    ax.axis("off")

    # Add labels with emojis above each image
    n_images = images.size(0)  # Number of images in the batch
    nrow = 8  # Number of images per row in the grid
    img_size = images.size(2) + 2  # Adjust based on padding
    for i in range(n_images):
        row = i // nrow
        col = i % nrow
        x_pos = col * img_size + img_size // 2 + 1
        y_pos = row * img_size
        # Ensure emoji is properly rendered
        label_text = f"{labels[i]}"
        ax.text(
            x_pos, y_pos, label_text,
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"),
        )

    plt.show()

def plot_class_examples(dataloader, save_fig=False, filename="class_examples.png"):
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
    if save_fig:
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/{filename}", bbox_inches='tight')
    
    plt.show()

def main():

    # Download Imagenette dataset
    if not os.path.exists("imagenette2-320"):
        download_imagenette()
    
    # Load the Imagenette dataset
    train_loader, _ = load_imagenette()
    
    # Access the class-to-index mapping
    class_to_idx = train_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Display a grid of images
    images, indices = next(iter(train_loader))
    labels = [idx_to_class[idx.item()] for idx in indices]
    plot_class_examples(train_loader)
    
    # Load a pretrained ResNet-50 model
    model = resnet50(pretrained=True)
    
    target_layers = [model.layer4[-1]]
    input_tensor = images

    # We have to specify the target we want to generate the CAM for.
    targets = [ClassifierOutputTarget(idx) for idx in indices]

    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam
        visualization = show_cam_on_image(input_tensor, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs

        print("Model outputs")
        print(model_outputs)

if __name__ == "__main__":
    main()    