import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import tarfile
import requests
import re
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import alexnet, AlexNet_Weights
from safetensors.torch import save_file, load_file
from typing import Dict, Union, List, Optional, Tuple, Callable
from PIL import Image
from xai_rai_units.src.paths import FIGURES_DIR, DATASETS_DIR, IMAGE_DIR, MODELS_DIR
from pytorch_grad_cam.utils.image import show_cam_on_image


# URL for the ImageNet class-to-index mapping JSON file
IMAGENET_CLASS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

def download_imagenet_mapping(data_dir: str = DATASETS_DIR, url: str = IMAGENET_CLASS_URL) -> str:
    """
    Downloads the ImageNet class-to-index mapping JSON file if not already present.

    Args:
        data_dir (str): Directory to store the mapping file.
        url (str): URL of the ImageNet class-to-index JSON file.

    Returns:
        str: File path to the JSON file in the data directory.
    """
    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Define the file path for the mapping
    data_file = os.path.join(data_dir, "imagenet_class_index.json")

    # Check if the file already exists
    if not os.path.exists(data_file):
        # Download the mapping and save it to the data directory
        response = requests.get(url)
        response.raise_for_status()

        # Save the JSON content to the file
        with open(data_file, "w") as f:
            json.dump(response.json(), f)
        print(f"Downloaded and saved ImageNet class mapping to {data_file}")

    return data_file


def get_imagenet_class_to_idx(data_dir: str = DATASETS_DIR, url: str = IMAGENET_CLASS_URL) -> Dict[str, int]:
    """
    Loads the class-to-index mapping from the ImageNet mapping JSON file.

    Args:
        data_dir (str): Directory where the mapping file is stored.
        url (str): URL of the ImageNet class-to-index JSON file.

    Returns:
        Dict[str, int]: Mapping of ImageNet class names to indices.
    """
    # Download and/or locate the file
    data_file = download_imagenet_mapping(data_dir, url)

    # Load the mapping from the file
    with open(data_file, "r") as f:
        class_idx = json.load(f)

    # Extract class-to-index mapping
    class_to_idx = {v[1].lower(): int(k) for k, v in class_idx.items()}
    return class_to_idx


def get_imagenet_idx_to_class(data_dir: str = DATASETS_DIR, url: str = IMAGENET_CLASS_URL) -> Dict[int, str]:
    """
    Loads the index-to-class mapping from the ImageNet mapping JSON file.

    Args:
        data_dir (str): Directory where the mapping file is stored.
        url (str): URL of the ImageNet class-to-index JSON file.

    Returns:
        Dict[int, str]: Mapping of ImageNet indices to class names.
    """
    # Download and/or locate the file
    data_file = download_imagenet_mapping(data_dir, url)

    # Load the mapping from the file
    with open(data_file, "r") as f:
        class_idx = json.load(f)

    # Extract index-to-class mapping
    idx_to_class = {int(k): v[1].lower() for k, v in class_idx.items()}
    return idx_to_class

# Get the mappings
CLASS_TO_IDX_IMAGENET = get_imagenet_class_to_idx()
IDX_TO_CLASS_IMAGENET = get_imagenet_idx_to_class()


def get_class_to_idx_imagenette() -> Dict[str, int]:
    """
    # TODO comment input-output
    """
    return {
        'tench': 0,
        'english_springer': 1,
        'cassette_player': 2,
        'chain_saw': 3,
        'church': 4,
        'french_horn': 5,
        'garbage_truck': 6,
        'gas_pump': 7,
        'golf_ball': 8,
        'parachute': 9
    }


def transform_imagenette_to_imagenet_indices(imagenette_indices: torch.Tensor) -> torch.Tensor:
    """
    # TODO comment input-output
    """
    # Mapping of ImageNet class names to their corresponding indices
    imagenette_class_to_idx = get_class_to_idx_imagenette()
    imagenette_idx_to_class = {v: k for k, v in imagenette_class_to_idx.items()}

    imagenet_indices = [CLASS_TO_IDX_IMAGENET[imagenette_idx_to_class[int(idx.item())]] for idx in imagenette_indices]

    return torch.tensor(imagenet_indices)


def normalize_images(images: torch.Tensor) -> np.ndarray:
    """
    Normalizes a Torch image or a batch of images to be in the range [0, 1] 
    and optionally converts to a NumPy array.

    :param images: Torch tensor of shape (batch, channels, height, width) or (channels, height, width).
    :returns: Normalized images as a Torch tensor or a NumPy array of shape (batch, height, width, channels).
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

    return images.cpu().numpy()  # Ensure conversion is performed on CPU for efficiency


def show_images(images: torch.Tensor,
                labels: Optional[Union[List[str], torch.Tensor]] = None,
                correct_match: Union[list[bool], None] = None,
                save_fig: bool = False,
                filename: str = "images"):
    """
    Displays a batch of images in a grid, optionally with labels and match indicators.

    :param images: Tensor of images with shape (batch_size, channels, height, width).
    :param labels: List of labels for each image, or None if no labels are provided.
    :param correct_match: List of booleans indicating match correctness, or None.
    :param save_fig: Whether to save the figure as an image file. Defaults to False.
    :param filename: Name of the file to save the figure as. Defaults to "images".
    """
    images = images.float()

    # Ensure the image tensor is in the correct shape
    if images.size(3) == 3:  # If channel is last, move it to first
        images = images.permute(0, 3, 1, 2)

    # Normalize and create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True)
    npimg = grid_img.permute(1, 2, 0).numpy()  # Rearrange to (H, W, C) format

    # Normalize labels
    if isinstance(labels, torch.Tensor):
        labels = [str(label.item()) for label in labels]
    elif labels is None:
        labels = []

    # Plot the images
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(npimg)
    ax.axis("off")

    if labels:
        n_images = images.size(0)  # Number of images in the batch
        nrow = 8  # Number of images per row in the grid
        img_size = images.size(2) + 2  # Adjust based on padding

        for i in range(n_images):
            row = i // nrow
            col = i % nrow
            x_pos = col * img_size + img_size // 2 + 1
            y_pos = row * img_size

            label_text = labels[i] if i < len(labels) else "No Label"
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
    if save_fig:
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def download_imagenette():
    """
    Downloads and extracts the Imagenette dataset safely to the designated data directory.
    """

    # URL for the 320px version of Imagenette
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    filename = DATASETS_DIR / "imagenette2-320.tgz"
    extract_dir = DATASETS_DIR / "imagenette2-320"

    if extract_dir.exists():
        print(f"{extract_dir} already exists. Skipping download.")
        return

    # Download the dataset
    print("Downloading Imagenette...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    # Save the tar file to the datasets directory
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    # Safely extract the dataset to DATASETS_DIR
    print("Extracting Imagenette...")
    with tarfile.open(filename, "r:gz") as tar:
        # Check for directory traversal
        for member in tar.getmembers():
            member_path = os.path.join(DATASETS_DIR, member.name)
            abs_directory = os.path.abspath(DATASETS_DIR)
            abs_target = os.path.abspath(member_path)
            if not os.path.commonpath([abs_directory, abs_target]) == abs_directory:
                raise Exception(f"Attempted Path Traversal in Tar File: {member.name}")

        # Extract files
        tar.extractall(path=DATASETS_DIR)

    # Remove the tar file after extraction
    filename.unlink()
    print("Imagenette is ready!")


def load_imagenette():
    """
    # TODO comment input-output + consider adding wnid_to_class to file
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load train and validation datasets
    train_dataset = datasets.ImageFolder(root=f"{str(DATASETS_DIR)}/imagenette2-320/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{str(DATASETS_DIR)}/imagenette2-320/val", transform=transform)

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
    """
    # TODO comment input
    """
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2)
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis("off")
    plt.show()


def plot_class_examples(dataloader: DataLoader,
                        class_to_idx: Dict[str, int],
                        save_fig: bool = False,
                        filename: str = "class_examples"
                        ):
    """
    # TODO comment input
    """
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
        filepath = FIGURES_DIR / f"{filename}.png"
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close()


def load_local_images(filename: Union[str, List[str]], img_size: int = 224) -> torch.Tensor:
    """
    Loads and preprocesses one or multiple images for model inference. You can provide just the filename without 
    the extension, and the function will locate the file in the specified directory (IMAGE_DIR).

    :param filename: Filename or a list of filenames without extensions to search for in IMAGE_DIR.
    :param img_size: The target size for resizing the images.
    :return: Preprocessed image tensor(s) with batch dimension (batch_size, 3, img_size, img_size).
    """
    # Define the transformations
    transforms_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize to img_size x img_size
        transforms.ToTensor(),  # Convert image to tensor
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
        #     std=[0.229, 0.224, 0.225],  # Normalize using ImageNet std
        # ),
    ])

    # Ensure filename is a list
    if isinstance(filename, str):
        filename = [filename]

    images = []
    for fname in filename:
        # Search for the file in IMAGE_DIR with common image extensions
        possible_extensions = ['.jpg', '.jpeg', '.png']

        for ext in possible_extensions:
            image_path = os.path.join(IMAGE_DIR, fname + ext)
            if os.path.exists(image_path):
                break
        else:
            raise FileNotFoundError(f"Image file '{fname}' with any of the extensions "
                                    f"{possible_extensions} not found in directory {IMAGE_DIR}")

        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        images.append(transforms_pipeline(image))

    # Stack all image tensors to form a batch
    return torch.stack(images)  # Shape: [batch_size, 3, img_size, img_size]


def set_seed(seed):
    """Sets the seed for Python, NumPy, and PyTorch to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_classes_proportion_dataloder(loader):
    """
    # TODO comment input-output
    """
    labels = np.concatenate([y.numpy() for _, y in loader])
    unique, counts = np.unique(labels, return_counts=True)
    return unique, np.round(counts / len(labels), 2)


def download_and_save_model_weights(model_name, model, weights_path, verbose: bool = False):
    """
    Downloads and saves the model weights to the specified path in SafeTensors format.
    
    Parameters:
    - model_name: str, the name of the model.
    - model: nn.Module, the model instance.
    - weights_path: str, the path to save the weights.
    """
    state_dict = model.state_dict()
    save_file(state_dict, weights_path)
    if verbose:
        print(f"Weights for {model_name} saved to {weights_path}.")


def load_model(model_name, verbose: bool =False):
    """
    Loads a pretrained model, downloading weights if not already stored locally.
    
    Parameters:
    - model_name: str, the name of the model to load.
    
    Returns:
    - model: nn.Module, the loaded model instance.
    """
    weights_path = os.path.join(MODELS_DIR, f"{model_name}_weights.safetensors")
    
    if os.path.exists(weights_path):
        # If weights already exist, load them using SafeTensors
        if verbose:
            print(f"Loading {model_name} weights from {weights_path}.")
        if model_name == "alexnet":
            model = alexnet()
        elif model_name == "resnet50":
            model = resnet50()
        elif model_name == "swin_transformer":
            model = timm.create_model("swin_base_patch4_window7_224", pretrained=False)
        elif model_name == "vit":
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
    else:
        # Download weights and save them in SafeTensors format
        print(f"Downloading weights for {model_name}.")
        if model_name == "alexnet":
            model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == "swin_transformer":
            model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        elif model_name == "vit":
            model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        download_and_save_model_weights(model_name, model, weights_path, verbose=verbose)
    
    return model


def preprocess_class_label(class_filename: str):

    """Preprocesses and validates the ImageNet class label.
    
    :param class_filename: The ImageNet class label to preprocess when loaded from a filename.
    """
    # Convert to lowercase
    class_filename = class_filename.lower()
    # Remove file extension
    class_filename = class_filename.split(".")[0]
    # Remove trailing underscores with numbers
    class_filename = re.sub(r'_\d+$', '', class_filename)
    
    if class_filename not in CLASS_TO_IDX_IMAGENET:
        raise ValueError(f"Invalid class label '{class_filename}'. Please provide a valid ImageNet class label.")
    return CLASS_TO_IDX_IMAGENET[class_filename]


def imagenet_class_prediction(images: torch.Tensor, model_name: str = 'resnet50') -> List[str]:
    """
    Predicts ImageNet class labels for a batch of images using a specified model.
    
    :param images: torch.Tensor, A batch of preprocessed image tensors with shape (N, C, H, W) 
        or a single image tensor with shape (C, H, W). If a single image is provided, 
        it will automatically add a batch dimension.
    :param model_name: str, The name of the model to use for prediction (e.g., "resnet50").
    :return: List[str], A list of predicted class labels for the input images.
    :raises ValueError: If the provided model name is not supported or if input tensor shape is invalid.
    """
    # Add a batch dimension if only one image is provided
    if len(images.shape) == 3:  # Single image (C, H, W)
        images = images.unsqueeze(0)  # Convert to (1, C, H, W)
    elif len(images.shape) != 4:  # Invalid shape
        raise ValueError("Input `images` must be a 4D tensor (N, C, H, W) or a single image (C, H, W).")
    
    # Load the specified model
    model = load_model(model_name)
    model.eval()
    
    # Perform prediction
    with torch.no_grad():
        outputs = model(images)  # Forward pass
        predictions = outputs.argmax(dim=1)  # Get indices of max logits
    
    # Map indices to class labels
    predicted_labels = [IDX_TO_CLASS_IMAGENET[idx.item()] for idx in predictions]
    
    return predicted_labels


def reshape_transform_swin_transformer(tensor, height=7, width=7):
    """
    Transforms the tensor to the desired shape for Vision Transformer models.
    """
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_vit(tensor, height=14, width=14):
    """
    Transforms the tensor to the desired shape for Vision Transformer models.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def overlay_heatmaps(attr: np.ndarray, perturbed_images: torch.Tensor) -> torch.Tensor:
    """Generates overlayed heatmaps on normalized images."""
    # Normalize input images to range [0, 1] and convert (B, C, H, W) to (B, H, W, C)
    imgs_normalized = normalize_images(perturbed_images)
    # Overlay heatmaps onto normalized images provided proper dimensions
    return torch.Tensor(
        np.array([show_cam_on_image(img, cam, use_rgb=True) for img, cam in zip(imgs_normalized, attr)])
    )


def setup_model_and_layers(model_name: str) -> Tuple[nn.Module, List[nn.Module], Optional[Callable]]:
    """
    Configures the model, selects target layers, and assigns reshape transformations based on the model architecture.

    This function loads the specified model and selects the appropriate target layers for explanation methods 
    (e.g., Grad-CAM, Captum). It also provides a reshape transformation if needed for transformer-based models 
    (like ViT or Swin-Transformer). A `ValueError` is raised if the model is unsupported.

    Args:
        model_name (str): The model architecture to configure. Options include: 'alexnet', 'resnet50', 
                          'swin_transformer', and 'vit'.
    
    Returns:
        Tuple[nn.Module, List[nn.Module], Optional[Callable]]:
            - The model (nn.Module) corresponding to the specified `model_name`.
            - A list of target layers (List[nn.Module]) for gradient-based methods.
            - An optional reshape transformation function (Callable) for models like ViT and Swin-Transformer.
    
    Raises:
        ValueError: If the `model_name` is unsupported.
    """
    model = load_model(model_name)

    reshape_transform, target_layers = None, []

    # Select the target layer based on the model architecture
    if model_name == "alexnet":
        target_layers = [model.features[10]]
    elif model_name == "resnet50":
        target_layers = [model.layer4[-1].conv3]
    elif model_name == "swin_transformer":
        target_layers = [model.layers[-1].blocks[-1].norm2]
        reshape_transform = reshape_transform_swin_transformer
    elif model_name == "vit":
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = reshape_transform_vit
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, target_layers, reshape_transform


def mean_square_error_difference(heatmap1, heatmap2):
    """Computes the mean squared error between two heatmaps."""
    return np.mean((heatmap1 - heatmap2) ** 2)


def heatmap_entropy(heatmap, epsilon=1e-10):
    """Computes the entropy of a heatmap."""
    return -np.mean(heatmap * np.log(heatmap + epsilon))


def heatmap_barycenter(heatmap):
    """Computes the center of the probability mass in a heatmap."""
    h, w = heatmap.shape
    y, x = np.mgrid[:h, :w]
    return np.sum(x * heatmap), np.sum(y * heatmap)


def heatmap_dispersion(heatmap):
    """Computes the dispersion (std) of the probability mass in a heatmap."""
    h, w = heatmap.shape
    y, x = np.mgrid[:h, :w]
    mx, my = heatmap_barycenter(heatmap)
    var_x = np.mean((x - mx) ** 2 * heatmap)
    var_y = np.mean((y - my) ** 2 * heatmap)
    return np.sqrt(var_x + var_y)


def sample_filenames(directory: str = IMAGE_DIR, n: int = 5) -> List[str]:
    """
    Randomly samples n filenames (without extensions) from the specified directory.

    Args:
        directory (str): Path to the directory containing images.
        n (int): Number of filenames to sample.

    Returns:
        List[str]: A list of randomly sampled filenames without extensions.
    """
    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out directories (only keep files)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(directory, f))]

    # Randomly sample n files
    sampled_files = random.sample(all_files, n)

    # Remove extensions from filenames
    sampled_filenames = [os.path.splitext(f)[0] for f in sampled_files]

    return sampled_filenames

