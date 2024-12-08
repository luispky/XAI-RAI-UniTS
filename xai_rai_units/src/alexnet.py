import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def download_alexnet(model_path):
    """
    A function to download and save the AlexNet model. If the model
    already exists, it will be loaded from the disk.

    Args:
        model_path (str): path to save the model

    Returns:
        model: an AlexNet model
    """
    if not os.path.exists(model_path):
        model = torchvision.models.alexnet(weights=True)
        torch.save(model.state_dict(), model_path)
    else:
        print(">>> Model already downloaded.")
        model = torchvision.models.alexnet()
        model.load_state_dict(torch.load(model_path))
    return model


def load_labels(label_path):
    """
    Load class labels from file.

    Args:
        label_path (str): path to the file containing class names

    Returns:
        tuple: tuple of class names
    """
    with open(label_path, 'r') as f:
        class_names = f.readlines()
    return tuple(name.strip() for name in class_names)


def transformation(resize=224, size=224):
    """
    Transformation to preprocess input image
    to make it compatible with AlexNet.

    Args:
        resize (int): size to resize the image
        size (int): size to crop the image

    Returns:
        torchvision.transforms.Compose: a sequence of transformations
    """
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ])


def load_images(image_dir: str, n=16):
    """
    Load and preprocess images from a directory.

    Args:
        image_dir (str): path to the directory containing images
        n (int): number of images to load

    Returns:
        torch.Tensor: a tensor of shape (n, 3, 224, 224)
    """
    images = []
    filenames = np.array(list(os.listdir(image_dir)))
    indices = np.random.choice(len(filenames), n, replace=False)
    filenames = filenames[indices]

    for i, filename in enumerate(filenames):
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)

        # remove alpha channel
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        print(f"\r>>> Loading image ({i+1}/{len(filenames)}): {img_path} size: {img.size} {img.mode}", end='   ')

        img_tensor = transformation()(img)
        images.append(img_tensor)
    print()

    return torch.stack(images)


def predict(model, images):
    """
    Make predictions on images using the model.

    Args:
        model: a PyTorch model
        images (torch.Tensor): a tensor of shape (n, 3, 224, 224)

    Returns:
        tuple: a tuple of tensors (outputs, predicted)
    """
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    return outputs, predicted


def visualize(outputs, predicted, images, labels):
    """
    A function to make predictions and visualize results.

    Args:
        outputs (torch.Tensor): a tensor of shape (n, num_classes)
        predicted (torch.Tensor): a tensor of shape (n,)
        images (torch.Tensor): a tensor of shape (n, 3, 224, 224)
        labels (tuple): a tuple of class names
    """
    n = len(images)
    nrows = int(n ** 0.5)
    rows = np.arange(n) % nrows
    cols = np.arange(n) // nrows
    ncols = int(cols[-1]) + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))

    for i, (x, y) in enumerate(zip(rows, cols)):
        ax = axes[x, y]
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        predicted_class = labels[predicted[i]]
        proba = outputs.data[i].max().item()
        ax.set_title(f"{predicted_class} ({proba:.2f}%)")
        ax.axis('off')
    plt.show()
