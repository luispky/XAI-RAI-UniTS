import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def download_alexnet(model_path):
    """A function to download and save the AlexNet model"""
    if not os.path.exists(model_path):
        model = torchvision.models.alexnet(weights=True)
        torch.save(model.state_dict(), model_path)
    else:
        print(">>> Model already downloaded.")
        model = torchvision.models.alexnet()
        model.load_state_dict(torch.load(model_path))
    return model


def load_labels(label_path='imagenet_classes.txt'):
    """Load class labels from file"""
    with open(label_path, 'r') as f:
        class_names = f.readlines()
    return tuple(name.strip() for name in class_names)


def transformation(resize=256, size=224,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225),
                   ):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),

        transforms.Lambda(lambda x: x.clamp(0, 1))
    ])


def load_images(image_dir, n=16):
    """Load and preprocess images"""

    images = []
    filenames = np.array(list(os.listdir(image_dir)))
    indices = np.random.choice(len(filenames), n, replace=False)
    filenames = filenames[indices]

    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = Image.open(img_path)

        # remove alpha channel
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        print(f">>> Loading image: {img_path} size: {img.size} {img.mode}")

        img_tensor = transformation()(img)
        images.append(img_tensor)

    return torch.stack(images)


def predict(model, images):
    """Make predictions on images using the model"""
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    return outputs, predicted


def visualize(outputs, predicted, images, labels):
    """A function to make predictions and visualize results"""
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


def main(model_path="models\\alexnet_weights.pth",
         image_dir="data\\images"):

    print(">>> Downloading the model...")
    model = download_alexnet(model_path)

    print(">>> Loading images...")
    images = load_images(image_dir, n=16)
    labels = load_labels()

    print(">>> Making predictions...")
    outputs, predicted = predict(model, images)
    visualize(outputs, predicted, images, labels)


if __name__ == '__main__':
    main()
