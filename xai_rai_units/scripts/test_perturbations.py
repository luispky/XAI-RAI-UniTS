import matplotlib.pyplot as plt
from xai_rai_units.src import perturbations as pert
from xai_rai_units.src.paths import IMAGE_DIR
from xai_rai_units.src.alexnet import load_images


def test_noisy_image_linspace(image_dir=IMAGE_DIR, magnitude=.5, n=5):
    """Test the function noisy_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.noisy_image_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i+1}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_blur_image_linspace(image_dir=IMAGE_DIR, magnitude=8, n=5):
    """Test the function blur_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.blur_image_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_occlusion_linspace(image_dir=IMAGE_DIR, magnitude=150, n=5):
    """Test the function blur_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.occlusion_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i+1}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_the_void_linspace(image_dir=IMAGE_DIR, magnitude=10., n=5):
    """Test the function the_void_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.the_void_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i+1}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


if __name__ == '__main__':
    # test_noisy_image_linspace()
    # test_blur_image_linspace()
    # test_occlusion_linspace()
    test_the_void_linspace()
