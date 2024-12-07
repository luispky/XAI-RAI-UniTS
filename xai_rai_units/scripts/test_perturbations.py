import matplotlib.pyplot as plt
from xai_rai_units.src.perturbations import blur_image_linspace
from xai_rai_units.src.paths import IMAGE_DIR
from xai_rai_units.src.alexnet import load_images


def test_blur_image_linspace(image_dir=IMAGE_DIR, magnitude=10, n=5):
    """Test the function blur_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    blurred_images = blur_image_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'sigma={i/n*magnitude:.2f}')
        ax[i].imshow(blurred_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


if __name__ == '__main__':
    test_blur_image_linspace()
