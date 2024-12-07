import torch
from torchvision.transforms import GaussianBlur


def image_linspace(image, noise, n=100):
    """
    Creates a linspace of images between image and image and image + a given noise
    image: a tensor of shape (n, C, H, W)
    noise: a tensor of the same shape
    n: number of samples to take
    return: a tensor of shape (n, C, H, W)
    """
    t = torch.linspace(0, 1, n).view(-1, 1, 1, 1)
    segment = image + t * noise
    segment = torch.clamp(segment, min=0, max=1)
    return segment


def noisy_image_linspace(image, magnitude, n, seed=None):
    """
    Creates a linspace of images between image and image + noise
    :param image: a tensor of shape (n, C, H, W)
    :param magnitude: maximum perturbation magnitude
    :param n: number of samples to take
    :param seed: random seed
    :return: a tensor of shape (n, C, H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # choose a random direction
    noise = torch.randn_like(image) * magnitude

    return image_linspace(image, noise, n)


def blur_image_linspace(image, magnitude, n, epsilon=1e-3):
    """
    Creates a linspace of increasingly blurred images
    :param image: a tensor of shape (n, C, H, W)
    :param magnitude: maximum blur magnitude (number of pixels)
    :param n: number of samples to take
    :param epsilon: minimum standard deviation (must be positive)
    :return: a tensor of shape (n, C, H, W)
    """
    kernel_size = int(2 * int(magnitude) + 1)
    sigma_values = torch.linspace(epsilon, magnitude, n, device=image.device)

    out = []
    for sigma in sigma_values:
        gaussian_blur = GaussianBlur(kernel_size, sigma=sigma.item())
        out.append(gaussian_blur(image))

    return torch.stack(out)
