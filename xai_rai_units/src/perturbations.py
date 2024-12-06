import torch


def image_linspace(image, noise, n=100):
    """
    Creates a linspace of images between image and image and image + a given noise
    image: a tensor
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
    :param image: a tensor
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
