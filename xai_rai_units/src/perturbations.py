import torch
from torchvision.transforms import GaussianBlur


def image_linspace(image, noise, n=100):
    """
    Creates a linspace of images between image and image and image + a given noise

    Args:
        image (Tensor): Image of shape (n, C, H, W)
        noise (Tensor): Image of the same shape
        n (int): Number of samples to take

    Returns:
        Tensor: Image of shape (n, C, H, W)
    """
    t = torch.linspace(0, 1, n).view(-1, 1, 1, 1)
    segment = image + t * noise
    segment = torch.clamp(segment, min=0, max=1)
    return segment


def noisy_image_linspace(image, magnitude, n, seed=None):
    """
    Creates a linspace of images between image and image + noise

    Args:
        image (Tensor): A tensor of shape (n, C, H, W)
        magnitude (float): Maximum perturbation magnitude
        n (int): Number of samples to take
        seed (int or None): Random seed

    Returns:
        Tensor: Image of shape (n, C, H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # choose a random direction
    noise = torch.randn_like(image) * magnitude

    return image_linspace(image, noise, n)


def blur_image_linspace(image, magnitude, n, epsilon=1e-3):
    """
    Creates a linspace of increasingly blurred images

    Args:
        image (Tensor): A tensor of shape (n, C, H, W)
        magnitude (float): Maximum blur magnitude (number of pixels)
        n (int): Number of samples to take
        epsilon (float): Minimum standard deviation (must be positive)

    Returns:
        Tensor: Image of shape (n, C, H, W)
    """
    kernel_size = int(2 * int(magnitude) + 1)
    sigma_values = torch.linspace(epsilon, magnitude, n, device=image.device)

    out = []
    for sigma in sigma_values:
        gaussian_blur = GaussianBlur(kernel_size, sigma=sigma.item())
        out.append(gaussian_blur(image))

    return torch.stack(out)


def occlusion_linspace(image, magnitude, n, fill_value=0):
    """
    Creates a linspace of increasingly occluded images

    Args:
        image (Tensor): A tensor of shape (n, C, H, W)
        magnitude (float): Maximum size of the occlusion (number of pixels)
        n (int): Number of samples to take
        fill_value (int): Value to fill the occlusion with

    Return:
        Tensor: Image of shape (n, C, H, W)
    """

    # choose a random occlusion region
    _, H, W = image.shape
    x1 = torch.linspace(W, W - magnitude, n, device=image.device).int() // 2
    y1 = torch.linspace(H, H - magnitude, n, device=image.device).int() // 2
    x2 = torch.linspace(W, W + magnitude, n, device=image.device).int() // 2
    y2 = torch.linspace(H, H + magnitude, n, device=image.device).int() // 2
    out = []
    for i in range(n):
        occluded = image.clone()
        occluded[:, y1[i]:y2[i], x1[i]:x2[i]] = fill_value
        out.append(occluded)

    return torch.stack(out)


def gaussian_kernel(kernel_size, sigma):
    """
    Creates a 2D Gaussian kernel.

    Args:
        kernel_size (int): The size of the kernel.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        Tensor: A 2D Gaussian kernel.
    """
    x = torch.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
    y = x.unsqueeze(1)

    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return kernel


def the_void_linspace(image, magnitude, n, fill_value=0):
    """
    Creates a linspace of increasingly voided images

    Args:
        image (Tensor): A tensor of shape (n, C, H, W)
        magnitude (float): Maximum size of the void (number of pixels)
        n (int): Number of samples to take
        fill_value (int): Value to fill the void with

    Returns:
        Tensor: Image of shape (n, C, H, W)
    """
    _, H, W = image.shape
    x = torch.linspace(-1, 1, W, device=image.device)
    y = torch.linspace(-1, 1, H, device=image.device)
    y = y.unsqueeze(1)

    out = []
    for i in range(n):
        k = magnitude * (i / n)**2
        mask = torch.exp(-(x ** 2 + y ** 2) * k)
        out.append(image * mask + fill_value * (1 - mask))

    return torch.stack(out)
