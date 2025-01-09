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


def gaussian_perturbation_linspace(image, magnitude, n, seed=None):
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


def blur_perturbation_linspace(image, magnitude, n, epsilon=1e-3):
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


def occlusion_perturbation_linspace(image, magnitude, n, fill_value=0):
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


def void_perturbation_linspace(image, magnitude, n, fill_value=0):
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


def inv_grad_perturbation(image, model, i_class):
    """
    Perturbs an image in the direction of the gradient of the output w.r.t the input

    Args:
        image (Tensor): Image of shape (C, H, W)
        model (nn.Module): A PyTorch model
        i_class (int): Index of the class to compute the gradient w.r.t the input

    Returns:
        Tensor: Gradient of shape (C, H, W)
    """

    input_tensor = image.clone()
    input_tensor.requires_grad = True
    input_tensor.retain_grad()

    # Unsqueeze the tensor before feeding it to the model
    input_tensor_for_model = input_tensor.unsqueeze(0)
    output = model(input_tensor_for_model)

    # Compute gradients of the output with respect to the input
    output_class = output[0, i_class]
    output_class.backward()
    grad = input_tensor.grad

    return grad


def inv_grad_perturbation_linspace(image, model, i_class, magnitude, n):
    """
    Creates a linspace of images between image and image + noise

    Args:
        image (Tensor): Image of shape (C, H, W)
        model (nn.Module): A PyTorch model
        i_class (int): Index of the class to compute the gradient w.r.t the input
        magnitude (float): Maximum perturbation magnitude
        n (int): Number of samples to take

    Returns:
        Tensor: Image of shape (n, C, H, W)
    """
    grad = inv_grad_perturbation(image, model, i_class)
    noise = grad * magnitude
    return image_linspace(image, noise, n)


def overlay_pattern(image, pattern, magnitude):
    """
    Overlay a pattern on an image

    Args:
        image (Tensor): Image of shape (C, H, W)
        pattern (Tensor): Pattern of smaller shape
        magnitude (float): Magnitude of the pattern

    Returns:
        Tensor: Image of shape (C, H, W)
    """

    # repeat periodically the pattern until it reaches the image size
    c, h, w = image.shape
    n_x = h // pattern.shape[1] + 1
    n_y = w // pattern.shape[2] + 1
    pattern_2 = pattern.repeat(1, n_x, n_y)
    pattern_2 = pattern_2[:, :h, :w]

    # overlay the pattern
    return image + magnitude * pattern_2
