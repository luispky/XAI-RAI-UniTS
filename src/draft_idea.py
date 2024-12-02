"""

"""


def random_perturbations(image, magnitude=0.1, n_images=100):
    """
    Returns a series of perturbed images of the input image.

    image[i] = original image + normal random noise * magnitude * i / n_images

    :param image: torch array of shape (H, W, C)
    :param magnitude: maximum perturbation magnitude
    :param n_images: number of perturbed images to generate
    :return: list of torch arrays of shape (H, W, C)
    """
    pass


def blur_perturbation(image, magnitude=0.1, n_images=100):
    """
    Returns a series of perturbed images of the input image.
    :param image: torch array of shape (H, W, C)
    :param magnitude: maximum perturbation magnitude
    :param n_images: number of perturbed images to generate
    :return: list of torch arrays of shape (H, W, C)
    """
    pass


def classification_1(images):
    """
    Returns the class of the input image given by AlexNet.
    :param images: list of torch array of shape (H, W, C)
    :return: probability of the most likely class for each image.
    """
    pass


def explanation_1(image):
    """"""
    ...


def main():

    # Load the image
    ...

    # Generate perturbed images
    ...

    # Generate a series of classifications/explanations/counterfactuals
    ...


if __name__ == '__main__':
    main()
