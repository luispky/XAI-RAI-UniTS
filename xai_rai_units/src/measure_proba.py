import numpy as np
import torch
from xai_rai_units.src.perturbations import gaussian_perturbation_linspace


class MeasureProba:
    """
    Given a model and an image, this class will return
    a list of probabilities of the right class, by running
    the model on a segment (in input space 'X') that joins
    the image and a point in a random direction relative
    to the image.

    The segment is defined by the weighted average of the image
    'X' and the perturbed image 'X + epsilon * U', where 'U' is
    a random direction in input space.

    Args:
        model: a PyTorch model for image classification
    """
    def __init__(self, model):
        self.model = model

    def run(self, image, magnitude=1., n=100, seed=None):
        """
        Compute how the probability of the right class for a given image
        changes as the noise in the image gets more intense.

        Args:
            image (Tensor): a tensor of shape (3, 224, 224)
            magnitude (float): maximum perturbation magnitude
            n (int): number of samples to take
            seed (int or None): random seed

        Returns:
            dict: A dictionary containing:
            - 'probabilities': A NumPy array of probabilities of the right class
            - 'max_class_index': The index of the class with the highest probability
            - 'final_image': The final image in the segment
        """
        segment = gaussian_perturbation_linspace(image, magnitude, n, seed=seed)

        with torch.no_grad():
            logits = self.model(segment)
            i_max = np.argmax(logits[0])
            p = torch.softmax(logits, dim=1)
            target = segment[-1]
            proba = np.array(p[:, i_max])
            return {
                'probabilities': proba,
                'max_class_index': i_max,
                'final_image': target
            }

    def run_multiple(self, image, n_segments, magnitude=1., n=100):
        out = []
        for _ in range(n_segments):
            results = self.run(image, magnitude=magnitude, n=n, seed=None)
            out.append(results["probabilities"])
        return np.array(out)
