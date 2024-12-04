import numpy as np
import torch
from utils import noisy_image_linspace


class MeasureProba:
    """
    Given a model and an image, this class will return
    a list of probabilities of the right class, by running
    the model on a segment (in input space 'X') that joins
    the image and a point in a random direction relative
    to the image.
    """
    def __init__(self, model):
        self.model = model

    def run(self, image, magnitude=1., n=100, seed=None):
        """
        image: a tensor of shape (3, 224, 224)
        n: number of samples to take
        """
        segment = noisy_image_linspace(image, magnitude, n, seed=seed)

        # run the model on the segment
        with torch.no_grad():
            logits = self.model(segment)
            i_max = np.argmax(logits[0])
            p = torch.softmax(logits, dim=1)
            target = segment[-1]
            return np.array(p[:, i_max]), i_max, target

    def run_multiple(self, image, n_segments, magnitude=1., n=100):
        out = []
        for _ in range(n_segments):
            p, i_max, target = self.run(image, magnitude=magnitude, n=n, seed=None)
            out.append(p)
        return np.array(out)
