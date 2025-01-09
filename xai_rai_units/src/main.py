"""
The purpose of this project is to find out whether XAI
tools are robust to various image perturbations. The
project will use the AlexNet model to classify images,
perturb them with noise, occlusions, and other methods,
and then generate explanations for the classification.
"""
import numpy as np
import matplotlib.pyplot as plt
from xai_rai_units.src.paths import IMAGE_DIR, LABELS_PATH
from xai_rai_units.src.alexnet import load_images, load_labels
from xai_rai_units.src import perturbations as pert
from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src.utils import setup_model_and_layers


PERTURBATIONS = (pert.gaussian_perturbation_linspace,
                 pert.blur_perturbation_linspace,
                 pert.occlusion_perturbation_linspace,
                 pert.void_perturbation_linspace,
                 pert.inv_grad_perturbation_linspace)

MODEL_NAMES = ("alexnet", "resnet50", "swin_transformer", "vit")


def main(image_dir=IMAGE_DIR,
         label_path=LABELS_PATH,
         n_images=3,
         library="gradcam", method="GradCAM",
         magnitude=0.02,
         n_cols=3,
         ):
    np.random.seed(0)
    labels = load_labels(str(label_path))
    print(labels)

    filenames = ...
    images = ...

    for model_name in MODEL_NAMES:

        # Configure the model, target layers, and reshape transformation based on the model architecture
        model, target_layers, reshape_transform = setup_model_and_layers(model_name)

        # Initialize the explanation generator with the specified library and method
        generator = ExplanationGenerator(model, library, method)

        for filename, image in zip(filenames, images):

            for perturbation in PERTURBATIONS:

                noisy_images = perturbation(image, magnitude, n_images)

                # Generate explanations and optionally retrieve predicted labels
                explanations, pred_labels, noise_fraction_changes, _ = generator.generate_explanations(
                    noisy_images,
                    filename,
                    target_layers,
                    reshape_transform
                )
                # explanation / data to show

                fig, ax = plt.subplots(nrows=1, ncols=2)

                plt.suptitle(f'{model_name}\n{perturbation.__name__}')

                plt.sca(ax[0])
                plt.imshow(image.permute(1, 2, 0))

                plt.show()


if __name__ == '__main__':
    main()
