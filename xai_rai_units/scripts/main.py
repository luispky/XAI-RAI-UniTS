"""
The purpose of this project is to find out whether XAI
tools are robust to various image perturbations. The
project will use the AlexNet model to classify images,
perturb them with noise, occlusions, and other methods,
and then generate explanations for the classification.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from xai_rai_units.src.paths import LABELS_PATH
from xai_rai_units.src.alexnet import load_labels
from xai_rai_units.src.explanations import ExplanationGenerator
from xai_rai_units.src import perturbations as pert
from xai_rai_units.src.utils import setup_model_and_layers, load_local_images, sample_filenames, set_seed


set_seed(42)

PERTURBATIONS = (pert.identity_perturbation_linspace,
                 pert.gaussian_perturbation_linspace,
                 pert.blur_perturbation_linspace,
                 pert.occlusion_perturbation_linspace,
                 pert.void_perturbation_linspace,
                 pert.inv_grad_perturbation_linspace)

MODEL_NAMES = ("alexnet",
               # "resnet50",
               # "swin_transformer",
               # "vit",
               )


def plot_model_classification(model, images, labels, magnitude=0.01,
                              n_classes=4, do_yticks=True):
    """Plot the classification probabilities of the model for a range of noise magnitudes."""
    model.eval()

    out = model(images[0].unsqueeze(0))
    indices = np.argsort(out.detach().numpy().flatten())[-n_classes:]
    epsilon_ = np.linspace(0, 1, len(images)) ** 2 * magnitude
    out = model(images)
    out = torch.softmax(out, dim=1)
    out = out.clone().detach().numpy()
    proba = out[:, list(reversed(indices[:n_classes]))]

    # plt.title(f'Proba of top {n_classes} classes')
    plt.plot(epsilon_/magnitude, proba, label=[labels[i] for i in list(reversed(indices))])
    plt.xlabel(f'magnitude x {magnitude:.2f}')
    plt.ylabel('proba')
    plt.ylim(0, 1.05)
    if not do_yticks:
        plt.yticks([])
        plt.ylabel('')
    plt.legend()


def main(label_path=LABELS_PATH,
         n_images=30,
         library="gradcam",
         method="GradCAM",
         magnitude=.1,
         ):
    labels = load_labels(str(label_path))
    print(labels)

    filenames = sample_filenames(n=5)
    images = load_local_images(filenames)

    for model_name in MODEL_NAMES:

        # Configure the model, target layers, and reshape transformation based on the model architecture
        model, target_layers, reshape_transform = setup_model_and_layers(model_name)

        # Initialize the explanation generator with the specified library and method
        generator = ExplanationGenerator(model, library, method)

        for filename, image in zip(filenames, images):

            fig, ax = plt.subplots(nrows=3, ncols=len(PERTURBATIONS))
            plt.suptitle(f'Model: {model_name} \n Image: {filename}')

            for i_pert, perturbation in enumerate(PERTURBATIONS):
                print(perturbation)

                # imshow perturbed image
                plt.sca(ax[0, i_pert])
                name = perturbation.__name__
                name = name.replace('_perturbation', '')
                name = name.replace('_linspace', '')
                name = name.replace('_', '\n')
                plt.title(f'{name}')
                perturbed_images = perturbation(image, magnitude=magnitude, n=n_images, model=model)
                plt.imshow(perturbed_images[-1].permute(1, 2, 0).numpy())
                plt.axis('off')

                # imshow noisy explanation
                plt.sca(ax[1, i_pert])
                # plt.title(f'Explanation')

                explanations, pred_labels, noise_fraction_changes, attributions = generator.generate_explanations(
                    perturbed_images,
                    filename,
                    target_layers,
                    reshape_transform
                )

                # todo output at least 2
                j = 1 if len(attributions) >= 2 else 0

                attribution = attributions[j]
                noisy_explanation = explanations[j]

                _min, _max = noisy_explanation.min(), noisy_explanation.max()
                noisy_explanation = (noisy_explanation - _min) / (_max - _min)

                plt.imshow(noisy_explanation.numpy())
                plt.axis('off')

                # plot top class proba
                plt.sca(ax[2, i_pert])
                if i_pert == 0:
                    plt.box(False)
                    plt.xticks([])
                    plt.yticks([])
                else:
                    # plt.title(f'Classification')
                    plot_model_classification(model, perturbed_images, labels,
                                              magnitude=magnitude, do_yticks=i_pert==1)

            fig.subplots_adjust(hspace=0.06, wspace=0.06)
            plt.show()


if __name__ == '__main__':
    main()
