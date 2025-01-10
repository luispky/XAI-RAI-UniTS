"""
The purpose of this project is to find out whether XAI
tools are robust to various image perturbations. The
project will use the AlexNet model to classify images,
perturb them with noise, occlusions, and other methods,
and then generate explanations for the classification.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from xai_rai_units.src.paths import LABELS_PATH, FIGURES_DIR
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
                 pert.inv_grad_perturbation_linspace,
                 )

MODEL_NAMES = ("alexnet",
               # "resnet50",
               # "swin_transformer",
               # "vit",
               )


def plot_model_classification(model, images, labels, magnitude=0.01,
                              n_classes=4, do_yticks=True, legend_size=6):
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
    if legend_size is None:
        plt.legend()
    else:
        plt.legend(prop={'size': legend_size})



def main(label_path=LABELS_PATH,
         n_images=30,
         library="gradcam", method="GradCAM",
         magnitude=.1,
         ):
    labels = load_labels(str(label_path))
    print(labels)

    filenames = sample_filenames(n=5)
    images = load_local_images(filenames)

    figures_list = os.listdir(f'{FIGURES_DIR}\\plots')

    while True:
        model_name = np.random.choice(MODEL_NAMES, 1)[0]

        # Configure the model, target layers, and reshape transformation based on the model architecture
        model, target_layers, reshape_transform = setup_model_and_layers(model_name)

        # Initialize the explanation generator with the specified library and method
        generator = ExplanationGenerator(model, library, method)

        i_img = np.random.choice(len(images), 1)[0]
        filename = filenames[i_img]
        image = images[i_img]

        plot_name = f'{FIGURES_DIR}\\plots\\plot_{model_name}_{filename}.png'
        if plot_name in figures_list:
            continue

        fig, ax = plt.subplots(nrows=3, ncols=len(PERTURBATIONS), figsize=(10, 6))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.1, top=0.88)

        suptitle = f'Model: {model_name}  |  Image: {filename}'
        print(f'\n\n{suptitle}')
        plt.suptitle(suptitle)

        for i_pert, perturbation in enumerate(PERTURBATIONS):
            print(perturbation.__name__)

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

        # Get the current figure manager
        manager = plt.get_current_fig_manager()

        # Toggle full screen mode
        manager.full_screen_toggle()

        # show
        # plt.show(block=True)

        # Save the figure in full screen mode
        fig.savefig(plot_name, dpi=1000)

        figures_list.append(plot_name)
        plt.close()


if __name__ == '__main__':
    main()
