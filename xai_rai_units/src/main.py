"""
The purpose of this project is to find out whether XAI
tools are robust to various image perturbations. The
project will use the AlexNet model to classify images,
perturb them with noise, occlusions, and other methods,
and then generate explanations for the classification.
"""
import matplotlib.pyplot as plt

from xai_rai_units.src.paths import IMAGE_DIR, MODELS_DIR, LABELS_PATH
from xai_rai_units.src.alexnet import load_images, load_labels
from xai_rai_units.src import alexnet
from xai_rai_units.src import perturbations as pert


PERTURBATIONS = ()


def main(image_dir=IMAGE_DIR,
         label_path=LABELS_PATH,
         model_path=MODELS_DIR / "alexnet_weights.pth",
         n_images=3,
         magnitude=0.02,
         n=30,
         ):
    images = load_images(str(image_dir), n=n_images)
    labels = load_labels(str(label_path))
    model = alexnet.download_alexnet(model_path)
    model.eval()

    for image in images:

        # classification

        fig, ax = plt.subplots(nrows=len(PERTURBATIONS), n_cols=2)
        for i, perturbation in enumerate(PERTURBATIONS):
            plt.sca(ax[i, 0])
            plt.imshow(image)

            plt.sca(ax[i, 1])

    plt.show()



if __name__ == '__main__':
    main()
