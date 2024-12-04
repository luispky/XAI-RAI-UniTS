import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

from pathlib import Path

# Add src directory to Python path
PARENT_DIRECTORY = str(Path(__file__).resolve().parent.parent)
sys.path.append(PARENT_DIRECTORY + "/src")

from paths import MODELS_DIR, IMAGE_DIR, LABELS_PATH
from measure_proba import MeasureProba
from alexnet import download_alexnet, load_images, load_labels

def test(model_path=f"{str(MODELS_DIR)}/alexnet_weights.pth",# !MODELS_DIR IS A PATH OBJECT, NOT A STRING
         image_dir=IMAGE_DIR, # !IMAGE_DIR IS ALREADY A STRING
         label_path=LABELS_PATH, # !LABELS_PATH IS ALREADY A STRING
         magnitude=.1, n=40,
         seed=14):
    np.random.seed(seed)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    image = load_images(image_dir, n=1)[0]
    labels = load_labels(label_path)
    model = download_alexnet(model_path)

    # Loop through all modules in the model to disable dropout
    model.eval()  # Set the model to evaluation mode
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.train = False  # Disable dropout for evaluation

    measure = MeasureProba(model)
    x_ = np.linspace(0, magnitude, n)
    proba, i_max, target = measure.run(image, magnitude=magnitude, n=n, seed=seed)

    plt.sca(ax[0, 0])
    plt.title(f'{labels[i_max]} ({proba[0]:.1%})')
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.gca().axis('off')

    plt.sca(ax[0, 1])
    plt.title(f'{labels[i_max]} ({proba[-1]:.1%})')
    plt.imshow(target.permute(1, 2, 0).numpy())
    plt.gca().axis('off')

    plt.sca(ax[1, 0])
    plt.xlabel('noise')
    plt.title('proba')
    plt.ylim(0, 1)
    plt.plot(x_, proba)
    plt.fill_between(x_, proba, color='#1f77b4', alpha=0.2)

    plt.sca(ax[1, 1])
    plt.xlabel('noise')
    plt.title('Log10 proba')
    plt.plot(x_, np.log10(proba))

    mat = measure.run_multiple(image, n_segments=10, magnitude=magnitude, n=n)
    plt.plot(x_, np.log10(mat.T), c='#1f77b4', alpha=0.4)

    plt.show()


if __name__ == '__main__':
    test()
