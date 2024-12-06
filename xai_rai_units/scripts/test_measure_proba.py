"""
Check how the perturbations of an image influence the classification.
"""
import numpy as np
import matplotlib.pyplot as plt
from xai_rai_units.src.paths import MODELS_DIR, IMAGE_DIR, LABELS_PATH
from xai_rai_units.src.measure_proba import MeasureProba
from xai_rai_units.src.alexnet import download_alexnet, load_images, load_labels
from xai_rai_units.src import utils


def plot_measure_proba(model, image, label_path=LABELS_PATH,
                       magnitude=.1, n=40, seed=14):
    """
    Plot how the probability of the most likely class of an images changes
    as the noise in the image gets more intense
    """
    np.random.seed(seed)

    fig, ax = plt.subplots(nrows=2, ncols=2)

    labels = load_labels(str(label_path))
    
    # Loop through all modules in the model to disable dropout
    model.eval()  # Set the model to evaluation mode (eval automatically disables dropout and batch normalization)

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


def test_1(image_dir=IMAGE_DIR):
    """Omar's test"""
    model_path = f"{str(MODELS_DIR)}/alexnet_weights.pth"
    model = download_alexnet(model_path)
    image = load_images(str(image_dir), n=1)[0]
    plot_measure_proba(model=model, image=image)


def test_2(image_dir=IMAGE_DIR):
    """Luis's test"""
    filename = "cassette_player.jpg"
    image = utils.load_local_images(str(image_dir) + "/" + filename)[0]  # remove the batch dimension
    model = utils.load_model('resnet50')
    plot_measure_proba(model=model, image=image)


if __name__ == '__main__':
    test_1()
    # test_2()
