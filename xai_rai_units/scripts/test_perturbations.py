import numpy as np
import matplotlib.pyplot as plt
import torch
from xai_rai_units.src import perturbations as pert
from xai_rai_units.src.paths import IMAGE_DIR
from xai_rai_units.src import alexnet
from xai_rai_units.src.paths import MODELS_DIR, LABELS_PATH
from xai_rai_units.src.alexnet import load_images, load_labels


def test_noisy_image_linspace(image_dir=IMAGE_DIR, magnitude=.5, n=5):
    """Test the function noisy_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.gaussian_perturbation_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i+1}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_blur_image_linspace(image_dir=IMAGE_DIR, magnitude=8, n=5):
    """Test the function blur_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.blur_perturbation_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_occlusion_linspace(image_dir=IMAGE_DIR, magnitude=150, n=5):
    """Test the function blur_image_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.occlusion_perturbation_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i+1}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_the_void_linspace(image_dir=IMAGE_DIR, magnitude=10., n=5):
    """Test the function the_void_linspace."""
    image = load_images(str(image_dir), n=1)[0]
    perturbed_images = pert.void_perturbation_linspace(image, magnitude=magnitude, n=n)
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 2))
    for i in range(n):
        ax[i].set_title(f'{i+1}/{n}')
        ax[i].imshow(perturbed_images[i].permute(1, 2, 0).numpy())
        ax[i].axis('off')
    plt.show()


def test_inverse_gradient(image_dir=IMAGE_DIR,
                          label_path=LABELS_PATH,
                          magnitude=0.2,
                          model_path=MODELS_DIR / "alexnet_weights.pth"):
    """Test the inverse gradient perturbation."""
    np.random.seed(12)
    image = load_images(str(image_dir), n=1)[0]
    labels = load_labels(str(label_path))
    model = alexnet.download_alexnet(model_path)
    model.eval()

    out = model(image.unsqueeze(0)) / 100

    # use argsort to get the top 5 classes
    indices = np.argsort(out.detach().numpy().flatten())[-5:]
    prediction = indices[-1]
    print(prediction)

    grad = pert.inv_grad_perturbation(image, model=model, i_class=int(indices[-2]))
    norm = grad.std() * 2
    norm_grad = grad / norm

    image_2 = image + grad * magnitude
    image_2 = torch.clamp(image_2, 0, 1)

    out_2 = model(image_2.unsqueeze(0)) / 100
    prediction_2 = out_2.argmax()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(5, 5))

    plt.sca(ax[0])
    plt.title(f'{labels[prediction]} ({out.max().item():.1%})')
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1])
    plt.title(f'+  gradient * {magnitude * norm:.3f}')
    plt.imshow(norm_grad.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[2])
    plt.title(f'=  {labels[prediction_2]} ({out_2.max().item():.1%})')
    plt.imshow(image_2.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def test_inverse_gradient_2(image_dir=IMAGE_DIR,
                            label_path=LABELS_PATH,
                            magnitude=0.4,
                            n_classes=4,
                            n=30,
                            model_path=MODELS_DIR / "alexnet_weights.pth"):
    """Test the inverse gradient perturbation."""
    np.random.seed(13)
    image = load_images(str(image_dir), n=1)[0]
    labels = load_labels(str(label_path))
    model = alexnet.download_alexnet(model_path)
    model.eval()
    out = model(image.unsqueeze(0))

    # use argsort to get the top 5 classes
    indices = np.argsort(out.detach().numpy().flatten())[-n_classes:]
    proba = [[] for _ in range(n_classes)]
    image_2 = None

    x_ = np.linspace(0, 1, n) ** 2 * magnitude

    for epsilon in x_:
        grad = pert.inv_grad_perturbation(image,
                                          model=model,
                                          i_class=int(indices[-2]))

        image_2 = image + grad * epsilon
        image_2 = torch.clamp(image_2, 0, 1)

        out_2 = model(image_2.unsqueeze(0))[0]
        out_2 = torch.softmax(out_2, dim=0)

        for i in range(n_classes):
            proba[i].append(out_2[indices[-1-i]].item())

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    plt.sca(ax[0])
    plt.title(f'Original image')
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1])
    plt.title(f'Perturbed image')
    plt.imshow(image_2.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[2])
    plt.title(f'Proba of top {n_classes} classes')
    for i in range(n_classes):
        plt.plot(x_, proba[i], label=labels[indices[-1-i]])
    plt.xlabel('magnitude')
    plt.ylabel('proba')
    plt.ylim(0, 1)
    plt.legend()

    plt.show()


def test_texture_perturbation(image_dir=IMAGE_DIR,
                              label_path=LABELS_PATH,
                              magnitude=0.1):
    """Test the texture perturbation."""
    np.random.seed(7)

    model = alexnet.download_alexnet(MODELS_DIR / "alexnet_weights.pth")
    model.eval()

    image, image_2 = load_images(str(image_dir), n=2)
    labels = load_labels(str(label_path))
    out = model(image.unsqueeze(0))
    out = torch.softmax(out, dim=1)
    prediction = out.argmax().item()

    # rescale image to make it smaller
    texture = image_2

    texture -= texture.mean()
    perturbed_image = pert.overlay_pattern(image, texture, magnitude)

    out_2 = model(perturbed_image.unsqueeze(0))
    out_2 = torch.softmax(out_2, dim=1)
    prediction_2 = out_2.argmax().item()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    plt.sca(ax[0])
    plt.title(f'{labels[prediction]} ({out.max().item():.1%})')
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1])
    plt.title('Texture')
    plt.imshow(texture.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[2])
    plt.title(f'{labels[prediction_2]} ({out_2.max().item():.1%})')
    plt.imshow(perturbed_image.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    # test_noisy_image_linspace()
    # test_blur_image_linspace()
    # test_occlusion_linspace()
    # test_the_void_linspace()
    # test_inverse_gradient()
    test_inverse_gradient_2()
    # test_texture_perturbation()
