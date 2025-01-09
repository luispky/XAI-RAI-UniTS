import numpy as np
import matplotlib.pyplot as plt
import torch
from xai_rai_units.src import perturbations as pert
from xai_rai_units.src import alexnet
from xai_rai_units.src.paths import IMAGE_DIR, MODELS_DIR, LABELS_PATH
from xai_rai_units.src.alexnet import load_images, load_labels
# from xai_rai_units.src import utils


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
                          magnitude=0.005,
                          model_path=MODELS_DIR / "alexnet_weights.pth"):
    """Test the inverse gradient perturbation."""
    np.random.seed(42)
    image = load_images(str(image_dir), n=1)[0]
    labels = load_labels(str(label_path))
    model = alexnet.download_alexnet(model_path)
    model.eval()

    out = model(image.unsqueeze(0))
    out = torch.softmax(out, dim=1)

    # use argsort to get the top 5 classes
    indices = np.argsort(out.detach().numpy().flatten())[-5:]
    prediction = indices[-1]
    grad = pert.inv_grad_perturbation(image, model=model, i_class=int(indices[-2]))
    grad = grad / grad.std()

    image_2 = image + grad * magnitude
    image_2 = torch.clamp(image_2, 0, 1)
    out_2 = model(image_2.unsqueeze(0))
    out_2 = torch.softmax(out_2, dim=1)
    prediction_2 = out_2.argmax()

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(5, 5))

    plt.sca(ax[0])
    plt.title(f'{labels[prediction]} ({out.max().item():.1%})')
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1])
    plt.title(f'+  gradient * {magnitude:.3f}')
    plt.imshow(grad.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[2])
    plt.title(f'=  {labels[prediction_2]} ({out_2.max().item():.1%})')
    plt.imshow(image_2.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def test_inverse_gradient_2(image_dir=IMAGE_DIR,
                            label_path=LABELS_PATH,
                            model_path=MODELS_DIR / "alexnet_weights.pth",
                            magnitude=0.01,
                            n_classes=4,
                            n=30,
                            ):
    """Test the inverse gradient perturbation."""
    np.random.seed(42)
    image = load_images(str(image_dir), n=None)[9]
    labels = load_labels(str(label_path))
    model = alexnet.download_alexnet(model_path)
    # model = utils.load_model('resnet50')
    # model = utils.load_model('swin_transformer')
    # model = utils.load_model('vit')
    # model = utils.load_model('alexnet')

    model.eval()

    out = model(image.unsqueeze(0))
    indices = np.argsort(out.detach().numpy().flatten())[-n_classes:]

    epsilon_ = np.linspace(0, 1, n) ** 2 * magnitude

    grad = pert.inv_grad_perturbation(image,
                                      model=model,
                                      i_class=int(indices[-2]))
    grad = grad / grad.std()

    x = torch.stack([image + grad * epsilon for epsilon in epsilon_])
    x = torch.clamp(x, 0, 1)
    image_2 = x[-1]
    out = model(x)
    out = torch.softmax(out, dim=1)
    out = out.clone().detach().numpy()
    proba = out[:, list(reversed(indices[:n_classes]))]

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
    plt.plot(epsilon_/magnitude, proba, label=[labels[i] for i in list(reversed(indices))])

    plt.xlabel(f'magnitude x {magnitude:.2f}')
    plt.ylabel('proba')
    plt.ylim(0, 1.05)
    plt.legend()

    plt.show()


def test_texture_perturbation(image_dir=IMAGE_DIR,
                              label_path=LABELS_PATH,
                              magnitude=0.1, k=0.7):
    """Test the texture perturbation."""
    np.random.seed(0)

    model = alexnet.download_alexnet(MODELS_DIR / "alexnet_weights.pth")
    model.eval()

    images_ = load_images(str(image_dir), n=15)
    image = images_[0]
    image_2 = images_[-1]

    labels = load_labels(str(label_path))
    out = model(image.unsqueeze(0))
    out = torch.softmax(out, dim=1)
    prediction = out.argmax().item()

    # crop the center of the image
    c, h, w = image_2.shape
    h2 = int(h*k/2)
    w2 = int(w*k/2)
    texture = image_2[:, h2:h-h2, w2:w-w2]

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


def test_undo_gradient_perturbation(image_dir=IMAGE_DIR,
                                    label_path=LABELS_PATH,
                                    magnitude=0.02):
    """Check if adding gaussian noise can correct for the gradient perturbation."""
    np.random.seed(42)

    model = alexnet.download_alexnet(MODELS_DIR / "alexnet_weights.pth")
    model.eval()

    # image
    image = load_images(str(image_dir), n=1)[0]
    labels = load_labels(str(label_path))
    out_1 = model(image.unsqueeze(0))
    out_1 = torch.softmax(out_1, dim=1)
    predictions_1 = out_1.argsort(descending=True)[0]

    # image + grad
    image.requires_grad = True
    image.retain_grad()
    output = model(image.unsqueeze(0))
    output_class = output[0, predictions_1[0]]
    output_class.backward()
    grad = image.grad
    grad = grad / grad.std()
    image_2 = image + grad * magnitude
    image_2 = torch.clamp(image_2, 0, 1)
    out_2 = model(image_2.unsqueeze(0))
    out_2 = torch.softmax(out_2, dim=1)
    predictions_2 = out_2.argsort(descending=True)[0]

    # image + noise
    noise = torch.randn_like(image)
    noise = noise / noise.std()
    image_3 = image + noise * magnitude
    image_3 = torch.clamp(image_3, 0, 1)
    out_3 = model(image_3.unsqueeze(0))
    out_3 = torch.softmax(out_3, dim=1)
    predictions_3 = out_3.argsort(descending=True)[0]

    # image + grad + noise
    image_4 = image_2 + noise * magnitude
    image_4 = torch.clamp(image_4, 0, 1)
    out_4 = model(image_4.unsqueeze(0))
    out_4 = torch.softmax(out_4, dim=1)
    predictions_4 = out_4.argsort(descending=True)[0]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))

    plt.sca(ax[0, 0])
    plt.title(f'Original image\n'
              f'{labels[predictions_1[0]]} ({out_1[0, predictions_1[0]].item():.1%})\n'
              f'{labels[predictions_1[1]]} ({out_1[0, predictions_1[1]].item():.1%})')
    plt.imshow(image.detach().permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1, 0])
    plt.title(f'Image + gradient\n'
              f'{labels[predictions_2[0]]} ({out_2[0, predictions_2[0]].item():.1%})\n'
              f'{labels[predictions_2[1]]} ({out_2[0, predictions_2[1]].item():.1%})')
    plt.imshow(image_2.detach().permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[0, 1])
    plt.title(f'Image + noise\n'
              f'{labels[predictions_3[0]]} ({out_3[0, predictions_3[0]].item():.1%})\n'
              f'{labels[predictions_3[1]]} ({out_3[0, predictions_3[1]].item():.1%})')
    plt.imshow(image_3.detach().permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1, 1])
    plt.title(f'Image + gradient + noise\n'
              f'{labels[predictions_4[0]]} ({out_4[0, predictions_4[0]].item():.1%})\n'
              f'{labels[predictions_4[1]]} ({out_4[0, predictions_4[1]].item():.1%})')
    plt.imshow(image_4.detach().permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[0, 2])
    plt.title('Gaussian Noise')
    plt.imshow(noise.detach().permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.sca(ax[1, 2])
    plt.title('Gradient')
    plt.imshow(grad.detach().permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # test_noisy_image_linspace()
    # test_blur_image_linspace()
    # test_occlusion_linspace()
    # test_the_void_linspace()
    # test_inverse_gradient()
    test_inverse_gradient_2()
    # test_texture_perturbation()
    # test_undo_gradient_perturbation()
