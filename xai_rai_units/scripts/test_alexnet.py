from xai_rai_units.src import alexnet
from xai_rai_units.src.paths import MODELS_DIR, IMAGE_DIR, LABELS_PATH


def test_alexnet(model_path=MODELS_DIR / "alexnet_weights.pth",
                 image_dir=IMAGE_DIR,
                 label_path=LABELS_PATH,
                 n=16):
    """
    Load and classify images with AlexNet
    """
    print(">>> Downloading the model...")
    model = alexnet.download_alexnet(model_path)

    print(">>> Loading images...")
    images = alexnet.load_images(image_dir, n=n)
    labels = alexnet.load_labels(label_path)

    print(">>> Making predictions...")
    outputs, predicted = alexnet.predict(model, images)
    alexnet.visualize(outputs, predicted, images, labels)


if __name__ == '__main__':
    test_alexnet()
