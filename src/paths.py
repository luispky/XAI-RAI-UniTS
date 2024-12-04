from pathlib import Path

# Base directory for utils.py
BASE_DIR = Path(__file__).resolve().parent

# Paths for saving results
MODELS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Paths for saving results
FIGURES_DIR = BASE_DIR.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = BASE_DIR.parent / "data"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = str(BASE_DIR.parent / "data/images")
# IMAGE_DIR.mkdir(parents=True, exist_ok=True) # Makes no sense because the directory is already created

LABELS_PATH = str(BASE_DIR.parent / "data/imagenet_classes.txt")
# LABELS_PATH.mkdir(parents=True, exist_ok=True) # Makes no sense because this is a path to a file

RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)