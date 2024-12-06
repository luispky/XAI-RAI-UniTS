"""
Location of directories and files outside the module
"""
from pathlib import Path


# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


# Paths for object outside the module
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "figures"
DATASETS_DIR = ROOT_DIR / "data"
IMAGE_DIR = ROOT_DIR / "data/images"
RESULTS_DIR = ROOT_DIR / "results"
LABELS_PATH = ROOT_DIR / "data/imagenet_classes.txt"


# Create directories
_paths = (MODELS_DIR, FIGURES_DIR, DATASETS_DIR, RESULTS_DIR)
for path in _paths:
    path.mkdir(parents=True, exist_ok=True)
del _paths
