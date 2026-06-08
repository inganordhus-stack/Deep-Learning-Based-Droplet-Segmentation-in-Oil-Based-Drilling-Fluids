from pathlib import Path
import torch



# --------------------
# Device
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------
# Project paths
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]

ORIGINAL_DIR = BASE_DIR / "Original"
MODEL_DIR = BASE_DIR / "Latest_Models"


# --------------------
# Dataset paths
# --------------------
IMAGE_DIR = ORIGINAL_DIR / "Input"
MASK_DIR = ORIGINAL_DIR / "Output" / "Masks"

ROUNDED_MASK_DIR = ORIGINAL_DIR / "Output" / "Masks_rounded"
CSV_RESULTS_DIR = ORIGINAL_DIR / "Output" / "Results_CSV"


# --------------------
# Model paths
# --------------------
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_model.pth"


# --------------------
# Image settings
# --------------------
IMAGE_SIZE = (256, 256)
IN_CHANNELS = 1
NUM_CLASSES = 1


# --------------------
# Model settings
# --------------------
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DECODER_USE_NORM = "batchnorm"


# --------------------
# Training settings
# --------------------
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0


# --------------------
# Prediction settings
# --------------------
THRESHOLD = 0.5


# --------------------
# File extensions
# --------------------
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --------------------
# Create output folders
# --------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ROUNDED_MASK_DIR.mkdir(parents=True, exist_ok=True)
CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

