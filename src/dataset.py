import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from src.process_mask import create_rounded_masks
import albumentations as A


from src.config import (
    ORIGINAL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    VAL_FRACTION,
    IMAGE_EXTENSIONS,
)


# --------------------
# Paths
# --------------------
DATASET_DIR = ORIGINAL_DIR.parent / "Dataset"

ORIG_IMAGE_DIR = ORIGINAL_DIR / "Input"
#for rounded masks:
#ORIG_MASK_DIR = ORIGINAL_DIR / "Output" / "Masks_rounded" # <---- Change to "Masks" if you do not want rounded masks
# Use this instead if you do not want rounded masks:
ORIG_MASK_DIR = ORIGINAL_DIR / "Output" / "Masks"

TRAIN_INPUT_DIR = DATASET_DIR / "train" / "Input"
TRAIN_MASK_DIR = DATASET_DIR / "train" / "Masks"

VAL_INPUT_DIR = DATASET_DIR / "val" / "Input"
VAL_MASK_DIR = DATASET_DIR / "val" / "Masks"


# --------------------
# Augmentations
# --------------------
AUGS = [
    "rot90", "rot180", "rot270",
    "hflip", "hflip_rot90", "hflip_rot180", "hflip_rot270",
    "vflip", "vflip_rot90", "vflip_rot180", "vflip_rot270",
]
train_transform_light = A.Compose([ #light
    A.RandomBrightnessContrast(
        brightness_limit=0.10,
        contrast_limit=0.10,
        p=0.3
    ),

    A.RandomGamma(
        gamma_limit=(90, 110),
        p=0.15
    ),

    A.GaussNoise(
        std_range=(0.01, 0.03),
        p=0.15
    ),

    A.GaussianBlur(
        blur_limit=(3, 3),
        p=0.10
    ),
])
train_transform = A.Compose([ #old
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.CLAHE(p=0.3),

    A.GaussNoise(p=0.4),
    A.GaussianBlur(p=0.25),

    
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.08,
        rotate_limit=10,
        p=0.4
    ),
    
    A.CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(0.02, 0.08),
        hole_width_range=(0.02, 0.08),
        p=0.25
    ),
])


def apply_transform(img: Image.Image, kind: str) -> Image.Image:
    if kind == "rot90":
        return img.transpose(Image.ROTATE_90)
    if kind == "rot180":
        return img.transpose(Image.ROTATE_180)
    if kind == "rot270":
        return img.transpose(Image.ROTATE_270)
    if kind == "hflip":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if kind == "hflip_rot90":
        return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    if kind == "hflip_rot180":
        return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_180)
    if kind == "hflip_rot270":
        return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
    if kind == "vflip":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if kind == "vflip_rot90":
        return img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90)
    if kind == "vflip_rot180":
        return img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_180)
    if kind == "vflip_rot270":
        return img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270)

    raise ValueError(f"Unknown augmentation: {kind}")


# --------------------
# Build train/val folders
# --------------------
def rebuild_dataset_folders():
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for folder in [
        TRAIN_INPUT_DIR,
        TRAIN_MASK_DIR,
        VAL_INPUT_DIR,
        VAL_MASK_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def get_original_ids():
    orig_imgs = sorted([
        p for p in ORIG_IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "__" not in p.stem
    ])

    missing_masks = [
        p.name for p in orig_imgs
        if not (ORIG_MASK_DIR / p.name).exists()
    ]

    if missing_masks:
        raise RuntimeError(f"Missing masks for: {missing_masks[:10]}")

    return [p.name for p in orig_imgs]


def copy_ids(ids, input_dir, mask_dir):
    for filename in ids:
        shutil.copy2(ORIG_IMAGE_DIR / filename, input_dir / filename)
        shutil.copy2(ORIG_MASK_DIR / filename, mask_dir / filename)


def augment_training_data():
    original_train_files = sorted([
        p for p in TRAIN_INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "__" not in p.stem
    ])

    created = 0

    for img_path in original_train_files:
        mask_path = TRAIN_MASK_DIR / img_path.name

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        base = img_path.stem
        ext = img_path.suffix

        for aug in AUGS:
            out_img = TRAIN_INPUT_DIR / f"{base}__{aug}{ext}"
            out_mask = TRAIN_MASK_DIR / f"{base}__{aug}{ext}"

            apply_transform(img, aug).save(out_img)
            apply_transform(mask, aug).save(out_mask)

            created += 1

    print(f"Train originals processed: {len(original_train_files)}")
    print(f"New augmented pairs created: {created}")


def prepare_train_val_dataset():
    create_rounded_masks(overwrite=True)
    rebuild_dataset_folders()

    orig_ids = get_original_ids()
    print(f"Original count: {len(orig_ids)}")

    train_ids, val_ids = train_test_split(
        orig_ids,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    copy_ids(train_ids, TRAIN_INPUT_DIR, TRAIN_MASK_DIR)
    copy_ids(val_ids, VAL_INPUT_DIR, VAL_MASK_DIR)

    print(f"Train originals: {len(train_ids)}")
    print(f"Validation originals: {len(val_ids)}")

    augment_training_data() #<-- Comment this line if you don't want to artifically increase the size of the training set with augmentations. Note that this will not create new unique images, but just transformed versions of the existing ones.

    print(f"Final train images: {len(list(TRAIN_INPUT_DIR.iterdir()))}")
    print(f"Final val images: {len(list(VAL_INPUT_DIR.iterdir()))}")


# --------------------
# PyTorch dataset
# --------------------
class DropletDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([
            p for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        self.mask_paths = sorted([
            p for p in Path(mask_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if len(self.image_paths) != len(self.mask_paths):
            raise RuntimeError(
                f"Mismatch: {len(self.image_paths)} images and "
                f"{len(self.mask_paths)} masks."
            )

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if img_path.name != mask_path.name:
                raise RuntimeError(
                    f"Image/mask mismatch: {img_path.name} != {mask_path.name}"
                )

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = image.resize(IMAGE_SIZE)
        mask = mask.resize(IMAGE_SIZE)

        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        mask = (mask > 0).float()

        return image, mask


def get_transforms():
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
    ])


def make_train_val_loaders():
    prepare_train_val_dataset()

    transform = get_transforms()

    train_dataset = DropletDataset(
        image_dir=TRAIN_INPUT_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform= train_transform, #for heavy
        #transform= train_transform_light, #for light
        #transform = None, #for no transform
    )

    val_dataset = DropletDataset(
        image_dir=VAL_INPUT_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE-4,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader

class FullOriginalDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([
            p for p in Path(image_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        self.mask_paths = [
            Path(mask_dir) / p.name for p in self.image_paths
        ]

        for mask_path in self.mask_paths:
            if not mask_path.exists():
                raise RuntimeError(f"Missing mask: {mask_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(IMAGE_SIZE)
        mask = mask.resize(IMAGE_SIZE)

        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        mask = (mask > 0).float()

        return image, mask, image_path.name