import torch
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from src.config import DEVICE, IMAGE_SIZE


def predict_mask(model, img_path, resize_to=IMAGE_SIZE, threshold=0.5):
    model.eval()

    img = Image.open(img_path).convert("L")

    if resize_to is not None:
        img = img.resize(resize_to, resample=Image.BILINEAR)

    x = TF.to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()

    return mask.squeeze().detach().cpu().numpy().astype(np.uint8)


def load_mask(mask_path, target_size=None):
    mask = Image.open(mask_path).convert("L")

    if target_size is not None:
        mask = mask.resize(target_size, Image.NEAREST)

    mask = np.array(mask)
    return (mask > 0).astype(np.uint8)


def to_tensor_mask(mask):
    if isinstance(mask, torch.Tensor):
        t = mask
    else:
        t = torch.from_numpy(np.array(mask))

    t = t.float()

    if t.max() > 1:
        t = t / 255.0

    t = (t > 0.5).float()

    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0)

    return t


def save_mask(mask, path):
    mask = np.array(mask)

    if mask.max() <= 1:
        mask = mask * 255

    mask = mask.astype(np.uint8)
    Image.fromarray(mask).save(path)


def make_overlay(original_img, mask, alpha=0.35):
    if not isinstance(original_img, Image.Image):
        original_img = Image.open(original_img).convert("L")

    original = np.array(original_img.convert("RGB"))
    mask = np.array(mask)

    if mask.max() <= 1:
        mask = mask * 255

    mask = mask.astype(np.uint8)

    overlay = original.copy()
    red_mask = np.zeros_like(original)
    red_mask[:, :, 0] = 255

    mask_bool = mask > 127

    overlay[mask_bool] = (
        (1 - alpha) * original[mask_bool]
        + alpha * red_mask[mask_bool]
    ).astype(np.uint8)

    return Image.fromarray(overlay)
