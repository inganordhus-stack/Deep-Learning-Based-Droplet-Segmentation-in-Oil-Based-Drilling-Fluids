from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.draw import disk as draw_disk
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

from src.config import MASK_DIR, ROUNDED_MASK_DIR, MASK_EXTENSIONS


def fill_mask(mask, r=1):
    m = mask.astype(bool)
    m = binary_fill_holes(m)
    m = closing(m, disk(r))
    return m


def roundify_by_equiv_diameter(mask01, min_area_px=5):
    m = mask01.astype(bool)
    m = remove_small_objects(m, max_size=min_area_px)

    lab = label(m)
    out = np.zeros_like(m, dtype=bool)

    height, width = out.shape

    for region in regionprops(lab):
        area = region.area
        radius = np.sqrt(area / np.pi)
        cy, cx = region.centroid

        rr, cc = draw_disk((cy, cx), radius, shape=(height, width))
        out[rr, cc] = True

    return out


def post_process_mask(mask, round=True):
    filled = fill_mask(mask)
    if round:
        rounded = roundify_by_equiv_diameter(filled)
        return rounded.astype(np.uint8)
    return filled.astype(np.uint8)


def create_rounded_masks(
    input_mask_dir=MASK_DIR,
    output_mask_dir=ROUNDED_MASK_DIR,
    overwrite=True,
):
    input_mask_dir = Path(input_mask_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted([
        p for p in input_mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS
    ])

    if not mask_paths:
        raise RuntimeError(f"No masks found in {input_mask_dir}")

    created = 0

    for mask_path in mask_paths:
        out_path = output_mask_dir / mask_path.name

        if out_path.exists() and not overwrite:
            continue

        mask = Image.open(mask_path).convert("L")
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        mask_pp = post_process_mask(mask_np)

        out_img = Image.fromarray((mask_pp * 255).astype(np.uint8))
        out_img.save(out_path)

        created += 1

    print(f"Rounded masks saved: {created}")
    print(f"Output folder: {output_mask_dir}")