import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


def count_objects(mask, min_area_px=10):
    """
    Count connected objects in a binary mask.
    """
    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, max_size=min_area_px)

    labeled_mask = label(mask_bool)
    regions = regionprops(labeled_mask)

    return len(regions)


def diameters_from_mask(mask, microns_per_pixel=None, min_area_px=4):
    """
    Extract equivalent circular diameters from a binary mask.

    Diameter is calculated from object area:
        d = 2 * sqrt(A / pi)
    """
    mask_bool = mask.astype(bool)
    mask_bool = remove_small_objects(mask_bool, max_size=min_area_px)

    labeled_mask = label(mask_bool)
    regions = regionprops(labeled_mask)

    diameters = [
        2 * np.sqrt(region.area / np.pi)
        for region in regions
    ]

    if microns_per_pixel is not None:
        diameters = [
            diameter * microns_per_pixel
            for diameter in diameters
        ]

    return diameters


def diameter_stats(diameters, round_digits=5):
    """
    Summary statistics for a list of diameters.
    """
    if len(diameters) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    d = np.array(diameters)

    return {
        "count": int(len(d)),
        "mean": round(float(d.mean()), round_digits),
        "median": round(float(np.median(d)), round_digits),
        "std": round(float(d.std()), round_digits),
        "min": round(float(d.min()), round_digits),
        "max": round(float(d.max()), round_digits),
    }


def diameter_error_stats(gt_diameters, pred_diameters, round_digits=5):
    """
    Compare ground-truth and predicted diameter distributions.

    The diameters are sorted and compared pairwise up to the shortest list.
    """
    if len(gt_diameters) == 0 or len(pred_diameters) == 0:
        return {
            "mse": None,
            "rmse": None,
            "mae": None,
            "bias": None,
            "std": None,
        }

    n = min(len(gt_diameters), len(pred_diameters))

    gt = np.sort(np.array(gt_diameters))[:n]
    pred = np.sort(np.array(pred_diameters))[:n]

    errors = pred - gt

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)
    std = np.std(errors)

    return {
        "mse": round(float(mse), round_digits),
        "rmse": round(float(rmse), round_digits),
        "mae": round(float(mae), round_digits),
        "bias": round(float(bias), round_digits),
        "std": round(float(std), round_digits),
    }


def phase_pixel_counts(mask):
    """
    Count water and oil pixels from a binary mask.

    Mask value 1 = water/dispersed phase
    Mask value 0 = oil/continuous phase
    """
    mask = (mask > 0).astype(np.uint8)

    water_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    oil_pixels = total_pixels - water_pixels

    return water_pixels, oil_pixels


def phase_fractions(mask):
    """
    Calculate water and oil pixel fractions from a binary mask.
    """
    water_pixels, oil_pixels = phase_pixel_counts(mask)
    total_pixels = water_pixels + oil_pixels

    if total_pixels == 0:
        return {
            "water_fraction": None,
            "oil_fraction": None,
            "water_percent": None,
            "oil_percent": None,
            "water_oil_ratio": None,
        }

    water_fraction = water_pixels / total_pixels
    oil_fraction = oil_pixels / total_pixels

    if oil_pixels == 0:
        water_oil_ratio = None
    else:
        water_oil_ratio = water_pixels / oil_pixels

    return {
        "water_fraction": water_fraction,
        "oil_fraction": oil_fraction,
        "water_percent": water_fraction * 100,
        "oil_percent": oil_fraction * 100,
        "water_oil_ratio": water_oil_ratio,
    }