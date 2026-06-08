
import time
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import csv
from src.mask_utils import predict_mask, load_mask, to_tensor_mask
from src.process_mask import post_process_mask
from src.metrics import dice_score, iou_score, precision_score, recall_score, pixel_accuracy
from src.dataset import (DataLoader, ORIG_IMAGE_DIR, ORIG_MASK_DIR, FullOriginalDataset)
from src.mask_analysis import (
    diameters_from_mask,
    diameter_stats,
    diameter_error_stats,
    phase_fractions,
)

from tqdm import tqdm

from pathlib import Path
import numpy as np
from PIL import Image


def DSD_from_mask(mask, microns_per_pixel=None, min_area_px=5):

    diameters = diameters_from_mask(
        mask,
        microns_per_pixel=microns_per_pixel,
        min_area_px=min_area_px
    )

    diameters = np.array(diameters)

    if len(diameters) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "d10": np.nan,
            "d50": np.nan,
            "d90": np.nan,
            "span": np.nan,
            "d32": np.nan,
            "cv": np.nan,
        }

    d10 = np.percentile(diameters, 10)
    d50 = np.percentile(diameters, 50)
    d90 = np.percentile(diameters, 90)

    return {
        "count": len(diameters),
        "mean": np.mean(diameters),
        "median": np.median(diameters),
        "std": np.std(diameters),
        "min": np.min(diameters),
        "max": np.max(diameters),

        "d10": d10,
        "d50": d50,
        "d90": d90,

        "span": (d90 - d10) / d50 if d50 != 0 else np.nan,

        "d32": (
        np.sum(diameters**3) / np.sum(diameters**2)
        if np.sum(diameters**2) != 0 else np.nan
),
        "cv": np.std(diameters) / np.mean(diameters) if np.mean(diameters) != 0 else np.nan
       
    }




def DSD_from_image(image_path, microns_per_pixel=None, min_area_px=10):

    mask = np.array(
        Image.open(image_path).convert("L")
    ) > 0

    return DSD_from_mask(
        mask,
        microns_per_pixel=microns_per_pixel,
        min_area_px=min_area_px
    )

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array(
        [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)],
        dtype=float
    )
colors = {
    "pink 1": hex_to_rgb("#FF8CC6"),
    "purple 1": hex_to_rgb("#571F4E"),
    "pink 3": hex_to_rgb("#D20069"),
    "pink 2": hex_to_rgb("#DE369D"),
    "purple 2" : hex_to_rgb("#6F5E76"),
}



def evaluate_dataset(
    model,
    input_dir,
    mask_dir,
    threshold=0.5,
    resize_to=(256, 256),
    microns_per_pixel=None,
    min_area_px=10,
):
    input_dir = Path(input_dir)
    mask_dir = Path(mask_dir)

    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_ext])

    rows = []
    masks = {}

    all_gt_d = []
    all_pred_raw_d = []
    all_pred_post_d = []

    for img_path in images:
        gt_path = mask_dir / img_path.name
        if not gt_path.exists():
            continue

        # -------------------------
        # Prediction
        # -------------------------
        pred_raw = predict_mask(model, img_path, resize_to=resize_to, threshold=threshold)
        pred_post = post_process_mask(pred_raw)

        # -------------------------
        # Ground truth
        # -------------------------
        gt_mask = load_mask(gt_path, target_size=pred_post.shape)
        
        masks[img_path.name] = {
            "gt": gt_mask,
            "raw": pred_raw,
            "post": pred_post,
        }

        # -------------------------
        # Tensor metrics
        # -------------------------
        gt_t = to_tensor_mask(gt_mask)
        raw_t = to_tensor_mask(pred_raw)
        post_t = to_tensor_mask(pred_post)

        # Raw metrics
        dice_raw = dice_score(raw_t, gt_t).item()
        iou_raw = iou_score(raw_t, gt_t).item()
        precision_raw = precision_score(raw_t, gt_t).item()
        recall_raw = recall_score(raw_t, gt_t).item()
        pixel_acc_raw = pixel_accuracy(raw_t, gt_t).item()

        # Post metrics
        dice_post = dice_score(post_t, gt_t).item()
        iou_post = iou_score(post_t, gt_t).item()
        precision_post = precision_score(post_t, gt_t).item()
        recall_post = recall_score(post_t, gt_t).item()
        pixel_acc_post = pixel_accuracy(post_t, gt_t).item()
        # -------------------------
        # Diameters
        # -------------------------
        gt_d = diameters_from_mask(gt_mask, microns_per_pixel, min_area_px)
        raw_d = diameters_from_mask(pred_raw, microns_per_pixel, min_area_px)
        post_d = diameters_from_mask(pred_post, microns_per_pixel, min_area_px)

        all_gt_d.extend(gt_d)
        all_pred_raw_d.extend(raw_d)
        all_pred_post_d.extend(post_d)

        gt_stats = diameter_stats(gt_d)
        raw_stats = diameter_stats(raw_d)
        post_stats = diameter_stats(post_d)

        err_raw = diameter_error_stats(gt_d, raw_d)
        err_post = diameter_error_stats(gt_d, post_d)

        # -------------------------
        # Phase fractions
        # -------------------------
        gt_phase = phase_fractions(gt_mask)
        raw_phase = phase_fractions(pred_raw)
        post_phase = phase_fractions(pred_post)

        rows.append({
            "image": img_path.name,

            # --- RAW ---
            "dice_raw": dice_raw,
            "iou_raw": iou_raw,
            "precision_raw": precision_raw,
            "recall_raw": recall_raw,
            "pixel_acc_raw": pixel_acc_raw,
            # --- POST ---
            "dice_post": dice_post,
            "iou_post": iou_post,
            "precision_post": precision_post,
            "recall_post": recall_post,
            "pixel_acc_post": pixel_acc_post,

            # --- Counts ---
            "gt_count": gt_stats["count"],
            "raw_count": raw_stats["count"],
            "post_count": post_stats["count"],

            # --- Diameter ---
            "gt_mean_diameter": gt_stats["mean"],
            "raw_mean_diameter": raw_stats["mean"],
            "post_mean_diameter": post_stats["mean"],
            
            "gt_median_diameter": gt_stats["median"],
            "raw_median_diameter": raw_stats["median"],
            "post_median_diameter": post_stats["median"],

            "gt_min_diameter": gt_stats["min"],
            "raw_min_diameter": raw_stats["min"],
            "post_min_diameter": post_stats["min"],

            "gt_max_diameter": gt_stats["max"],
            "raw_max_diameter": raw_stats["max"],
            "post_max_diameter": post_stats["max"],

            "rmse_raw": err_raw["rmse"],
            "rmse_post": err_post["rmse"],

            "mae_raw": err_raw["mae"],
            "mae_post": err_post["mae"],

            "bias_raw": err_raw["bias"],
            "bias_post": err_post["bias"],

            # --- Phase ---
            "gt_water_percent": gt_phase["water_percent"],
            "raw_water_percent": raw_phase["water_percent"],
            "post_water_percent": post_phase["water_percent"],

            "gt_water_oil_ratio": gt_phase["water_oil_ratio"],
            "raw_water_oil_ratio": raw_phase["water_oil_ratio"],
            "post_water_oil_ratio": post_phase["water_oil_ratio"],
        })

    df = pd.DataFrame(rows)

    return df, all_gt_d, all_pred_raw_d, all_pred_post_d, masks

def plot_diameter_comparison(gt_d, pred_d, title="Diameter Comparison"):
    

    plt.figure(figsize=(6, 6))
    plt.scatter(gt_d, pred_d, alpha=0.5)
    plt.plot([0, max(gt_d)], [0, max(gt_d)], 'r--')  # y=x line
    plt.xlabel("Ground Truth Diameter (µm)")
    plt.ylabel("Predicted Diameter (µm)")
    plt.title(title)
    plt.grid()
    plt.show()
    

def plot_histogram(mask, m_per_p=None, b=20, log=True, min_area_px=10, ):
    diameters = diameters_from_mask(
        mask,
        microns_per_pixel=m_per_p,
        min_area_px=min_area_px
    )

    diameter_print = diameter_stats(diameters)

    plt.figure(figsize=(7, 5))
    plt.hist(diameters, bins=b, log=log, color = colors["pink 2"])

    if m_per_p is not None:
        plt.xlabel("Diameter (µm)")
        unit = "µm"
    else:
        plt.xlabel("Diameter (pixels)")
        unit = "px"

    plt.ylabel("Frequency")
    plt.title("Distribution of Droplet Diameters")

    plt.text(
        0.95,
        0.95,
        f"Count: {diameter_print['count']}\n"
        f"Mean: {diameter_print['mean']:.2f} {unit}\n"
        f"Median: {diameter_print['median']:.2f} {unit}\n"
        f"Min: {diameter_print['min']:.2f} {unit}\n"
        f"Max: {diameter_print['max']:.2f} {unit}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            alpha=0.8
        )
    )

    plt.grid(alpha=0.3)
    plt.show()
        
        
def predict_full_image(
    model,
    image_path,
    threshold=0.5,
    divisor=16,
    as_grayscale=True,
    microns_per_pixel=None,
    postprocess=True,
    plot=True,
):
    image_path = Path(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(image_path)
    if as_grayscale:
        img = img.convert("L")

    big_arr = np.array(img)

    arr = big_arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0

    x = torch.from_numpy(arr)[None, None, ...].to(device)

    B, C, H, W = x.shape

    newH = int(np.ceil(H / divisor) * divisor)
    newW = int(np.ceil(W / divisor) * divisor)

    pad_bottom = newH - H
    pad_right = newW - W

    if pad_bottom > 0 or pad_right > 0:
        import torch.nn.functional as F
        x_pad = F.pad(x, (0, pad_right, 0, pad_bottom), mode="replicate")
    else:
        x_pad = x

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(x_pad)
        probs = torch.sigmoid(logits)
        pred_raw_tensor = (probs > threshold).float()

    if pad_bottom > 0:
        pred_raw_tensor = pred_raw_tensor[..., :-pad_bottom, :]

    if pad_right > 0:
        pred_raw_tensor = pred_raw_tensor[..., :, :-pad_right]

    raw_mask = pred_raw_tensor.squeeze().cpu().numpy().astype(np.uint8)

    if postprocess:
        final_mask = post_process_mask(raw_mask).astype(np.uint8)
        overlay_color = colors["pink 3"]
        mask_label = "Post-processed"
    else:
        final_mask = raw_mask
        overlay_color = colors["pink 2"]
        mask_label = "Raw"

    diameters = diameters_from_mask(
        final_mask,
        microns_per_pixel=microns_per_pixel
    )

    diameter_print = diameter_stats(diameters)
    #print(f"{image_path.name}: {diameter_print}")
    
    dsd = DSD_from_mask(final_mask, microns_per_pixel=microns_per_pixel)
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(big_arr, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(final_mask, cmap="gray")
        axes[1].set_title(f"{mask_label} Mask")
        axes[1].axis("off")

        mask_bin = (final_mask > 0).astype(float)

        overlay = np.zeros((*mask_bin.shape, 4), dtype=float)
        overlay[..., :3] = overlay_color
        overlay[..., 3] = mask_bin * 0.45

        axes[2].imshow(big_arr, cmap="gray")
        axes[2].imshow(overlay)
        axes[2].set_title(f"{mask_label} Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "image": big_arr,
        "raw_mask": raw_mask,
        "final_mask": final_mask,
        "diameters": diameters,
        "dsd": dsd,
        "diameter_stats": diameter_print,
        "image_path": image_path,
        "postprocess": postprocess,
    }
    

def plot_compare_masks(image, gt_mask, raw_mask, post_mask, title=None):
    """
    image: original image (HxW eller HxWx3)
    gt_mask, raw_mask, post_mask: 2D arrays (HxW)
    """

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    rows = [
        ("Ground Truth", gt_mask, colors["pink 1"]),
        ("Raw Prediction", raw_mask, colors["pink 2"]),
        ("Post-processed", post_mask, colors["pink 3"]),
    ]

    for i, (row_title, mask, color) in enumerate(rows):

        # 1️⃣ Original
        axes[i, 0].imshow(image, cmap="gray")
        axes[i, 0].set_title(f"{row_title} - Image")
        axes[i, 0].axis("off")

        # 2️⃣ Mask
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"{row_title} - Mask")
        axes[i, 1].axis("off")
        
        mask_bin = (mask > 0).astype(float)

        overlay = np.zeros((*mask_bin.shape, 4), dtype=float)
        overlay[..., :3] = color
        overlay[..., 3] = mask_bin * 0.45   # alpha only where mask exists

        axes[i, 2].imshow(image, cmap="gray")
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"{row_title} - Overlay")
        axes[i, 2].axis("off")

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()



def plot_compare_masks_version2(image, gt_mask, raw_mask, post_mask, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    columns = [
        ("Ground Truth", gt_mask, colors["pink 1"]),
        ("Raw Prediction", raw_mask, colors["pink 2"]),
        ("Post-processed", post_mask, colors["pink 3"]),
    ]

    for i, (col_title, mask, color) in enumerate(columns):
        mask_bin = (mask > 0).astype(float)

        overlay = np.zeros((*mask_bin.shape, 4), dtype=float)
        overlay[..., :3] = color
        overlay[..., 3] = mask_bin * 0.45   # alpha only where mask exists

        axes[i].imshow(image, cmap="gray")
        axes[i].imshow(overlay)
        axes[i].set_title(col_title)
        axes[i].axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()
    
    
def plot_mask_venn(gt_mask, pred_mask, title=None, al=False, loc_text="upper right"):
    """
    Visualizes overlap between GT and prediction masks.
    """

    # Sørg for binære masker
    gt = (gt_mask > 0)
    pred = (pred_mask > 0)

    # Regions
    tp = gt & pred          # overlap
    fn = gt & ~pred         # GT only
    fp = ~gt & pred         # Pred only

    # Lag RGB bilde
    h, w = gt.shape
    venn = np.ones((h, w, 3))  # hvit bakgrunn

    # 🎨 Farger (0–1 range)
    color_gt_only = colors["pink 1"]   
    color_pred_only = colors["pink 3"] 
    color_overlap = colors["purple 1"]    

    # Apply colors
    venn[fn] = color_gt_only
    venn[fp] = color_pred_only
    venn[tp] = color_overlap
    legend_elements = [
    Patch(facecolor=color_gt_only, label="GT only (FN)"),
    Patch(facecolor=color_pred_only, label="Pred only (FP)"),
    Patch(facecolor=color_overlap, label="Overlap (TP)"),
    ]
    
    if al:
        legend_elements = [
            Patch(facecolor=color_overlap, label="Overlap"),
            Patch(facecolor=color_gt_only, label="Raw mask"),
            Patch(facecolor=color_pred_only, label="Post-processed "),
        ]

    # Plot
    plt.figure(figsize=(6, 6))
    

    plt.legend(
        handles=legend_elements,
        loc=loc_text,
        frameon=True,
        fontsize=16,
    )
    
    plt.imshow(venn)
    plt.axis("off")

    if title:
        plt.title(title)

    plt.show()


def evaluate_original_dataset(model, device, csv_path):
    model.eval()

    dataset = FullOriginalDataset(
        image_dir=ORIG_IMAGE_DIR,
        mask_dir=ORIG_MASK_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rows = []

    with torch.no_grad():
        for image, mask, filename in loader:
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()

            dice = dice_score(pred, mask).item()

            rows.append({
                "filename": filename[0],
                "dice": dice,
            })

            print(f"{filename[0]} | Dice = {dice:.4f}")

    # Lagre CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    scores = [r["dice"] for r in rows]

    print("\nSummary:")
    print(f"Mean Dice: {np.mean(scores):.4f}")
    print(f"Min Dice:  {np.min(scores):.4f}")
    print(f"Max Dice:  {np.max(scores):.4f}")

    print(f"\nSaved to: {csv_path}")
    
def evaluate_original_dataset(model, device, csv_path):
    model.eval()

    dataset = FullOriginalDataset(
        image_dir=ORIG_IMAGE_DIR,
        mask_dir=ORIG_MASK_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rows = []

    with torch.no_grad():
        for image, mask, filename in loader:
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()

            dice = dice_score(pred, mask).item()

            rows.append({
                "filename": filename[0],
                "dice": dice,
            })

            print(f"{filename[0]} | Dice = {dice:.4f}")

    # Lagre CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "dice"])
        writer.writeheader()
        writer.writerows(rows)

    scores = [r["dice"] for r in rows]

    print("\nSummary:")
    print(f"Mean Dice: {np.mean(scores):.4f}")
    print(f"Min Dice:  {np.min(scores):.4f}")
    print(f"Max Dice:  {np.max(scores):.4f}")

    print(f"\nSaved to: {csv_path}")


from pathlib import Path
import matplotlib.pyplot as plt


def plot_training_curves(history, save_dir=None):

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # -------------------
    # Loss
    # -------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train loss", color=colors["pink 2"])
    plt.plot(history["val_loss"], label="Validation loss", color=colors["purple 1"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(save_dir / "loss_curve.png", dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------
    # Dice
    # -------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_dice"], label="Train Dice", color=colors["pink 2"])
    plt.plot(history["val_dice"], label="Validation Dice", color=colors["purple 1"])
    plt.xlabel("Epoch")
    plt.ylabel("Dice score")
    plt.title("Training and validation Dice score")
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(save_dir / "dice_curve.png", dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------
    # IoU
    # -------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_iou"], label="Train IoU", color=colors["pink 2"])
    plt.plot(history["val_iou"], label="Validation IoU", color=colors["purple 1"])
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Training and validation IoU")
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(save_dir / "iou_curve.png", dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------
    # Precision / Recall
    # -------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["val_precision"], label="Validation precision", color=colors["pink 1"])
    plt.plot(history["val_recall"], label="Validation recall", color=colors["purple 2"])
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation precision and recall")
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(save_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")

    plt.show()
    
    
    ##################
    
    from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.metrics import dice_score, iou_score, precision_score, recall_score, pixel_accuracy
from src.mask_utils import to_tensor_mask
from src.process_mask import post_process_mask
from src.mask_analysis import diameter_stats, diameter_error_stats


def _extract_mask(pred_result):
    """Handle predict_full_image returning either dict or array."""
    if isinstance(pred_result, dict):
        for key in ["mask", "pred_mask", "raw_mask", "prediction"]:
            if key in pred_result:
                return pred_result[key]
        raise KeyError(f"Could not find mask in predict_full_image output. Keys: {pred_result.keys()}")
    return pred_result


def _as_bool_mask(mask, threshold=0.5):
    """Convert tensor/array/PIL mask to boolean numpy mask."""
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    elif hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    elif isinstance(mask, Image.Image):
        mask = np.array(mask)

    mask = np.squeeze(np.array(mask))

    if mask.dtype == bool:
        return mask

    if mask.max() > 1:
        return mask > 127

    return mask > threshold


def _match_shape(mask, target_shape):
    """
    Resize mask to target_shape = (height, width).
    PIL resize expects (width, height).
    """
    mask = np.squeeze(mask)

    if mask.shape == target_shape:
        return mask

    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize(
        (target_shape[1], target_shape[0]),
        resample=Image.NEAREST
    )

    return np.array(mask_img) > 0


def evaluate_dataset2(
    model,
    input_dir,
    mask_dir,
    threshold=0.5,
    resize_to=None,          # viktig: default None for full-size evaluation
    microns_per_pixel=None,
    min_area_px=10,
):
    input_dir = Path(input_dir)
    mask_dir = Path(mask_dir)

    rows = []
    gt_masks = {}
    raw_masks = {}
    post_masks = {}
    all_masks = {}

    image_paths = sorted(
        list(input_dir.glob("*.tif")) +
        list(input_dir.glob("*.tiff")) +
        list(input_dir.glob("*.png")) +
        list(input_dir.glob("*.jpg")) +
        list(input_dir.glob("*.jpeg"))
    )

    for img_path in image_paths:
        mask_path = mask_dir / img_path.name

        if not mask_path.exists():
            print(f"Skipping {img_path.name}: no matching mask found")
            continue

        gt = np.array(Image.open(mask_path).convert("L")) > 0

        # Optional resizing, only if you really want 256x256-style eval
        if resize_to is not None:
            gt = np.array(
                Image.fromarray(gt.astype(np.uint8) * 255).resize(
                    resize_to, 
                    resample=Image.NEAREST
                )
            ) > 0

        pred_result = predict_full_image(
            model,
            img_path,
            postprocess=False,
            plot = False
        )

        pred_raw = _extract_mask(pred_result)
        pred_raw = _as_bool_mask(pred_raw, threshold=threshold)

        # Make prediction same shape as GT
        pred_raw = _match_shape(pred_raw, gt.shape)

        pred_post = post_process_mask(
            pred_raw
        )

        pred_post = _as_bool_mask(pred_post, threshold=threshold)
        pred_post = _match_shape(pred_post, gt.shape)

        gt_t = to_tensor_mask(gt)
        raw_t = to_tensor_mask(pred_raw)
        post_t = to_tensor_mask(pred_post)

        dice_raw = dice_score(raw_t, gt_t).item()
        iou_raw = iou_score(raw_t, gt_t).item()
        precision_raw = precision_score(raw_t, gt_t).item()
        recall_raw = recall_score(raw_t, gt_t).item()
        pixel_acc_raw = pixel_accuracy(raw_t, gt_t).item()

        dice_post = dice_score(post_t, gt_t).item()
        iou_post = iou_score(post_t, gt_t).item()
        precision_post = precision_score(post_t, gt_t).item()
        recall_post = recall_score(post_t, gt_t).item()
        pixel_acc_post = pixel_accuracy(post_t, gt_t).item()
        
        gt_diameters = diameters_from_mask(gt,microns_per_pixel=microns_per_pixel)

        raw_diameters = diameters_from_mask(pred_raw, microns_per_pixel=microns_per_pixel)

        post_diameters = diameters_from_mask(pred_post, microns_per_pixel=microns_per_pixel)
                

        gt_stats = diameter_stats(gt_diameters)
        raw_stats = diameter_stats(raw_diameters)
        post_stats = diameter_stats(post_diameters)
        
        raw_err = diameter_error_stats(gt_diameters, raw_diameters)
        post_err = diameter_error_stats(gt_diameters, post_diameters)
        gt_water_percent = gt.mean() * 100
        raw_water_percent = pred_raw.mean() * 100
        post_water_percent = pred_post.mean() * 100

        rows.append({
            "image": img_path.name,

            "dice_raw": dice_raw,
            "iou_raw": iou_raw,
            "precision_raw": precision_raw,
            "recall_raw": recall_raw,
            "pixel_acc_raw": pixel_acc_raw,

            "dice_post": dice_post,
            "iou_post": iou_post,
            "precision_post": precision_post,
            "recall_post": recall_post,
            "pixel_acc_post": pixel_acc_post,

            "gt_count": gt_stats["count"],
            "raw_count": raw_stats["count"],
            "post_count": post_stats["count"],

            "gt_mean_diameter": gt_stats["mean"],
            "raw_mean_diameter": raw_stats["mean"],
            "post_mean_diameter": post_stats["mean"],

            "gt_median_diameter": gt_stats["median"],
            "raw_median_diameter": raw_stats["median"],
            "post_median_diameter": post_stats["median"],

            "gt_min_diameter": gt_stats["min"],
            "raw_min_diameter": raw_stats["min"],
            "post_min_diameter": post_stats["min"],

            "gt_max_diameter": gt_stats["max"],
            "raw_max_diameter": raw_stats["max"],
            "post_max_diameter": post_stats["max"],

            "rmse_raw": raw_err["rmse"],
            "rmse_post": post_err["rmse"],
            "mae_raw": raw_err["mae"],
            "mae_post": post_err["mae"],
            "bias_raw": raw_err["bias"],
            "bias_post": post_err["bias"],

            "gt_water_percent": gt_water_percent,
            "raw_water_percent": raw_water_percent,
            "post_water_percent": post_water_percent,

            "gt_water_oil_ratio": gt_water_percent / max(100 - gt_water_percent, 1e-8),
            "raw_water_oil_ratio": raw_water_percent / max(100 - raw_water_percent, 1e-8),
            "post_water_oil_ratio": post_water_percent / max(100 - post_water_percent, 1e-8),
        })

        gt_masks[img_path.name] = gt
        raw_masks[img_path.name] = pred_raw
        post_masks[img_path.name] = pred_post
        all_masks[img_path.name] = {
            "gt": gt,
            "raw": pred_raw,
            "post": pred_post,
        }

        print(
            f"{img_path.name}: "
            f"GT {gt.shape}, raw {pred_raw.shape}, "
            f"Dice raw={dice_raw:.3f}, Dice post={dice_post:.3f}"
        )

    df = pd.DataFrame(rows)

    return df, gt_masks, raw_masks, post_masks, all_masks

def save_predictions_and_overlays(
    model,
    model_name,
    image_dir,
    save_dir,
    threshold=0.5,
    device=None,
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
):



    image_dir = Path(image_dir)
    save_dir = Path(save_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    output_dir = save_dir / model_name

    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"

    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_paths)} images")
    print(f"Saving to: {output_dir}")

    for img_path in tqdm(image_paths, desc=f"Predicting with {model_name}"):

        # predict_full_image returns DICT
        result = predict_full_image(
            model=model,
            image_path=img_path,
            threshold=threshold,
            plot=False,
            postprocess=False,
        )

        pred_mask = result["final_mask"]

        pred_mask_uint8 = (pred_mask > 0).astype(np.uint8) * 255

        # Save mask
        mask_save_path = (
            mask_dir /
            f"{img_path.stem}_{model_name}_mask.png"
        )

        Image.fromarray(pred_mask_uint8).save(mask_save_path)

        # Original image
        original = Image.open(img_path).convert("L")
        original_np = np.array(original)

        # Overlay
        overlay = np.stack(
            [original_np]*3,
            axis=-1
        )

        pink = np.array([210, 0, 105], dtype=np.uint8)

        mask_bool = pred_mask_uint8 > 0

        overlay[mask_bool] = (
            0.6 * overlay[mask_bool] +
            0.4 * pink
        ).astype(np.uint8)

        overlay_save_path = (
            overlay_dir /
            f"{img_path.stem}_{model_name}_overlay.png"
        )

        Image.fromarray(overlay).save(overlay_save_path)

    print("Done saving predictions and overlays.")
    
def measure_average_inference_time(
    model,
    image_dir,
    threshold=0.5,
    postprocess=False,
    warmup=2,
    save_csv_path=None,
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ):
    image_dir = Path(image_dir)

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Warmup, so first prediction does not distort timing
    for img_path in image_paths[:warmup]:
        _ = predict_full_image(
            model=model,
            image_path=img_path,
            threshold=threshold,
            postprocess=postprocess,
            plot=False,
        )

    rows = []

    for img_path in tqdm(image_paths, desc="Measuring inference time"):
        start = time.perf_counter()

        _ = predict_full_image(
            model=model,
            image_path=img_path,
            threshold=threshold,
            postprocess=postprocess,
            plot=False,
        )

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        rows.append({
            "image": img_path.name,
            "inference_time_sec": end - start,
        })

    df = pd.DataFrame(rows)

    print(f"Images: {len(df)}")
    print(f"Mean inference time: {df['inference_time_sec'].mean():.4f} s/image")
    print(f"Std inference time:  {df['inference_time_sec'].std():.4f} s")
    print(f"Min inference time:  {df['inference_time_sec'].min():.4f} s")
    print(f"Max inference time:  {df['inference_time_sec'].max():.4f} s")

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"Saved to: {save_csv_path}")

    return df

def create_dsd_comparison_dataframe(
    masks,
    annotated_dir,
    pred_key="raw",
    microns_per_pixel=None,
):
    """
    Creates dataframe comparing GT and predicted DSD statistics.

    Parameters
    ----------
    masks : dict
        Dictionary like:
        masks[img_name]["gt"]
        masks[img_name]["raw"]

    annotated_dir : Path
        Folder containing annotated images.

    pred_key : str
        Which prediction to compare against GT.
        Example: "raw" or "post"

    Returns
    -------
    df : pandas.DataFrame
    """

    rows = []

    for img_path in annotated_dir.iterdir():

        img_name = img_path.name

        if img_name not in masks:
            continue

        gt_mask = masks[img_name]["gt"]
        pred_mask = masks[img_name][pred_key]

        D_gt = DSD_from_mask(
        gt_mask,
        microns_per_pixel=microns_per_pixel
        )

        D_pred = DSD_from_mask(
        pred_mask,
        microns_per_pixel=microns_per_pixel
        )

        row = {
            "image": img_name
        }

        for key in D_gt.keys():

            gt_val = D_gt[key]
            pred_val = D_pred[key]

            row[f"gt_{key}"] = gt_val
            row[f"pred_{key}"] = pred_val
            row[f"error_{key}"] = pred_val - gt_val

        rows.append(row)

    df = pd.DataFrame(rows)

    return df

def summarize_dsd_errors(
    df,
    metrics=None,
):
    """
    Creates global DSD error summary dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from create_dsd_comparison_dataframe()

    metrics : list or None
        Metrics to evaluate.

    Returns
    -------
    summary_df : pandas.DataFrame
    """

    if metrics is None:
        metrics = [
            "mean",
            "median",
            "std",
            "d10",
            "d50",
            "d90",
            "span",
            "d32",
            "cv"
        ]

    summary_rows = []

    for metric in metrics:

        errors = df[f"error_{metric}"].dropna()

        summary_rows.append({
            "metric": metric,

            "mse": np.mean(errors**2),

            "rmse": np.sqrt(
                np.mean(errors**2)
            ),

            "mae": np.mean(
                np.abs(errors)
            ),

            "bias": np.mean(errors),

            "std_error": np.std(errors),
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df