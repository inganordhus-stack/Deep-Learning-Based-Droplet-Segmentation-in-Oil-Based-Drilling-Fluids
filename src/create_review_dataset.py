from pathlib import Path
import shutil
import csv

from PIL import Image

from src.metrics import dice_score
from src.model import load_model

from src.mask_utils import (
    predict_mask,
    load_mask,
    to_tensor_mask,
    save_mask,
    make_overlay,
)

from src.process_mask import post_process_mask

# How to run this script:
    #In terminal:
        # "cd "FinalModel" "
        # "python -m src.create_review_dataset"
# python -m src.create_review_dataset in terminal to run this script. Make sure to update the MODEL_PATH and SOURCE_FOLDER variables to point to your model and dataset respectively before running. This will create a review dataset in the REVIEW_FOLDER path, which can then be reviewed using the GUI_mask_review.py script.
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "Latest_Models" / "aug_light_full" / "best_model.pth"
SOURCE_FOLDER = BASE_DIR / "Original"
REVIEW_FOLDER = BASE_DIR / "Review"

model = load_model(MODEL_PATH)


def create_review_container(model, folder_path, review_path, dice_threshold=0.6):
    folder_path = Path(folder_path)
    review_path = Path(review_path)
    
    if review_path.exists():
        shutil.rmtree(review_path)

    review_path.mkdir(parents=True, exist_ok=True)
    
    

    input_folder = folder_path / "Input"
    gt_folder = folder_path / "Output" / "Masks"

    cases_folder = review_path / "cases"
    cases_folder.mkdir(parents=True, exist_ok=True)

    (review_path / "accepted_gt").mkdir(exist_ok=True)
    (review_path / "accepted_raw_pred").mkdir(exist_ok=True)
    (review_path / "accepted_post_pred").mkdir(exist_ok=True)
    (review_path / "manual_redo").mkdir(exist_ok=True)

    log_path = review_path / "review_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "case_name",
            "dice_raw",
            "dice_postprocessed",
            "original_path",
            "gt_mask_path",
            "raw_pred_mask_path",
            "post_pred_mask_path",
            "decision"
        ])

        for img_path in input_folder.iterdir():
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                continue

            gt_mask_path = gt_folder / img_path.name

            if not gt_mask_path.exists():
                print(f"No GT mask found for {img_path.name}")
                continue

            raw_pred_mask = predict_mask(model, img_path)
            post_pred_mask = post_process_mask(raw_pred_mask)

            gt_mask = load_mask(gt_mask_path, target_size=raw_pred_mask.shape)

            raw_t = to_tensor_mask(raw_pred_mask)
            post_t = to_tensor_mask(post_pred_mask)
            gt_t = to_tensor_mask(gt_mask)

            dice_raw = dice_score(raw_t, gt_t).item()
            dice_post = dice_score(post_t, gt_t).item()

            # Add to review if either raw or post-processed Dice is below threshold
            if min(dice_raw, dice_post) >= dice_threshold:
                continue

            case_name = img_path.stem
            case_folder = cases_folder / case_name
            case_folder.mkdir(exist_ok=True)

            original_out = case_folder / "original.png"
            gt_out = case_folder / "gt_mask.png"
            raw_pred_out = case_folder / "pred_mask_raw.png"
            post_pred_out = case_folder / "pred_mask_postprocessed.png"

            original_img = Image.open(img_path).convert("L")
            original_img.save(original_out)

            save_mask(gt_mask, gt_out)
            save_mask(raw_pred_mask, raw_pred_out)
            save_mask(post_pred_mask, post_pred_out)

            make_overlay(original_img, gt_mask).save(case_folder / "gt_overlay.png")
            make_overlay(original_img, raw_pred_mask).save(case_folder / "pred_overlay_raw.png")
            make_overlay(original_img, post_pred_mask).save(case_folder / "pred_overlay_postprocessed.png")

            writer.writerow([
                case_name,
                f"{dice_raw:.4f}",
                f"{dice_post:.4f}",
                original_out,
                gt_out,
                raw_pred_out,
                post_pred_out,
                ""
            ])

            print(
                f"Added to review: {case_name} | "
                f"Raw Dice: {dice_raw:.4f} | "
                f"Post Dice: {dice_post:.4f}"
            )


def accept_gt(case_folder, review_path):
    case_folder = Path(case_folder)
    review_path = Path(review_path)

    shutil.copy(
        case_folder / "gt_mask.png",
        review_path / "accepted_gt" / f"{case_folder.name}.png"
    )


def accept_raw_pred(case_folder, review_path):
    case_folder = Path(case_folder)
    review_path = Path(review_path)

    shutil.copy(
        case_folder / "pred_mask_raw.png",
        review_path / "accepted_raw_pred" / f"{case_folder.name}.png"
    )


def accept_post_pred(case_folder, review_path):
    case_folder = Path(case_folder)
    review_path = Path(review_path)

    shutil.copy(
        case_folder / "pred_mask_postprocessed.png",
        review_path / "accepted_post_pred" / f"{case_folder.name}.png"
    )


def send_to_manual(case_folder, review_path):
    case_folder = Path(case_folder)
    review_path = Path(review_path)

    shutil.copy(
        case_folder / "original.png",
        review_path / "manual_redo" / f"{case_folder.name}.png"
    )


if __name__ == "__main__":
    create_review_container(
        model,
        SOURCE_FOLDER,
        REVIEW_FOLDER,
        dice_threshold=0.6 # change this threshold to include more or fewer cases in the review dataset. A higher threshold will include more cases with better predictions, while a lower threshold will focus on cases where the model struggled more.
    )