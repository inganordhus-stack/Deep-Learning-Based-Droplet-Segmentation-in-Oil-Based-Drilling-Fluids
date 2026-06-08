from pathlib import Path
"""
    Run the training process for the model. This script will:
    1. Load the training and validation data
    2. Create the model
    3. Train the model
    4. Save the trained model and training history
    5. Plot training curves"""

import pandas as pd
from src.config import DEVICE, MODEL_DIR, IMAGE_DIR, MASK_DIR
from src.dataset import make_train_val_loaders
from src.model import create_model, load_model
from src.train_utils import train_model, show_training_log
from src.evaluation import plot_training_curves, evaluate_original_dataset


def main():
    run_name = "NORMAL_no_aug"
    notes = "Full dataset, no augmentations"

    assert Path(IMAGE_DIR).exists(), f"Image folder not found: {IMAGE_DIR}"
    assert Path(MASK_DIR).exists(), f"Mask folder not found: {MASK_DIR}"

    print(f"Device: {DEVICE}")
    print(f"Run name: {run_name}")

    train_loader, val_loader = make_train_val_loaders()

    model = create_model()

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        run_name=run_name,
        notes=notes,
        epochs=1,
    )

    run_dir = MODEL_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_path = run_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved history to: {history_path}")

    plot_training_curves(history, save_dir=run_dir)

    best_model_path = run_dir / "best_model.pth"
    csv_path = run_dir / "dice_scores_full_dataset.csv"

    best_model = load_model(best_model_path)

    

    show_training_log(history)


if __name__ == "__main__":
    main()