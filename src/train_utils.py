import time
from IPython.display import display, HTML
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.config import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_DIR,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BEST_MODEL_PATH,
    LAST_MODEL_PATH,
)

from src.metrics import (
    get_binary_preds,
    dice_score,
    iou_score,
    pixel_accuracy,
    precision_score,
    recall_score,
)


# --------------------
# Loss function
# --------------------
bce_loss = torch.nn.BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

def get_model_paths(run_name):
    run_dir = MODEL_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best_model.pth"
    last_path = run_dir / "last_model.pth"

    return best_path, last_path


def loss_fn(logits, masks):
    return 0.5 * bce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)


# --------------------
# Optional boundary loss
# --------------------
def edge_map(x):
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return dx.mean() + dy.mean()


def boundary_loss(pred, target):
    pred_edges = edge_map(pred)
    target_edges = edge_map(target)
    return torch.abs(pred_edges - target_edges)


# --------------------
# Checkpoint
# --------------------
def save_checkpoint(model, optimizer, scheduler, epoch, path, best_val_dice=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if best_val_dice is not None:
        checkpoint["best_val_dice_score"] = best_val_dice

    torch.save(checkpoint, path)


# --------------------
# Train one epoch
# --------------------
def train_one_epoch(model, train_loader, optimizer, epoch):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_pixel_acc = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", leave=False)

    for images, masks in pbar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).float()

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        preds = get_binary_preds(logits)

        batch_dice = dice_score(preds, masks).item()
        batch_iou = iou_score(preds, masks).item()
        batch_precision = precision_score(preds, masks).item()
        batch_recall = recall_score(preds, masks).item()
        pixel_acc = pixel_accuracy(preds, masks).item()

        running_loss += loss.item()
        running_dice += batch_dice
        running_iou += batch_iou
        running_precision += batch_precision
        running_recall += batch_recall
        running_pixel_acc += pixel_acc

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice:.4f}",
            iou=f"{batch_iou:.4f}",
        )

    n_batches = len(train_loader)

    return {
        "loss": running_loss / n_batches,
        "dice": running_dice / n_batches,
        "iou": running_iou / n_batches,
        "precision": running_precision / n_batches,
        "recall": running_recall / n_batches,
        "pixel_acc": running_pixel_acc / n_batches,
    }


# --------------------
# Validate one epoch
# --------------------
def validate_one_epoch(model, val_loader, epoch):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_recall = 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            logits = model(images)
            loss = loss_fn(logits, masks)

            preds = get_binary_preds(logits)

            batch_dice = dice_score(preds, masks).item()
            batch_iou = iou_score(preds, masks).item()
            batch_precision = precision_score(preds, masks).item()
            batch_recall = recall_score(preds, masks).item()

            running_loss += loss.item()
            running_dice += batch_dice
            running_iou += batch_iou
            running_precision += batch_precision
            running_recall += batch_recall

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{batch_dice:.4f}",
                iou=f"{batch_iou:.4f}",
            )

    n_batches = len(val_loader)

    return {
        "loss": running_loss / n_batches,
        "dice": running_dice / n_batches,
        "iou": running_iou / n_batches,
        "precision": running_precision / n_batches,
        "recall": running_recall / n_batches,
    }


# --------------------
# Full training loop
# --------------------
def train_model(model, train_loader, val_loader, run_name="U-net", notes="", epochs=NUM_EPOCHS): 
    BEST_MODEL_PATH, LAST_MODEL_PATH = get_model_paths(run_name)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=0.002,
        threshold_mode="abs",
        min_lr=1e-6,
    )

    min_delta = 0.002
    patience = 6
    epochs_without_improvement = 0
    best_val_dice = -float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    start_total_time = time.time()

    for epoch in range(epochs):
        start_epoch_time = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, epoch)
        val_metrics = validate_one_epoch(model, val_loader, epoch)

        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_epoch_time

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            path=LAST_MODEL_PATH,
            best_val_dice=best_val_dice,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_metrics["dice"] > best_val_dice + min_delta:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                path=BEST_MODEL_PATH,
                best_val_dice=best_val_dice,
            )

            print("Saved best model")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break
    total_time = time.time() - start_total_time
    mean_epoch_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
    print(f"Total training time: {total_time // 60:.0f} minutes, {total_time % 60:.2f} seconds")
        # --------------------
    # Save run statistics
    # --------------------
    run_stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs_planned": epochs,
        "run_name": run_name,
        "notes": notes,
        "epochs_completed": len(history["train_loss"]),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "early_stopping_patience": patience,
        "min_delta": min_delta,
        "best_val_dice": best_val_dice,
        "best_val_loss": min(history["val_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_dice": history["train_dice"][-1],
        "final_val_dice": history["val_dice"][-1],
        "final_train_iou": history["train_iou"][-1],
        "final_val_iou": history["val_iou"][-1],
        "final_val_precision": history["val_precision"][-1],
        "final_val_recall": history["val_recall"][-1],
        "total_training_time_sec": round(total_time, 2),
        "best_model_path": str(BEST_MODEL_PATH),
        "last_model_path": str(LAST_MODEL_PATH),
        "mean_epoch_time_sec": round(mean_epoch_time, 2),
        "total_training_time_sec": round(total_time, 2),
        "total_training_time_min": round(total_time / 60, 2),
    }

    csv_path = Path("runs/training_runs.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([run_stats])

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(csv_path, index=False)
    print(f"Saved run statistics to {csv_path}")
    

    return history

def show_training_log(history):
    lines = []

    n_epochs = len(history["train_loss"])

    for i in range(n_epochs):
        line = (
            f"Epoch {i+1}/{n_epochs} | "
            f"Train Loss: {history['train_loss'][i]:.4f} | "
            f"Train Dice Score: {history['train_dice'][i]:.4f} | "
            f"Train IoU: {history['train_iou'][i]:.4f} | "
            f"Train Precision: {history['train_precision'][i]:.4f} | "
            f"Train Recall: {history['train_recall'][i]:.4f} | "
            f"Val Loss: {history['val_loss'][i]:.4f} | "
            f"Val Dice Score: {history['val_dice'][i]:.4f} | "
            f"Val IoU: {history['val_iou'][i]:.4f} | "
            f"Val Precision: {history['val_precision'][i]:.4f} | "
            f"Val Recall: {history['val_recall'][i]:.4f} | "
            f"LR: {history['lr'][i]:.6f}"
        )
        lines.append(line)

    html = f"""
    <div style="
        max-height: 350px;
        overflow-y: auto;
        border: 1px solid #444;
        padding: 10px;
        background-color: #111;
        color: #eaeaea;
        font-family: Consolas, monospace;
        font-size: 13px;
        white-space: pre;
    ">
    {chr(10).join(lines)}
    </div>
    """

    display(HTML(html))


