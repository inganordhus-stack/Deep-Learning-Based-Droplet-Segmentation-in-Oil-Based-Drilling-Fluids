"""
Evaluation metrics for binary image segmentation.
All metrics operate on batch tensors of shape (B, 1, H, W).
"""

import torch


def get_binary_preds(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    return preds

def dice_score(preds, targets, eps=1e-7):
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()

def iou_score(preds, targets, eps=1e-7):
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

def precision_score(preds, targets, eps=1e-7):
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fp + eps)).mean()

def recall_score(preds, targets, eps=1e-7):
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + fn + eps)).mean()

def pixel_accuracy(preds, targets):
    correct = (preds == targets).sum()
    total = torch.numel(preds)
    return correct.float() / total


def compute_all_metrics(logits, targets):
    preds = get_binary_preds(logits)
    targets = targets.float()

    return {
        "dice": dice_score(preds, targets).item(),
        "iou": iou_score(preds, targets).item(),
        "precision": precision_score(preds, targets).item(),
        "recall": recall_score(preds, targets).item(),
        "pixel_accuracy": pixel_accuracy(preds, targets).item()
    }