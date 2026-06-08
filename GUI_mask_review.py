import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import csv

from src.create_review_dataset import (
    accept_gt,
    accept_raw_pred,
    accept_post_pred,
    send_to_manual,
)
from src.mask_utils import (
    predict_mask,
    load_mask,
    to_tensor_mask,
    save_mask,
    make_overlay,
)

from src.process_mask import post_process_mask

"""Before running, run create_review_dataset.py with the model you want to use!! 
    Run by writing : "cd C:/Users/47469/master_enviorment/FinalModel"
    and then "python -m src.create_review_dataset"
"""

REVIEW_FOLDER = r"C:\Users\47469\master_enviorment\FinalModel\Review"


class MaskReviewGUI:
    def __init__(self, root, review_path):
        self.root = root
        self.review_path = Path(review_path)
        self.cases = sorted((self.review_path / "cases").iterdir())
        self.idx = 0
        self.tk_imgs = []

        self.root.title("Mask Review GUI")
        self.root.geometry("900x650")
        self.root.minsize(950, 650)

        if len(self.cases) == 0:
            tk.Label(
                root,
                text="No cases found. Run create_review_dataset.py first.",
                font=("Arial", 14)
            ).pack(pady=30)
            return

        self.dice_scores = self.load_dice_scores()

        # Top info
        self.header_label = tk.Label(root, text="", font=("Arial", 14, "bold"))
        self.header_label.pack(pady=(8, 2))

        self.info_label = tk.Label(root, text="", font=("Arial", 11))
        self.info_label.pack(pady=(0, 8))

        # Image area
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        self.column_titles = ["Ground truth", "Raw prediction", "Post-processed prediction"]

        for col, title in enumerate(self.column_titles):
            tk.Label(
                self.image_frame,
                text=title,
                font=("Arial", 13, "bold")
            ).grid(row=0, column=col, padx=10, pady=(0, 6))

        self.mask_labels = []
        self.overlay_labels = []

        for col in range(3):
            mask_label = tk.Label(self.image_frame, bd=1, relief="solid")
            mask_label.grid(row=1, column=col, padx=10, pady=6)
            self.mask_labels.append(mask_label)

            overlay_label = tk.Label(self.image_frame, bd=1, relief="solid")
            overlay_label.grid(row=2, column=col, padx=10, pady=6)
            self.overlay_labels.append(overlay_label)

        # Main decision buttons
        self.decision_frame = tk.Frame(root)
        self.decision_frame.pack(pady=(14, 4))

        tk.Button(
            self.decision_frame,
            text="Keep GT",
            command=self.keep_gt,
            width=22,
            height=2,
            font=("Arial", 10, "bold")
        ).grid(row=0, column=0, padx=8, pady=4)

        tk.Button(
            self.decision_frame,
            text="Use Raw Prediction",
            command=self.keep_raw_pred,
            width=22,
            height=2,
            font=("Arial", 10, "bold")
        ).grid(row=0, column=1, padx=8, pady=4)

        tk.Button(
            self.decision_frame,
            text="Use Post Prediction",
            command=self.keep_post_pred,
            width=22,
            height=2,
            font=("Arial", 10, "bold")
        ).grid(row=0, column=2, padx=8, pady=4)

        tk.Button(
            self.decision_frame,
            text="Manual Redo",
            command=self.manual_redo,
            width=24,
            height=2,
            font=("Arial", 10, "bold")
        ).grid(row=1, column=0, columnspan=3, pady=(6, 4))

        # Navigation buttons
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack(pady=(4, 10))

        tk.Button(
            self.nav_frame,
            text="← Previous",
            command=self.previous_case,
            width=16
        ).grid(row=0, column=0, padx=6)

        tk.Button(
            self.nav_frame,
            text="Next →",
            command=self.next_case,
            width=16
        ).grid(row=0, column=1, padx=6)

        # Navigation buttons
        self.nav_frame = tk.Frame(root)
        self.nav_frame.pack(pady=(4, 10))

        self.show_case()

    def load_dice_scores(self):
        log_path = self.review_path / "review_log.csv"
        scores = {}

        if not log_path.exists():
            return scores

        with open(log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores[row["case_name"]] = {
                    "raw": row.get("dice_raw", "N/A"),
                    "post": row.get("dice_postprocessed", "N/A"),
                }

        return scores

    def load_preview(self, image_path, size=(180, 180)):
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size)
        return ImageTk.PhotoImage(img)

    def show_case(self):
        case = self.cases[self.idx]

        mask_paths = [
            case / "gt_mask.png",
            case / "pred_mask_raw.png",
            case / "pred_mask_postprocessed.png",
        ]

        overlay_paths = [
            case / "gt_overlay.png",
            case / "pred_overlay_raw.png",
            case / "pred_overlay_postprocessed.png",
        ]

        self.tk_imgs = []

        for i in range(3):
            mask_img = self.load_preview(mask_paths[i])
            overlay_img = self.load_preview(overlay_paths[i])

            self.tk_imgs.extend([mask_img, overlay_img])

            self.mask_labels[i].config(image=mask_img)
            self.overlay_labels[i].config(image=overlay_img)

        dice = self.dice_scores.get(case.name, {"raw": "N/A", "post": "N/A"})

        self.header_label.config(
            text=f"{self.idx + 1}/{len(self.cases)}: {case.name}"
        )

        self.info_label.config(
            text=f"Raw Dice: {dice['raw']}   |   Post-processed Dice: {dice['post']}"
        )

    def keep_gt(self):
        accept_gt(self.cases[self.idx], self.review_path)
        self.next_case()

    def keep_raw_pred(self):
        accept_raw_pred(self.cases[self.idx], self.review_path)
        self.next_case()

    def keep_post_pred(self):
        accept_post_pred(self.cases[self.idx], self.review_path)
        self.next_case()

    def manual_redo(self):
        send_to_manual(self.cases[self.idx], self.review_path)
        self.next_case()

    def next_case(self):
        if self.idx < len(self.cases) - 1:
            self.idx += 1
            self.show_case()
        else:
            self.info_label.config(text="Finished reviewing all cases.")

    def previous_case(self):
        if self.idx > 0:
            self.idx -= 1
            self.show_case()


if __name__ == "__main__":
    root = tk.Tk()
    app = MaskReviewGUI(root, REVIEW_FOLDER)
    root.mainloop()