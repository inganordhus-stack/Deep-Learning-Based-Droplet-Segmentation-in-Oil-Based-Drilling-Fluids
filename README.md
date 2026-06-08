# Deep Learning-Based Droplet Segmentation in Oil-Based Drilling Fluids

This repository contains the source code developed as part of a Master's thesis investigating deep learning-based segmentation of water droplets in oil-based drilling fluids (OBDFs). The objective of the work was to develop an automated workflow for extracting droplet size distributions (DSDs) from optical microscopy images using semantic segmentation.

The project uses a U-Net architecture with a ResNet34 encoder implemented in PyTorch and segmentation_models_pytorch. The trained models are used to segment dispersed water droplets in microscopy images of OBDF emulsions, after which droplet size statistics such as mean diameter, D10, D50, D90, D32, span, coefficient of variation, and dispersed-phase fraction can be estimated automatically.

## Selected Results

The best-performing model achieved the following average segmentation performance on the held-out test dataset:

| Metric | Value |
|----------|----------|
| Dice Score | 0.797 |
| IoU | 0.667 |
| Precision | 0.829 |
| Recall | 0.786 |
| Pixel Accuracy | 0.961 |

The model was trained on manually annotated microscopy images of oil-based drilling fluids and demonstrated robustness across different droplet morphologies and imaging conditions. Additional experiments showed improved generalization when stronger intensity-based augmentations were applied during training.

## Example Prediction

![Example segmentation](362_HAD_test_overlay.png)

Example prediction (pink) overlaid in the corresponding microscopy image.

## Key Findings

- Deep learning-based semantic segmentation can successfully identify water droplets in OBDF microscopy images.
- The best model achieved a mean Dice score of 0.80 on the test dataset.
- Strong image augmentations improved robustness and generalization.
- The developed workflow enables automatic extraction of droplet size distributions (DSDs) from microscopy images in approximately 2 seconds per full-scale image.


## Repository Structure

```text
.
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── mask_analysis.py
│   ├── mask_utils.py
│   ├── metrics.py
│   ├── model.py
│   ├── process_mask.py
│   ├── train_utils.py
│   └── create_review_dataset.py
│
├── run_training.py
├── GUI_mask_review.py
├── evaluation_on_validation_set.ipynb
├── test_model.ipynb
├── requirements.txt
└── .gitignore
```

## Main Components

### Training

The script:

```bash
python run_training.py
```

contains the training workflow used for model development.

The implementation includes:

* U-Net with ResNet34 encoder
* Transfer learning support
* Dice + BCE loss
* Learning rate scheduling
* Early stopping
* Data augmentation
* Model checkpointing

### Evaluation

The files:

```text
evaluation_on_validation_set.ipynb
```
and 

```text
test_model.ipynb
```

contain examples of model evaluation, validation, and inference workflows. These notebooks demonstrate how trained models can be loaded, applied to microscopy images, and evaluated using segmentation metrics such as Dice score, IoU, precision, recall, and pixel accuracy. The notebooks have been simplified for public release and therefore do not include all result-generation scripts and thesis-specific analysis.

Most evaluation functionality is implemented in:

```text
src/evaluation.py
```

which contains helper functions for:

* Dice score
* IoU
* Precision
* Recall
* Pixel accuracy
* Statistical analysis of segmentation performance
* Aggregation of evaluation metrics

### Droplet Size Distribution Analysis

The module:

```text
src/mask_analysis.py
```

contains utilities for extracting droplet statistics from segmentation masks, including:

* Equivalent circular diameter
* D10, D50 and D90
* Sauter mean diameter (D32)
* Span
* Coefficient of variation
* Phase fraction estimation

### Manual Review Tool

The script:

```bash
python GUI_mask_review.py
```

launches a graphical interface for reviewing segmentation masks and microscopy images.

## Dataset and Models

The original microscopy images, annotations, trained model weights, TensorBoard logs, and intermediate training outputs are not included in this repository due to size restrictions and project-specific data management requirements.

Only the code required to reproduce the methodology is provided.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

The project was developed using Python 3.11 and relies primarily on:

* PyTorch
* segmentation-models-pytorch
* Albumentations
* NumPy
* SciPy
* scikit-image
* OpenCV
* Matplotlib
* Pandas

## Thesis Context

This work was conducted as part of a Master's thesis in Industrial Chemistry and Biotechnology at NTNU in collaboration with SINTEF Industry Petroleum Research.

The study investigates whether deep learning-based semantic segmentation can be used to automatically characterize the microstructure of oil-based drilling fluids and estimate droplet size distributions from microscopy images.

