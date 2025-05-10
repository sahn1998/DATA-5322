# Bird Species Classification from Audio

##  Project Overview

This project focuses on classifying bird species from audio clips using spectrograms and convolutional neural networks (CNNs). The task includes:

- **Binary classification**: Predict between two bird species.
- **Multi-class classification**: Classify audio into one of 12 bird species.
- **test mp3 prediction**: Predict species from raw MP3 clips (some may contain multiple bird calls).
- **Model comparison**: Evaluate different custom CNN architectures and hyperparameters to identify the most effective solution.

---

##  Model Design and Training

### Binary Classification Model  
Two species selected: `houspa` vs `sonspa`.

| Model Variant    | Layers          | Recall (target)  | Accuracy |
|------------------|-----------------|------------------|----------|
| `two_layer_cnn`  | 2 conv + dense  | 0.0              | 0.5      |
| `four_layer_cnn` | 4 conv + dropout|**0.581818**      | 0.766716 |

### Multi-Class Classification Model  
Target: Classify between all **12 bird species**.

| Model Variant        | Layers              | Macro F1 | Accuracy |
|----------------------|---------------------|----------|----------|
| `five_layer_cnn`     | 5 conv + 2 dense    | **0.44** | **0.62** |
| `four_layer_cnn`     | 4 conv + 1 dense    | 0.15     | 0.42     |

---

## MP3 Test Clips

Each of the 3 test clips was segmented into 3-second windows, converted into spectrograms, and passed through the **multi-class model**.


> Plots and segment-by-segment predictions can be found in `output/test_results/`.

---

## Training Strategy

- **Loss Function**: Categorical Crossentropy & Binary Crossentropy
- **Optimizer**: Adam
- **Regularization**: Dropout (0.2–0.3), BatchNorm
- **Validation Method**: Stratified K-Fold (k=5)
- **Metrics of Focus**:
  - Binary Model: **Recall** — important for rare class detection.
  - Multi-Class Model: **Macro F1** — fair evaluation across imbalanced classes.
---