# Deep Learning & Federated Learning for Disease Prediction

## Project Overview

This project aims to provide an efficient solution for **multi-task disease prediction** by combining **Deep Learning (DL)** and **Federated Learning (FL)**. The system processes data from multiple hospitals to build a privacy-preserving collaborative learning model, enabling cross-institutional collaboration **without sharing raw data**. A **Multi-Layer Perceptron (MLP)** serves as the core model architecture, optimized for each disease using **task-specific heads**.

---

## Dependencies & Environment Setup

### Software Requirements:
- Python 3.x
- TensorFlow 2.x
- TensorFlow Federated (TFF)
- Scikit-learn
- Pandas
- Numpy
- Matplotlib (for visualization)
- Jupyter (optional)

## Configuration Notes:
- Linux is recommended (Windows may require VM for compatibility).

- Ensure correct TensorFlow Federated (TFF) versions as specified.

- Detailed configurations are managed in main.py and task_scalers_by_basename.py.

## File Structure
```
├── datasets/                  # Raw medical datasets (preprocessing required)
├── datasets_training/         # Preprocessed training datasets
├── datasets_testing/          # Preprocessed testing datasets
├── saved_heads/               # Trained task-specific heads (no privacy)
├── saved_heads_dp_dropout0.2/ # Task heads with differential privacy (DP) & dropout
├── clean.py                   # Data preprocessing (missing values, normalization)
├── main.py                    # Main training/evaluation script
├── task_scalers_by_basename.pkl            # Task-specific scalers
├── evaluation_results_dp_dropout0.2.pkl    # Evaluation metrics (AUC, F1, etc.)
├── final_label_encoders_dp_dropout0.2.pkl  # Label encoders
└── testing.py                 # Model validation on test sets
```

## How to Run
### Step 1: Prepare Datasets
Place preprocessed CSV files in datasets_training and datasets_testing. Each file corresponds to a disease prediction task.

### Step 2: Train the Model
```bash
python main.py
```
This trains the model, saves task-specific heads, and stores evaluation results.

### Step 3: Evaluate Performance
``` bash
python testing.py
```
Outputs accuracy, AUC, and F1 scores for each task. Results are saved in evaluation_results_dp_dropout0.2.pkl.

## Notes
- Dependency Compatibility: Ensure exact versions of TensorFlow and TFF.

- Imbalanced Data: Test set performance may vary due to class imbalance.

- Skipped Tasks: Some tasks are skipped during evaluation if test data is insufficient or highly imbalanced.
