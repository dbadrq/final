# testing.py
# -*- coding: utf-8 -*-
import re
import tensorflow as tf
# ... (other imports) ...
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings

# --- Configuration Parameters ---
# ... (parameters remain the same) ...
BASE_FEATURE_DIM = 64
SHARED_BASE_OUTPUT_DIM = 128
HEAD_HIDDEN_DIM = 64
SEED = 42
BATCH_SIZE = 32

# --- File and Directory Paths ---
TESTING_DIR = 'datasets_testing'
# !!! IMPORTANT: Set these based on the main.py run !!!
DP_ENABLED_DURING_TRAINING = True
TRAINED_DROPOUT_RATE = 0.2 # Ensure this matches the main.py run you are testing
# --- End Configuration ---

dp_suffix = "_dp" if DP_ENABLED_DURING_TRAINING else ""
dropout_suffix = f"_dropout{TRAINED_DROPOUT_RATE}" if TRAINED_DROPOUT_RATE > 0 else ""
SCALER_PATH = 'task_scalers_by_basename.pkl'
LABEL_ENCODER_PATH = f'final_label_encoders{dp_suffix}{dropout_suffix}.pkl'
HEAD_MODELS_DIR = f'saved_heads{dp_suffix}{dropout_suffix}'
BASE_MODEL_WEIGHTS_PATH = f'federated_final_base_model_weights{dp_suffix}{dropout_suffix}.h5'

# --- Set random seeds and log level ---
# ... (unchanged) ...
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
tf.get_logger().setLevel('ERROR'); warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning); os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Model Definitions ---
# ... (unchanged) ...
def create_shared_base_model(input_dim=BASE_FEATURE_DIM, output_dim=SHARED_BASE_OUTPUT_DIM, dropout_rate=0.0):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,), name='base_input'), tf.keras.layers.Dense(output_dim, activation='relu', name='base_dense_1'), tf.keras.layers.Dropout(dropout_rate, name='base_dropout')], name='shared_base'); return model
def create_task_head_model(num_classes, input_dim=SHARED_BASE_OUTPUT_DIM, hidden_dim=HEAD_HIDDEN_DIM, dropout_rate=0.0):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,), name='head_input'), tf.keras.layers.Dense(hidden_dim, activation='relu', name='head_dense_1'), tf.keras.layers.Dropout(dropout_rate, name='head_dropout'), tf.keras.layers.Dense(num_classes, name='head_output_logits')], name=f'task_head_{num_classes}cls'); return model

# --- Data Loading and Preprocessing ---
# ... (unchanged) ...
def pad_or_truncate_features_np(features, target_dim=BASE_FEATURE_DIM):
    current_dim = features.shape[1]
    if current_dim == 0: return np.zeros((features.shape[0], target_dim), dtype=features.dtype)
    if current_dim < target_dim:
        pad_width = target_dim - current_dim; constant_val = 0 if np.issubdtype(features.dtype, np.number) else features.dtype.type(0)
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant', constant_values=constant_val)
    elif current_dim > target_dim: features = features[:, :target_dim]
    return features
def load_and_preprocess_test_data(file_path, scaler, label_encoder):
    basename = os.path.basename(file_path)
    if not scaler: return None, None, 0
    if not label_encoder: return None, None, 0
    try:
        data = pd.read_csv(file_path, low_memory=False)
        if data.shape[1] <= 1: return None, None, 0
        features = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        raw_labels = data.iloc[:, -1].values
        if features.shape[1] == 0: return None, None, 0
        features_padded = pad_or_truncate_features_np(features, BASE_FEATURE_DIM)
        if np.any(np.isnan(features_padded)) or np.any(np.isinf(features_padded)): features_padded = np.nan_to_num(features_padded)
        if hasattr(scaler, 'scale_') and np.any(scaler.scale_ == 0): print(f"    Warn: Scaler {basename} zero variance.")
        features_scaled = scaler.transform(features_padded)
        try: labels = label_encoder.transform(raw_labels.astype(str)).astype(np.int32)
        except ValueError:
            unseen = set(raw_labels.astype(str)) - set(label_encoder.classes_); print(f"  Warn: Unseen labels in test {basename}: {unseen}. Skipping samples.")
            valid = np.isin(raw_labels.astype(str), list(label_encoder.classes_))
            if not np.any(valid): return None, None, 0
            features_scaled = features_scaled[valid]; labels = label_encoder.transform(raw_labels[valid].astype(str)).astype(np.int32)
        n_samples = len(labels)
        if n_samples == 0: return None, None, 0
        n_classes = len(label_encoder.classes_)
        if np.any(labels >= n_classes) or np.any(labels < 0): labels = np.clip(labels, 0, n_classes - 1)
        return features_scaled, labels, n_samples
    except FileNotFoundError: return None, None, 0
    except Exception as e: print(f"  Err processing test {basename}: {e}"); return None, None, 0

# --- Main Evaluation Logic ---
def run_evaluation():
    print("--- Independent Evaluation Script ---")
    print(f"Loading artifacts for DP={DP_ENABLED_DURING_TRAINING}, Dropout={TRAINED_DROPOUT_RATE}")
    # --- 1. Load Components ---
    # ... (Loading logic unchanged) ...
    print("\n--- Loading Components ---")
    if not os.path.exists(SCALER_PATH): print(f"FATAL: Scaler file missing: {SCALER_PATH}"); return
    try: task_scalers = pickle.load(open(SCALER_PATH, 'rb')); print(f"Loaded {len(task_scalers)} scalers.")
    except Exception as e: print(f"FATAL loading scalers: {e}"); return
    if not os.path.exists(LABEL_ENCODER_PATH): print(f"FATAL: LE file missing: {LABEL_ENCODER_PATH}"); return
    try: client_label_encoders = pickle.load(open(LABEL_ENCODER_PATH, 'rb')); print(f"Loaded {len(client_label_encoders)} LE mappings.")
    except Exception as e: print(f"FATAL loading LEs: {e}"); return
    if not os.path.exists(BASE_MODEL_WEIGHTS_PATH): print(f"FATAL: Base weights missing: {BASE_MODEL_WEIGHTS_PATH}"); return
    try:
        base_model = create_shared_base_model(dropout_rate=TRAINED_DROPOUT_RATE); base_model.build((None, BASE_FEATURE_DIM))
        base_model.load_weights(BASE_MODEL_WEIGHTS_PATH); base_model.trainable = False; print("Loaded base model weights.")
    except Exception as e: print(f"FATAL loading base weights: {e}"); return
    if not os.path.isdir(HEAD_MODELS_DIR): print(f"FATAL: Head models dir missing: {HEAD_MODELS_DIR}"); return
    print("Head models directory found.")

    # --- 2. Define Test Data Paths (Updated) ---
    print("\nDefining test data paths...")
    client_data_paths_test = {
        'hospital1': [os.path.join(TESTING_DIR, f) for f in [
            'HepatitisCdata.csv',
            'aids_clinical_trials_data.csv',
            'breast-cancer-dataset.csv',
            'data_cardiovascular_risk.csv' # Replaced diabetes.csv
        ]],
        'hospital2': [os.path.join(TESTING_DIR, f) for f in [
            'Liver_disease_data.csv',
            'cardio_data_processed.csv',
            'Chronic_Kidney_Dsease_data.csv',
            'Hypertension-risk-model-main.csv'
        ]],
        'hospital3': [os.path.join(TESTING_DIR, f) for f in [
            'kidney_disease.csv',
            'Heart_Disease_Prediction.csv',
            'asthma_disease_data.csv',
            'survey lung cancer.csv'
        ]],
        'hospital4': [os.path.join(TESTING_DIR, f) for f in [
            'heart2.csv',
            'obesity_data.csv',
            'healthcare-dataset-stroke-data.csv',
            'cancer patient data sets.csv' # Reverted back
        ]]
    }
    all_client_ids = list(client_data_paths_test.keys())
    print("Test data paths defined.")

    # --- 3. Evaluation Loop ---
    # ... (Evaluation loop logic unchanged) ...
    print("\n--- Starting Evaluation Loop ---")
    final_results = defaultdict(dict)
    for cid in all_client_ids:
        print(f"\nEvaluating Client: {cid}")
        c_paths = client_data_paths_test.get(cid, [])
        if not c_paths: continue
        for path in c_paths:
            bn = os.path.basename(path); print(f"  Eval Task: {bn}"); metrics = {'status': 'Init'}
            if not os.path.exists(path): metrics['status'] = 'Fail (Test File Missing)'; final_results[cid][bn] = metrics; continue
            scaler = task_scalers.get(bn)
            if not scaler: metrics['status'] = 'Fail (No Scaler)'; final_results[cid][bn] = metrics; continue
            le = client_label_encoders.get((cid, bn))
            if not le: le = next((l for (c, b), l in client_label_encoders.items() if b == bn), None)
            if not le: metrics['status'] = 'Fail (No LE)'; final_results[cid][bn] = metrics; continue
            s_bn = re.sub(r'[^A-Za-z0-9_.\/-]', '_', bn); h_fname = f'head_{cid}_{s_bn}.h5'; h_path = os.path.join(HEAD_MODELS_DIR, h_fname)
            if not os.path.exists(h_path): metrics['status'] = 'Fail (No Head File)'; final_results[cid][bn] = metrics; continue
            try: task_head = tf.keras.models.load_model(h_path); task_head.trainable = False
            except Exception as e: metrics['status'] = f'Fail (Head Load Err: {e})'; final_results[cid][bn] = metrics; continue
            features, labels, n_samples = load_and_preprocess_test_data(path, scaler, le) # Use correct var name 'le'
            if features is None or labels is None or n_samples == 0: metrics['status'] = 'Fail (Data Load)'; final_results[cid][bn] = metrics; continue
            n_classes = len(le.classes_); class_names = le.classes_
            if n_classes <= 1: metrics['status'] = 'Skip (<=1 Class)'; final_results[cid][bn] = metrics; continue
            try:
                if task_head.output_shape[-1] != n_classes: print(f"    Warn: Head output mismatch {bn}.")
                inp = tf.keras.layers.Input(shape=(BASE_FEATURE_DIM)); base_out = base_model(inp, training=False); task_out = task_head(base_out, training=False)
                name = f"eval_{cid}_{s_bn}"; model = tf.keras.Model(inputs=inp, outputs=task_out, name=name)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True); acc_m = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
                model.compile(optimizer='adam', loss=loss, metrics=[acc_m])
            except Exception as e: metrics['status'] = f'Fail (Compile Err: {e})'; final_results[cid][bn] = metrics; continue
            try:
                res = model.evaluate(features, labels, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
                loss_v, acc_v = res.get('loss', np.nan), res.get('accuracy', np.nan)
                metrics['accuracy'] = f"{acc_v:.4f}" if not np.isnan(acc_v) else 'N/A'; metrics['loss'] = f"{loss_v:.4f}" if not np.isnan(loss_v) else 'N/A'
                logits = model.predict(features, batch_size=BATCH_SIZE, verbose=0); probs = tf.nn.softmax(logits).numpy(); preds = np.argmax(probs, axis=1)
                unique_lbls, n_unique = np.unique(labels), len(np.unique(labels)); report_lbls = np.arange(n_classes)
                if n_unique >= 2:
                    try:
                        if probs.shape[1] == n_classes: auc = roc_auc_score(labels, probs[:, 1] if n_classes == 2 else probs, multi_class='ovr', average='macro', labels=report_lbls); metrics['auc'] = f"{auc:.4f}"
                        else: metrics['auc'] = 'Err (Shape)'
                    except ValueError as e_auc: metrics['auc'] = 'Skip (1 class)' if "Only one class present" in str(e_auc) else f'Err ({e_auc})'
                    except Exception as e_auc_o: metrics['auc'] = f'Err ({e_auc_o})'
                else: metrics['auc'] = 'Skip (<2 classes)'
                kwargs = {'labels': report_lbls, 'zero_division': 0}
                metrics['f1_macro'] = f"{f1_score(labels, preds, average='macro', **kwargs):.4f}"
                try: cm = confusion_matrix(labels, preds, labels=report_lbls); metrics['confusion_matrix_labels'] = class_names.tolist(); metrics['confusion_matrix'] = cm.tolist()
                except Exception as e_cm: metrics['confusion_matrix'] = f'Err ({e_cm})'
                metrics['status'] = 'Success'
            except Exception as e_eval: metrics['status'] = f'Fail (Eval Runtime Err: {e_eval})'; metrics.update({k: 'N/A' for k in metrics if k != 'status'})
            final_results[cid][bn] = metrics

    # --- 4. Print Final Summary ---
    # ... (Summary printing unchanged) ...
    print("\n--- Final Evaluation Summary ---")
    for cid, tasks in sorted(final_results.items()):
        print(f"Client: {cid}")
        if not tasks: print("  No results.")
        else:
            for bn, metrics in sorted(tasks.items()):
                print(f"  Task: {bn}")
                print(f"    Status:   {metrics.get('status', 'N/A')}")
                if metrics.get('status') == 'Success':
                    print(f"    Accuracy: {metrics.get('accuracy', 'N/A')}")
                    print(f"    Loss:     {metrics.get('loss', 'N/A')}")
                    print(f"    AUC:      {metrics.get('auc', 'N/A')}")
                    print(f"    F1 Macro: {metrics.get('f1_macro', 'N/A')}")
                    cm = metrics.get('confusion_matrix'); cm_lbls = metrics.get('confusion_matrix_labels')
                    if cm and isinstance(cm, list) and cm_lbls:
                         try: print(f"    Confusion Matrix:\n{pd.DataFrame(cm, index=[f'T:{c}' for c in cm_lbls], columns=[f'P:{c}' for c in cm_lbls]).to_string()}")
                         except: print(f"    Confusion Matrix (raw):\n{np.array(cm)}")
        print("-" * 30)

    # --- Optional: Save Results ---
    # ... (Saving logic unchanged) ...
    try: results_filename = f"evaluation_results{dp_suffix}{dropout_suffix}.pkl"; pickle.dump(final_results, open(results_filename, 'wb')); print(f"\nEval results saved: {results_filename}")
    except Exception as e: print(f"\nErr saving eval results: {e}")


if __name__ == '__main__':
    run_evaluation()
    print("\n--- Evaluation Script Finished ---")

