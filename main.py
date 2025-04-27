# main.py
# -*- coding: utf-8 -*-
import re
import tensorflow as tf
# ... (other imports remain the same) ...
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle
import random
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
try:
    import tensorflow_privacy as tfp
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import VectorizedDPKerasAdamOptimizer
    DP_AVAILABLE = True
    print("Successfully imported TensorFlow Privacy library.")
except ImportError:
    DP_AVAILABLE = False
    print("TensorFlow Privacy library not found. DP features will be disabled.")

# --- Configuration Parameters ---
# ... (parameters remain the same) ...
BASE_FEATURE_DIM = 64
SHARED_BASE_OUTPUT_DIM = 128
HEAD_HIDDEN_DIM = 64
SCALER_FILENAME = 'task_scalers_by_basename.pkl'
TRAINING_DIR = 'datasets_training'
TESTING_DIR = 'datasets_testing'
SEED = 42
NUM_ROUNDS = 40
CLIENTS_PER_ROUND = 2
LOCAL_EPOCHS = 7
LEARNING_RATE = 5e-4
CANCER_TASK_NAME = 'cancer patient data sets.csv'
CANCER_TASK_HEAD_HIDDEN_DIM = 32
BATCH_SIZE = 32
ENABLE_DP = True
L2_NORM_CLIP = 1.3
NOISE_MULTIPLIER = 0.1
DROPOUT_RATE = 0.2

# --- Set random seeds and log level ---
# ... (unchanged) ...
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
tf.get_logger().setLevel('ERROR'); warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Model Definitions ---
def create_shared_base_model(input_dim=BASE_FEATURE_DIM, output_dim=SHARED_BASE_OUTPUT_DIM, dropout_rate=0.0):
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(input_dim,), name='base_input'),
                                 tf.keras.layers.Dense(output_dim, activation='relu', name='base_dense_1'),
                                 tf.keras.layers.Dropout(dropout_rate, name='base_dropout')], name='shared_base');
    return model

# ** Modified Function **
def create_task_head_model(num_classes, input_dim=SHARED_BASE_OUTPUT_DIM, hidden_dim=HEAD_HIDDEN_DIM, dropout_rate=0.0):
    """
    Creates a task-specific head model (MLP) with a configurable hidden dimension.
    """
    print(f"    Creating head with hidden_dim={hidden_dim}, num_classes={num_classes}") # Added print for debugging
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='head_input'),
        # Use the passed hidden_dim parameter here
        tf.keras.layers.Dense(hidden_dim, activation='relu', name='head_dense_1'),
        tf.keras.layers.Dropout(dropout_rate, name='head_dropout'),
        tf.keras.layers.Dense(num_classes, name='head_output_logits')
        # Removed kernel_regularizer
    ], name=f'task_head_{num_classes}cls_h{hidden_dim}'); # Updated name for clarity
    return model

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
def compute_task_scalers(hospital_paths_train):
    task_scalers = {}; files_to_process = {}
    for paths in hospital_paths_train.values():
        for path in paths:
            bn = os.path.basename(path); fp = os.path.abspath(path)
            if os.path.exists(fp):
                if bn not in files_to_process: files_to_process[bn] = fp
            else: print(f"Warn: Train file not found for scaler: {fp}")
    print(f"Found {len(files_to_process)} unique datasets for scaler computation.")
    for bn, fp in files_to_process.items():
        print(f"  Computing scaler: {bn}")
        try:
            data = pd.read_csv(fp, low_memory=False)
            if data.shape[1] <= 1: print(f"    Skip {bn}: <=1 col."); continue
            feats = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
            if feats.shape[1] == 0: print(f"    Skip {bn}: No features."); continue
            feats_pad = pad_or_truncate_features_np(feats, BASE_FEATURE_DIM)
            if feats_pad.shape[0] == 0: print(f"    Skip {bn}: 0 samples."); continue
            if np.all(np.var(feats_pad, axis=0) < 1e-6): print(f"    Skip {bn}: Zero variance."); continue
            scaler = StandardScaler().fit(feats_pad); task_scalers[bn] = scaler; print(f"    Scaler computed: {bn}.")
        except Exception as e: print(f"    Err computing scaler {bn}: {e}")
    if not task_scalers: raise ValueError("Failed to compute any valid scalers.")
    print(f"\nComputed {len(task_scalers)} scalers.")
    return task_scalers
def load_and_preprocess_task_data(file_path, scaler, label_encoder_map=None):
    bn = os.path.basename(file_path)
    if not scaler: return None, None, 0, None
    try:
        data = pd.read_csv(file_path, low_memory=False)
        if data.shape[1] <= 1: return None, None, 0, None
        feats = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)
        lbls_raw = data.iloc[:, -1].values
        if feats.shape[1] == 0: return None, None, 0, None
        feats_pad = pad_or_truncate_features_np(feats, BASE_FEATURE_DIM)
        if np.any(np.isnan(feats_pad)) or np.any(np.isinf(feats_pad)): feats_pad = np.nan_to_num(feats_pad)
        if hasattr(scaler, 'scale_') and np.any(scaler.scale_ == 0): print(f"Warn: Scaler {bn} zero variance.")
        feats_scaled = scaler.transform(feats_pad)
        le = None; le_key = bn
        if label_encoder_map and le_key in label_encoder_map:
            le = label_encoder_map[le_key]
            try: lbls = le.transform(lbls_raw.astype(str)).astype(np.int32)
            except ValueError: unseen = set(lbls_raw.astype(str)) - set(le.classes_); print(f"Warn: Unseen {unseen} in {bn}. Re-fitting LE."); le = LabelEncoder(); lbls = le.fit_transform(lbls_raw.astype(str)).astype(np.int32)
        else: le = LabelEncoder(); lbls = le.fit_transform(lbls_raw.astype(str)).astype(np.int32)
        n_classes = len(le.classes_); n_samples = len(lbls)
        if np.any(lbls < 0): valid = lbls >= 0; feats_scaled = feats_scaled[valid]; lbls = lbls[valid]; n_samples = len(lbls);
        if n_samples == 0: return None, None, 0, None
        if n_classes <= 1: return None, None, 0, None
        if n_samples > 0 and np.any(lbls >= n_classes): lbls = np.clip(lbls, 0, n_classes - 1)
        return feats_scaled, lbls, n_samples, le
    except FileNotFoundError: print(f"Err loading: File not found {file_path}"); return None, None, 0, None
    except Exception as e: print(f"Err processing {bn}: {e}"); return None, None, 0, None

# --- Client Simulation ---
# ... (unchanged) ...
# --- Client Simulation ---
def client_update(client_id, client_task_paths, global_base_weights, task_scalers, client_heads, local_epochs, learning_rate, enable_dp, dp_params, dropout_rate):
    """
    Executes local training for a single client (applying Strategy 3: specific head architecture).
    Uses model.compile and model.fit.
    """
    use_dp = enable_dp and DP_AVAILABLE
    local_batch_size = BATCH_SIZE

    # 1. Create local instance of base model and load global weights (unchanged)
    base_model_instance = create_shared_base_model(dropout_rate=dropout_rate)
    try:
        base_model_instance.build((None, BASE_FEATURE_DIM))
        base_model_instance.set_weights(global_base_weights)
    except ValueError as e:
        print(f"  Err setting weights {client_id}: {e}")
        return None, 0, {}
    base_model_instance.trainable = True

    total_samples = 0
    success = False
    local_les = {}

    # 2. Local training epochs (unchanged)
    for epoch in range(local_epochs):
        print(f"    Client {client_id} - Epoch {epoch + 1}/{local_epochs}")
        random.shuffle(client_task_paths)
        epoch_s = 0

        # 3. Iterate through client's tasks
        for path in client_task_paths:
            bn = os.path.basename(path)
            scaler = task_scalers.get(bn)

            if not scaler:
                # ... (scaler check unchanged) ...
                continue

            # Load and preprocess data (unchanged)
            current_client_le_map = {b: l for (c, b), l in local_les.items() if c == client_id}
            feats, lbls, n_samp, le_fit = load_and_preprocess_task_data(path, scaler, current_client_le_map)

            if feats is None or n_samp == 0 or le_fit is None:
                # ... (data loading check unchanged) ...
                continue

            local_les[(client_id, bn)] = le_fit
            n_classes = len(le_fit.classes_)

            if n_classes <= 1:
                # ... (class check unchanged) ...
                continue

            # <<< --- Strategy 3 Modification Start --- >>>
            # Determine the correct hidden dimension for this task
            current_hidden_dim = HEAD_HIDDEN_DIM # Default
            if bn == CANCER_TASK_NAME:
                current_hidden_dim = CANCER_TASK_HEAD_HIDDEN_DIM
                print(f"      Task {bn} uses specific hidden_dim: {current_hidden_dim}")

            # Check if head exists and if its structure is correct
            rebuild_head = False
            if bn not in client_heads:
                print(f"      New head needed for {bn}.")
                rebuild_head = True
            else:
                task_head = client_heads[bn]
                # Check output classes
                if task_head.output_shape[-1] != n_classes:
                    print(f"      Warn: Head mismatch {bn} (Output classes: {task_head.output_shape[-1]} vs {n_classes}). Rebuilding.")
                    rebuild_head = True
                # Check hidden dimension
                try:
                    # Ensure the layer exists and check its units
                    if task_head.get_layer('head_dense_1').units != current_hidden_dim:
                        print(f"      Warn: Head mismatch {bn} (Hidden dim: {task_head.get_layer('head_dense_1').units} vs {current_hidden_dim}). Rebuilding.")
                        rebuild_head = True
                except ValueError: # Layer 'head_dense_1' might not exist if structure changed drastically before
                     print(f"      Warn: Head structure mismatch {bn} (Cannot find head_dense_1). Rebuilding.")
                     rebuild_head = True

            # Create or rebuild head if necessary
            if rebuild_head:
                client_heads[bn] = create_task_head_model(
                    num_classes=n_classes,
                    hidden_dim=current_hidden_dim, # Pass the determined dimension
                    dropout_rate=dropout_rate
                )

            task_head = client_heads[bn] # Get the final head for this task
            # <<< --- Strategy 3 Modification End --- >>>


            # 5. Create TensorFlow Dataset (unchanged)
            try:
                ds = tf.data.Dataset.from_tensor_slices((feats, lbls)).shuffle(max(n_samp, 1), seed=SEED + epoch)
                if use_dp:
                    ds = ds.batch(local_batch_size, drop_remainder=True)
                    num_batches = tf.data.experimental.cardinality(ds).numpy()
                    task_s = 0 if (num_batches == tf.data.experimental.UNKNOWN_CARDINALITY or num_batches <= 0) else num_batches * local_batch_size
                    if task_s == 0: print(f"      Skip {bn}: No samples for DP batch."); continue
                else:
                    ds = ds.batch(local_batch_size)
                    task_s = n_samp # Use original count if not dropping remainder
                ds = ds.prefetch(tf.data.AUTOTUNE)
                print(f"      Train {bn} ({n_classes} cls) w/ {task_s}/{n_samp} samples.")
            except Exception as e:
                print(f"      Err creating Dataset {bn}: {e}")
                continue

            # 6. Build, Compile, and Fit (Reverted to model.fit)
            try:
                # Build the combined model
                inp = tf.keras.layers.Input(shape=(BASE_FEATURE_DIM))
                base_out = base_model_instance(inp, training=True)
                task_out = task_head(base_out, training=True) # Use the potentially modified task_head
                name = f"combined_{client_id}_{re.sub(r'[^A-Za-z0-9_.-]', '_', bn)}"
                model = tf.keras.Model(inputs=inp, outputs=task_out, name=name)

                # Setup optimizer and loss based on DP enabled/disabled
                if use_dp:
                    l2, noise, micro = dp_params['l2_norm_clip'], dp_params['noise_multiplier'], dp_params.get('num_microbatches', local_batch_size)
                    if micro is None or micro <= 0: micro = local_batch_size
                    # Use Vectorized optimizer with model.fit
                    opt = VectorizedDPKerasAdamOptimizer(
                        l2_norm_clip=l2,
                        noise_multiplier=noise,
                        num_microbatches=micro,
                        learning_rate=learning_rate
                    )
                    # Loss needs reduction=NONE for Vectorized DP optimizer
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
                else:
                    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    # Standard loss for non-DP
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                # Compile the model
                model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

                # Fit the model
                model.fit(ds, epochs=1, verbose=0) # Single inner epoch as before

                # Aggregate samples only on the first outer epoch
                if epoch == 0:
                    epoch_s += task_s
                success = True
            except Exception as e_fit:
                print(f"      Err during fit {bn}: {e_fit}")
                # Optional: Add traceback print here if errors persist
                # import traceback
                # traceback.print_exc()

        # Aggregate total samples after first epoch
        if epoch == 0:
            total_samples = epoch_s

    # 7. Return results (unchanged)
    if not success:
        print(f"  Client {client_id}: Failed all tasks.")
        return None, 0, {}
    print(f"  Client {client_id} finished. Samples: {total_samples}.")
    return base_model_instance.get_weights(), total_samples, local_les

# ... (Rest of the code: server_aggregate, evaluate_task, main execution block remains the same) ...


# --- Server Aggregation ---
# ... (unchanged) ...
def server_aggregate(client_weight_updates):
    if not client_weight_updates: return None
    valid = [(w, s, le) for w, s, le in client_weight_updates if w is not None and s >= 0]
    if not valid: return None
    total_s = sum(s for _, s, _ in valid)
    if total_s == 0: return valid[0][0] if valid else None
    struct = valid[0][0]; agg_w = [np.zeros_like(w) for w in struct]
    for w, n, _ in valid:
        if n > 0: factor = n / total_s; [agg_w[i].__iadd__(w[i] * factor) for i in range(len(agg_w))]
    return agg_w

# --- Evaluation Function ---
# ... (unchanged) ...
def evaluate_task(global_base_weights, client_id, task_basename, test_file_path, task_scalers, client_heads, client_label_encoders, dropout_rate):
    print(f"  Eval: {task_basename} (Client: {client_id})")
    metrics = {'status': 'Started', 'accuracy': 'N/A', 'loss': 'N/A', 'auc': 'N/A', 'f1_macro': 'N/A'}
    scaler = task_scalers.get(task_basename)
    if not scaler: metrics['status'] = 'Fail (No Scaler)'; return metrics
    le = client_label_encoders.get((client_id, task_basename))
    if not le: le = next((l for (c, b), l in client_label_encoders.items() if b == task_basename), None)
    if not le: metrics['status'] = 'Fail (No LE)'; return metrics
    feats, lbls, n_s, _ = load_and_preprocess_task_data(test_file_path, scaler, {task_basename: le})
    if feats is None or n_s == 0: metrics['status'] = 'Fail (Data Load)'; return metrics
    n_classes = len(le.classes_); class_names = le.classes_
    if n_classes <= 1: metrics['status'] = 'Skip (<=1 Class)'; return metrics
    base_model = create_shared_base_model(dropout_rate=dropout_rate)
    try: base_model.build((None, BASE_FEATURE_DIM)); base_model.set_weights(global_base_weights)
    except ValueError: metrics['status'] = 'Fail (Weight Load)'; return metrics
    base_model.trainable = False
    if client_id not in client_heads or task_basename not in client_heads[client_id]: metrics['status'] = 'Fail (No Head)'; return metrics
    task_head = client_heads[client_id][task_basename]; task_head.trainable = False
    if task_head.output_shape[-1] != n_classes: print(f"    Warn: Eval head mismatch {task_basename}.")
    try:
        inp = tf.keras.layers.Input(shape=(BASE_FEATURE_DIM)); base_out = base_model(inp, training=False); task_out = task_head(base_out, training=False)
        name = f"eval_{client_id}_{re.sub(r'[^A-Za-z0-9_.-]', '_', task_basename)}"; model = tf.keras.Model(inputs=inp, outputs=task_out, name=name)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True); acc_m = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        model.compile(optimizer='adam', loss=loss, metrics=[acc_m])
    except Exception as e: metrics['status'] = f'Fail (Compile Err: {e})'; return metrics
    try:
        res = model.evaluate(feats, lbls, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
        loss_v, acc_v = res.get('loss', np.nan), res.get('accuracy', np.nan)
        metrics['accuracy'] = f"{acc_v:.4f}" if not np.isnan(acc_v) else 'N/A'; metrics['loss'] = f"{loss_v:.4f}" if not np.isnan(loss_v) else 'N/A'
        logits = model.predict(feats, batch_size=BATCH_SIZE, verbose=0); probs = tf.nn.softmax(logits).numpy(); preds = np.argmax(probs, axis=1)
        unique_lbls, n_unique = np.unique(lbls), len(np.unique(lbls)); report_lbls = np.arange(n_classes)
        if n_unique >= 2:
            try:
                if probs.shape[1] == n_classes: auc = roc_auc_score(lbls, probs[:, 1] if n_classes == 2 else probs, multi_class='ovr', average='macro', labels=report_lbls); metrics['auc'] = f"{auc:.4f}"
                else: metrics['auc'] = 'Err (Shape)'
            except ValueError as e_auc: metrics['auc'] = 'Skip (1 class)' if "Only one class present" in str(e_auc) else f'Err ({e_auc})'
            except Exception as e_auc_o: metrics['auc'] = f'Err ({e_auc_o})'
        else: metrics['auc'] = 'Skip (<2 classes)'
        kwargs = {'labels': report_lbls, 'zero_division': 0}
        metrics['f1_macro'] = f"{f1_score(lbls, preds, average='macro', **kwargs):.4f}"
        metrics['status'] = 'Success'
    except Exception as e_eval: metrics['status'] = f'Fail (Eval Err: {e_eval})'; metrics.update({k: 'N/A' for k in metrics if k != 'status'})
    print(f"    Eval status {task_basename}: {metrics['status']}.")
    return metrics

# --- Main Program Entry Point ---
if __name__ == "__main__":
    print("--- Federated Multi-Task Learning Simulation ---")
    # ... (DP Check - unchanged) ...
    if not DP_AVAILABLE: ENABLE_DP = False
    use_dp = DP_AVAILABLE and ENABLE_DP
    if use_dp: dp_params = {'l2_norm_clip': L2_NORM_CLIP, 'noise_multiplier': NOISE_MULTIPLIER, 'num_microbatches': BATCH_SIZE}; print(f"DP ENABLED: {dp_params}")
    else: dp_params = {}; print("DP DISABLED.")

    # --- Define Client Data Paths (Updated) ---
    print("\nDefining client data paths...")
    client_data_paths_train = {
        'hospital1': [os.path.join(TRAINING_DIR, f) for f in [
            'HepatitisCdata.csv',
            'aids_clinical_trials_data.csv',
            'breast-cancer-dataset.csv',
            'data_cardiovascular_risk.csv' # Replaced diabetes.csv
        ]],
        'hospital2': [os.path.join(TRAINING_DIR, f) for f in [
            'Liver_disease_data.csv',
            'cardio_data_processed.csv',
            'Chronic_Kidney_Dsease_data.csv',
            'Hypertension-risk-model-main.csv'
        ]],
        'hospital3': [os.path.join(TRAINING_DIR, f) for f in [
            'kidney_disease.csv',
            'Heart_Disease_Prediction.csv',
            'asthma_disease_data.csv',
            'survey lung cancer.csv'
        ]],
        'hospital4': [os.path.join(TRAINING_DIR, f) for f in [
            'heart2.csv',
            'obesity_data.csv',
            'healthcare-dataset-stroke-data.csv',
            'cancer patient data sets.csv' # Reverted back
        ]]
    }
    client_data_paths_test = { # Mirror structure
        client: [os.path.join(TESTING_DIR, os.path.basename(p)) for p in paths]
        for client, paths in client_data_paths_train.items()
    }
    all_client_ids = list(client_data_paths_train.keys())
    print("Client data paths defined.")

    # --- Verify Training File Existence ---
    print("Verifying training data file existence...")
    # ... (Verification logic unchanged) ...
    for client, paths in client_data_paths_train.items():
        for path in paths:
            if not os.path.exists(path): print(f"  WARN: File not found: {path}")

    # --- Load or Compute Scalers ---
    # ... (Loading/Computing Logic - unchanged) ...
    print(f"\nLoading or computing scalers ({SCALER_FILENAME})...")
    task_scalers = {}
    if os.path.exists(SCALER_FILENAME):
        try: task_scalers = pickle.load(open(SCALER_FILENAME, 'rb')); print(f"Loaded {len(task_scalers)} scalers.")
        except Exception as e: print(f"Err loading scalers: {e}. Recomputing.")
    if not task_scalers:
        try: task_scalers = compute_task_scalers(client_data_paths_train); pickle.dump(task_scalers, open(SCALER_FILENAME, 'wb')); print(f"Computed/saved {len(task_scalers)} scalers.")
        except Exception as e: print(f"FATAL: Err computing/saving scalers: {e}"); exit()

    # --- Initialize Server Model and Client States ---
    # ... (Initialization Logic - unchanged) ...
    print("\nInitializing server model and client states...")
    global_base_model = create_shared_base_model(dropout_rate=DROPOUT_RATE)
    global_base_weights = global_base_model.get_weights()
    client_heads = defaultdict(dict)
    client_label_encoders = {}
    print("Initialization complete.")

    # --- Federated Learning Loop ---
    # ... (FL Loop Logic - unchanged) ...
    print("\n--- Starting Federated Learning Training Loop ---")
    for round_num in range(NUM_ROUNDS):
        print(f"\n--- Round {round_num + 1}/{NUM_ROUNDS} ---")
        selected_clients = random.sample(all_client_ids, min(CLIENTS_PER_ROUND, len(all_client_ids)))
        print(f"Selected clients: {selected_clients}")
        round_updates = []; round_les = {}
        for cid in selected_clients:
            print(f"  Training client: {cid}")
            c_heads = client_heads.get(cid, {}); c_paths = client_data_paths_train[cid]
            c_les = {k: v for k, v in client_label_encoders.items() if k[0] == cid}
            weights, samples, fitted_les = client_update(cid, c_paths, global_base_weights, task_scalers, c_heads, LOCAL_EPOCHS, LEARNING_RATE, use_dp, dp_params, DROPOUT_RATE)
            client_heads[cid] = c_heads # Update main dict
            if weights is not None and samples >= 0: round_updates.append((weights, samples, fitted_les)); round_les.update(fitted_les)
            else: print(f"  Client {cid} no valid update.")
        if round_updates:
            print("Aggregating updates..."); agg_w = server_aggregate(round_updates)
            if agg_w: global_base_weights = agg_w; print("Global model updated.")
            else: print("Aggregation failed.")
            client_label_encoders.update(round_les); print(f"Updated LE map ({len(client_label_encoders)} entries).")
        else: print("No valid updates this round.")

        # --- Periodic Evaluation ---
        # ... (Periodic Evaluation Logic - unchanged) ...
        if (round_num + 1) % 10 == 0 or (round_num + 1) == NUM_ROUNDS:
             print(f"\n--- Periodic Eval (Round {round_num + 1}) ---")
             eval_cid, eval_bn = 'hospital1', 'HepatitisCdata.csv'; eval_path = os.path.join(TESTING_DIR, eval_bn)
             if eval_cid in client_heads and eval_bn in client_heads[eval_cid]:
                 if os.path.exists(eval_path):
                    print(f"Evaluating {eval_bn} on {eval_cid}...")
                    metrics = evaluate_task(global_base_weights, eval_cid, eval_bn, eval_path, task_scalers, client_heads, client_label_encoders, DROPOUT_RATE)
                    print(f"  Eval Acc ({eval_bn}): {metrics.get('accuracy', 'N/A')}")
                 else: print(f"Skip eval: Test file {eval_path} missing.")
             else: print(f"Skip eval: Head {eval_cid}/{eval_bn} not trained.")
             print("-" * 20)

    print("\n--- Federated Learning Training Finished ---")

    # --- Final Evaluation ---
    # ... (Final Evaluation Logic - unchanged) ...
    print("\n--- Final Evaluation ---")
    final_results = defaultdict(dict)
    for cid in all_client_ids:
        print(f"\nEvaluating Client: {cid}")
        c_paths = client_data_paths_test.get(cid, [])
        if not c_paths: continue
        if cid not in client_heads: print(f"  Skip client {cid}: No heads trained."); continue
        c_heads_dict = client_heads[cid]
        for path in c_paths:
            bn = os.path.basename(path)
            if not os.path.exists(path): print(f"  Skip {bn}: Test file missing."); final_results[cid][bn] = {'status': 'Fail (Test File Missing)'}; continue
            if bn not in c_heads_dict: print(f"  Skip {bn}: Head not trained."); final_results[cid][bn] = {'status': 'Skip (No Head)'}; continue
            metrics = evaluate_task(global_base_weights, cid, bn, path, task_scalers, client_heads, client_label_encoders, DROPOUT_RATE)
            final_results[cid][bn] = metrics

    # --- Print Final Summary ---
    # ... (Summary Printing Logic - unchanged) ...
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
        print("-" * 30)

    # --- Save Final Artifacts ---
    # ... (Saving Logic - unchanged) ...
    print("\n--- Saving Final Artifacts ---")
    dp_sfx = "_dp" if use_dp else ""; drop_sfx = f"_dropout{DROPOUT_RATE}" if DROPOUT_RATE > 0 else ""
    try:
        f_path = f'federated_final_base_model_weights{dp_sfx}{drop_sfx}.h5'; f_model = create_shared_base_model(dropout_rate=DROPOUT_RATE)
        f_model.build((None, BASE_FEATURE_DIM)); f_model.set_weights(global_base_weights); f_model.save_weights(f_path); print(f"Base weights saved: {f_path}")
    except Exception as e: print(f"Err saving base weights: {e}")
    try:
        le_path = f'final_label_encoders{dp_sfx}{drop_sfx}.pkl'; pickle.dump(client_label_encoders, open(le_path, 'wb')); print(f"LEs saved: {le_path}")
    except Exception as e: print(f"Err saving LEs: {e}")
    try:
        head_dir = f'saved_heads{dp_sfx}{drop_sfx}'; os.makedirs(head_dir, exist_ok=True); saved_count = 0
        for cid, heads in client_heads.items():
            for bn, head in heads.items():
                s_bn = re.sub(r'[^A-Za-z0-9_.\/-]', '_', bn); h_fname = f'head_{cid}_{s_bn}.h5'; h_path = os.path.join(head_dir, h_fname)
                head.save(h_path); saved_count += 1
        print(f"Saved {saved_count} heads to: {head_dir}")
    except Exception as e: print(f"Err saving heads: {e}")

    print("\n--- Simulation Finished ---")

