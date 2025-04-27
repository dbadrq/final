# clean.py
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import warnings

# --- Configuration and Constants ---
INPUT_DIR = Path("datasets")
TRAINING_DIR = Path("datasets_training")
TESTING_DIR = Path("datasets_testing")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
TESTING_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS_TO_DROP_GENERAL = {
    "patientid", "doctorincharge", "index",
    "patient id", #"id", # Keep 'id' potentially, handle via specific drops
    "doctor", "N_Days"
}

# --- Dataset Specific Configurations ---
DATASET_CONFIGS = {
    # --- NEW / UPDATED ---
    'data_cardiovascular_risk.csv':   {'target': 'TenYearCHD', 'drop': []}, # ADDED replacement
    'Heart_Disease_Prediction.csv':   {'target': 'Heart Disease', 'drop': []},
    'cardio_data_processed.csv':    {'target': 'bp_category_encoded', 'drop': ['id']},
    'cancer patient data sets.csv':   {'target': 'Level', 'drop': []}, # ADDED BACK

    # --- KEPT DATASETS ---
    'aids_clinical_trials_data.csv':      {'target': 'cid', 'drop': []},
    'breast-cancer-dataset.csv':          {'target': 'Diagnosis Result', 'drop': ['S/N']},
    'Hypertension-risk-model-main.csv':   {'target': 'Risk', 'drop': []},
    'survey lung cancer.csv':             {'target': 'LUNG_CANCER', 'drop': []},
    'HepatitisCdata.csv':                 {'target': 'Category', 'drop': []},
    'Liver_disease_data.csv':             {'target': 'Diagnosis', 'drop': []},
    'Chronic_Kidney_Dsease_data.csv':     {'target': 'Diagnosis', 'drop': []},
    'kidney_disease.csv':                 {'target': 'classification', 'drop': ['id']},
    'asthma_disease_data.csv':            {'target': 'Diagnosis', 'drop': []},
    'heart2.csv':                         {'target': 'target', 'drop': []},
    'obesity_data.csv':                   {'target': 'ObesityCategory', 'drop': []},
    'healthcare-dataset-stroke-data.csv': {'target': 'stroke', 'drop': ['id']},

    # --- REMOVED DATASETS (No longer listed) ---
    # 'Thyroid_Dataset_Resampled.csv': ...
    # 'KosteckiDillon.csv': ...
    # 'indian_liver_patient.csv': ...
    # 'Thyroid_Diff.csv': ...
    # 'diabetes_data_upload.csv': ...
}


# --- Helper Function (Keep as is) ---
def move_column_to_end(df, column_name):
    """Moves the specified column to be the last column in the DataFrame."""
    if column_name in df.columns:
        col = df.pop(column_name)
        df[column_name] = col
    return df

# --- Main Processing Function (Revised Structure) ---
# ... (The rest of the process_file function and main function remain unchanged) ...
def process_file(file_path):
    print(f"\nProcessing: {file_path.name}")
    if file_path.name not in DATASET_CONFIGS:
        print(f"  Skipping {file_path.name}: File not in DATASET_CONFIGS.")
        return

    try:
        df = pd.read_csv(file_path, na_values=["", "NA", "N/A", "null", "?"], encoding='utf-8')
    except Exception as e:
        print(f"  Error reading {file_path.name}: {e}. Skipping.")
        return

    if df.empty:
        print(f"  Skipping {file_path.name}: File is empty.")
        return

    # --- 0. Get Target and Specific Drops ---
    file_config = DATASET_CONFIGS[file_path.name]
    target_col_name = file_config.get('target')
    specific_cols_to_drop = file_config.get('drop', [])

    if not target_col_name or target_col_name not in df.columns:
        print(f"  Error: Target column '{target_col_name}' not found or defined for {file_path.name}. Columns: {df.columns.tolist()}. Skipping.")
        return
    print(f"  Identified target column: '{target_col_name}'")

    # --- 1. Perform Initial Column Drops ---
    cols_to_drop_now = set(specific_cols_to_drop)
    for col in df.columns:
        if col == target_col_name: continue
        col_lower_stripped = col.lower().strip()
        if col_lower_stripped in COLUMNS_TO_DROP_GENERAL or \
           col.strip() == '' or \
           col.startswith('Unnamed'):
            if col not in cols_to_drop_now:
                 cols_to_drop_now.add(col)

    valid_cols_to_drop = [col for col in cols_to_drop_now if col in df.columns]
    if valid_cols_to_drop:
        df.drop(columns=valid_cols_to_drop, inplace=True, errors="ignore")
        print(f"  Dropped columns: {sorted(list(valid_cols_to_drop))}")

    if target_col_name not in df.columns:
        print(f"  FATAL Error: Target column '{target_col_name}' lost after drops. Skipping.")
        return

    # --- 2. Attempt Numeric Conversion ---
    feature_columns = [col for col in df.columns if col != target_col_name]
    print(f"  Attempting numeric conversion for features: {feature_columns}")
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 3. Separate Features/Target & Move Target Last ---
    df = move_column_to_end(df, target_col_name)
    if df.columns[-1] != target_col_name:
         print(f"  FATAL Error: Failed to move target '{target_col_name}' to end. Skipping.")
         return
    print(f"  Target column '{target_col_name}' confirmed last.")

    if df.shape[1] < 2:
        print(f"  Skipping {file_path.name}: < 2 columns remain.")
        return

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(f"  Features shape before split: {X.shape}")

    # --- 4. Split Data ---
    try:
        stratify_target = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_target)
        split_type = "stratified" if stratify_target is not None else "unstratified"
        print(f"  Data split ({split_type}): Train shape {X_train.shape}, Test shape {X_test.shape}")
    except Exception as e:
        print(f"  Error during data splitting: {e}. Skipping file.")
        return

    # --- 5. Identify Column Types ---
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    print(f"  Numerical features identified: {numerical_cols}")
    print(f"  Categorical features identified: {categorical_cols}")

    # --- 6. Handle All-NaN Columns ---
    cols_to_drop_all_nan = []
    if numerical_cols:
        all_nan_mask = X_train[numerical_cols].isnull().all()
        cols_to_drop_all_nan = all_nan_mask[all_nan_mask].index.tolist()
        if cols_to_drop_all_nan:
            print(f"  Dropping columns entirely NaN in training set: {cols_to_drop_all_nan}")
            X_train.drop(columns=cols_to_drop_all_nan, inplace=True)
            X_test.drop(columns=cols_to_drop_all_nan, inplace=True)
            numerical_cols = [col for col in numerical_cols if col not in cols_to_drop_all_nan]
            print(f"  Remaining numerical features for imputation: {numerical_cols}")

    # --- 7. Impute Missing Values ---
    # 7.1 Numerical Imputation
    if numerical_cols:
        if X_train[numerical_cols].isnull().sum().sum() > 0:
            print(f"  Imputing numerical columns: {numerical_cols}")
            try:
                print("    Trying IterativeImputer...")
                num_imputer = IterativeImputer(max_iter=10, random_state=42)
                X_train_finite = X_train[numerical_cols].replace([np.inf, -np.inf], np.nan)
                finite_cols = X_train_finite.dropna(axis=1, how='all').columns
                if not finite_cols.empty:
                    num_imputer.fit(X_train_finite[finite_cols])
                    X_train_imputed_raw = num_imputer.transform(X_train[numerical_cols])
                    X_train_imputed_rounded = np.round(np.nan_to_num(X_train_imputed_raw), decimals=4)
                    X_train[numerical_cols] = pd.DataFrame(X_train_imputed_rounded, index=X_train.index, columns=numerical_cols)
                    X_test_imputed_raw = num_imputer.transform(X_test[numerical_cols])
                    X_test_imputed_rounded = np.round(np.nan_to_num(X_test_imputed_raw), decimals=4)
                    X_test[numerical_cols] = pd.DataFrame(X_test_imputed_rounded, index=X_test.index, columns=numerical_cols)
                    print(f"    IterativeImputer successful.")
                else: raise ValueError("No columns with finite values for IterativeImputer")
            except Exception as e_iter:
                print(f"    IterativeImputer failed: {e_iter}. Trying SimpleImputer (median)...")
                try:
                    num_imputer_fallback = SimpleImputer(strategy='median')
                    X_train[numerical_cols] = num_imputer_fallback.fit_transform(X_train[numerical_cols])
                    X_test[numerical_cols] = num_imputer_fallback.transform(X_test[numerical_cols])
                    print("    SimpleImputer (median) successful.")
                except Exception as e_simple:
                    print(f"    SimpleImputer also failed: {e_simple}. Dropping numerical columns with remaining NaNs.")
                    cols_to_drop_post_impute = X_train.columns[X_train[numerical_cols].isnull().any()].intersection(numerical_cols)
                    if not cols_to_drop_post_impute.empty:
                         print(f"      Dropping: {cols_to_drop_post_impute.tolist()}")
                         X_train.drop(columns=cols_to_drop_post_impute, inplace=True, errors='ignore')
                         X_test.drop(columns=cols_to_drop_post_impute, inplace=True, errors='ignore')
                         numerical_cols = [col for col in numerical_cols if col not in cols_to_drop_post_impute]
        else: print("  No missing values found in remaining numerical columns.")
    else: print("  No numerical columns left for imputation.")
    # 7.2 Categorical Imputation
    if categorical_cols:
        if X_train[categorical_cols].isnull().sum().sum() > 0:
            print(f"  Imputing categorical columns using SimpleImputer (most_frequent): {categorical_cols}")
            cat_imputer = SimpleImputer(strategy='most_frequent')
            try:
                X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
                X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
                print("  Categorical imputation completed.")
            except Exception as e:
                 print(f"  Error during categorical imputation: {e}. Dropping problematic categorical columns.")
                 cols_to_drop_cat_nan = X_train.columns[X_train[categorical_cols].isnull().any()].intersection(categorical_cols)
                 if not cols_to_drop_cat_nan.empty:
                     print(f"      Dropping: {cols_to_drop_cat_nan.tolist()}")
                     X_train.drop(columns=cols_to_drop_cat_nan, inplace=True, errors='ignore')
                     X_test.drop(columns=cols_to_drop_cat_nan, inplace=True, errors='ignore')
                     categorical_cols = [col for col in categorical_cols if col not in cols_to_drop_cat_nan]
        else: print("  No missing values found in categorical training columns.")
    else: print("  No categorical columns found/remaining for imputation.")

    # --- 8. Encode Categorical Features ---
    if categorical_cols:
        print("  Encoding remaining categorical features (if any)...")
        encoders = {}
        for col in categorical_cols:
            if col not in X_train.columns: continue
            le = LabelEncoder()
            try:
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                encoders[col] = le
                test_col_str = X_test[col].astype(str)
                seen_labels = set(le.classes_)
                unknown_value_placeholder = -1
                test_col_transformed = [le.transform([item])[0] if item in seen_labels else unknown_value_placeholder for item in test_col_str]
                X_test[col] = test_col_transformed
            except Exception as e:
                print(f"  Error encoding categorical column '{col}': {e}. Dropping column.")
                X_train.drop(columns=[col], inplace=True, errors='ignore')
                X_test.drop(columns=[col], inplace=True, errors='ignore')
        print("  Categorical encoding completed.")
    else: print("  No categorical features to encode.")

    # --- 9. Encode Target Column ---
    print("  Encoding target column...")
    target_le = LabelEncoder()
    try:
        y_train = target_le.fit_transform(y_train.astype(str))
        print(f"    Target classes found in training: {target_le.classes_}")
        y_test_str = y_test.astype(str)
        seen_target_labels = set(target_le.classes_)
        y_test_transformed = []
        valid_test_indices = []
        for i, item in enumerate(y_test_str):
            if item in seen_target_labels:
                y_test_transformed.append(target_le.transform([item])[0])
                valid_test_indices.append(True)
            else: valid_test_indices.append(False)
        y_test = np.array(y_test_transformed)
        original_test_count = len(valid_test_indices)
        X_test = X_test[valid_test_indices]
        if X_test.shape[0] < original_test_count:
             print(f"    Removed {original_test_count - X_test.shape[0]} test samples with unseen target labels.")
        print("  Target encoding completed.")
    except Exception as e:
        print(f"  Error encoding target column '{target_col_name}': {e}. Skipping file.")
        return

    # --- 10. Combine Processed Data ---
    final_feature_columns = X_train.columns.tolist()
    if not final_feature_columns:
        print(f"  Error: No feature columns left after processing for {file_path.name}. Skipping.")
        return
    X_test = X_test[final_feature_columns]
    final_train_df = X_train.copy(); final_train_df[target_col_name] = y_train
    final_test_df = X_test.copy()
    if len(y_test) == final_test_df.shape[0]: final_test_df[target_col_name] = y_test
    else: print(f"  Error: Mismatch between X_test rows and y_test length. Skipping save."); return

    # --- 11. Save Files ---
    train_output_path = TRAINING_DIR / file_path.name
    test_output_path = TESTING_DIR / file_path.name
    try:
        final_train_df.to_csv(train_output_path, index=False)
        if not final_test_df.empty:
            final_test_df.to_csv(test_output_path, index=False)
            save_status = f"Train -> {train_output_path}, Test -> {test_output_path}"
        else: save_status = f"Train -> {train_output_path} (Test empty, not saved)"
        print(f"  Successfully processed and saved: {save_status}")
        print(f"    Final features ({len(final_feature_columns)}): {final_feature_columns}")
        print(f"    Final Train shape: {final_train_df.shape}, Final Test shape: {final_test_df.shape}")
    except Exception as e: print(f"  Error saving processed files: {e}")


# --- Main Execution Logic ---
def main():
    print("--- Starting Data Processing (Config-Based Drops, Target Handling) ---")
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

    if not INPUT_DIR.exists(): print(f"Error: Input directory '{INPUT_DIR}' not found."); return
    files_in_config = set(DATASET_CONFIGS.keys())
    print(f"Processing {len(files_in_config)} files defined in configuration...")

    processed_count = 0
    found_files = set()
    for file_path in INPUT_DIR.glob("*.csv"):
        found_files.add(file_path.name)
        if file_path.name in files_in_config:
            process_file(file_path)
            processed_count += 1

    missing_files = files_in_config - found_files
    if missing_files:
        print(f"\nWarning: The following configured files were not found in '{INPUT_DIR}': {missing_files}")

    print(f"\n--- Data Processing Finished (Attempted: {processed_count}) ---")


if __name__ == "__main__":
    main()
