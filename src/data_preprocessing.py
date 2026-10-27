"""
data_preprocessing.py — Download, load, clean, and preprocess the Dry Bean Dataset.

Handles:
- Downloading the dataset from UCI ML Repository
- Loading and inspecting the data
- Robust column standardization (AspectRation -> AspectRatio, roundness -> Roundness)
- Handling missing values and duplicates
- Domain-informed feature engineering (derived shape ratios)
- Feature scaling with StandardScaler
- Label encoding the target variable
- Stratified train-test splitting
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "dry_beans.csv")

# UCI ML Repository download URL
DATASET_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"


# --- Feature definitions ----------------------------------------------------
ORIGINAL_FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]

ENGINEERED_FEATURES = [
    "PerimeterToAreaRatio",
    "AreaToConvexAreaRatio",
    "CompactnessTimesRoundness",
    "MajorMinorProduct",
    "ShapeIndex",
]

ALL_FEATURES = ORIGINAL_FEATURES + ENGINEERED_FEATURES

TARGET_COLUMN = "Class"

COLUMN_STANDARDIZATION_MAP = {
    "AspectRation": "AspectRatio",
    "roundness": "Roundness",
    "aspectratio": "AspectRatio",
    "aspectration": "AspectRatio",
}


def standardize_columns(df):
    """
    Standardize column names across raw dataset and manual inputs.
    Addresses raw dataset variations such as 'AspectRation' and 'roundness'.
    """
    df = df.copy()
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_STANDARDIZATION_MAP:
            rename_dict[col] = COLUMN_STANDARDIZATION_MAP[col]
        elif col.lower() in COLUMN_STANDARDIZATION_MAP:
            rename_dict[col] = COLUMN_STANDARDIZATION_MAP[col.lower()]
    if rename_dict:
        df = df.rename(columns=rename_dict)
    return df


def download_dataset():
    """Download the Dry Bean Dataset from UCI ML Repository if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_DATA_PATH) and os.path.getsize(RAW_DATA_PATH) > 100000:
        return RAW_DATA_PATH

    zip_path = os.path.join(DATA_DIR, "dry_bean_dataset.zip")
    print(f"Downloading Dry Bean Dataset from UCI ML Repository...")
    print(f"URL: {DATASET_URL}")

    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print(f"Downloaded to {zip_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset: {e}\n"
            f"Please download manually from:\n"
            f"  https://archive.ics.uci.edu/dataset/602/dry-bean-dataset\n"
            f"and place the CSV file at: {RAW_DATA_PATH}"
        )

    # Extract the ZIP file
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    # Find the Excel/CSV file inside the extracted contents
    extracted_file = None
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith((".csv", ".xlsx", ".xls", ".arff")):
                extracted_file = os.path.join(root, f)
                break
        if extracted_file:
            break

    if extracted_file is None:
        raise FileNotFoundError(
            "Could not find data file in the downloaded archive."
        )

    # Convert to CSV if needed
    if extracted_file.lower().endswith((".xlsx", ".xls")):
        print(f"Converting {os.path.basename(extracted_file)} to CSV...")
        df = pd.read_excel(extracted_file)
        df.to_csv(RAW_DATA_PATH, index=False)
    elif extracted_file.lower().endswith(".arff"):
        print("Converting ARFF to CSV...")
        df = _read_arff(extracted_file)
        df.to_csv(RAW_DATA_PATH, index=False)
    else:
        if os.path.abspath(extracted_file) != os.path.abspath(RAW_DATA_PATH):
            import shutil
            shutil.copy2(extracted_file, RAW_DATA_PATH)

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Dataset ready at {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def _read_arff(path):
    """Simple ARFF reader for the Dry Bean Dataset."""
    data_started = False
    rows = []
    columns = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                parts = line.split()
                columns.append(parts[1])
            elif line.upper().startswith("@DATA"):
                data_started = True
            elif data_started:
                rows.append(line.split(","))
    return pd.DataFrame(rows, columns=columns)


def load_data(filepath=None):
    """Load the dry bean dataset from CSV and standardize column names."""
    if filepath is None:
        filepath = RAW_DATA_PATH

    if not os.path.exists(filepath):
        print("Dataset not found locally. Downloading...")
        download_dataset()

    df = pd.read_csv(filepath)
    df = standardize_columns(df)

    # Ensure numeric columns are properly typed
    for col in ORIGINAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def inspect_data(df):
    """Print a comprehensive summary of the dataset."""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumn names:\n{list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")

    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}")
    print(f"Total missing: {missing.sum()}")

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")

    print(f"\nDescriptive statistics:")
    print(df.describe().round(4))

    if TARGET_COLUMN in df.columns:
        print(f"\nTarget class distribution:")
        print(df[TARGET_COLUMN].value_counts())
        print(f"\nTarget class proportions:")
        print(df[TARGET_COLUMN].value_counts(normalize=True).round(4))

    return {
        "shape": df.shape,
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "classes": df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df.columns else {},
    }


def clean_data(df):
    """Clean the dataset: standardize columns, handle missing values and duplicates."""
    df_clean = standardize_columns(df)
    initial_rows = len(df_clean)

    # Handle missing values
    missing_count = int(df_clean.isnull().sum().sum())
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Imputing with median...")
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                print(f"  Filled {col} missing values with median ({median_val:.4f})")
    else:
        print("No missing values found.")

    # Handle duplicates
    dup_count = int(df_clean.duplicated().sum())
    if dup_count > 0:
        print(f"Found {dup_count} duplicate rows. Removing...")
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"  Rows after deduplication: {len(df_clean)} (removed {dup_count})")
    else:
        print("No duplicate rows found.")

    print(f"Cleaning complete: {initial_rows} -> {len(df_clean)} rows")
    return df_clean


def engineer_features(df):
    """
    Create domain-informed features from geometric measurements:
    - PerimeterToAreaRatio: Higher values indicate more elongated/irregular shapes
    - AreaToConvexAreaRatio: Measures convexity / compactness
    - CompactnessTimesRoundness: Combined geometric regularity score
    - MajorMinorProduct: Approximate bounding ellipse area proxy
    - ShapeIndex: Product of AspectRatio and Eccentricity
    """
    df_eng = standardize_columns(df)

    # Ensure required base columns exist
    for col in ["Perimeter", "Area", "ConvexArea", "Compactness", "Roundness", "MajorAxisLength", "MinorAxisLength", "AspectRatio", "Eccentricity"]:
        if col not in df_eng.columns:
            raise KeyError(f"Required column '{col}' missing for feature engineering.")

    df_eng["PerimeterToAreaRatio"] = df_eng["Perimeter"] / (df_eng["Area"] + 1e-8)
    df_eng["AreaToConvexAreaRatio"] = df_eng["Area"] / (df_eng["ConvexArea"] + 1e-8)
    df_eng["CompactnessTimesRoundness"] = df_eng["Compactness"] * df_eng["Roundness"]
    df_eng["MajorMinorProduct"] = df_eng["MajorAxisLength"] * df_eng["MinorAxisLength"]
    df_eng["ShapeIndex"] = df_eng["AspectRatio"] * df_eng["Eccentricity"]

    print(f"Engineered {len(ENGINEERED_FEATURES)} features: {ENGINEERED_FEATURES}")
    return df_eng


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline:
    1. Separate features and target
    2. Encode target labels
    3. Stratified train-test split (avoids target leakage)
    4. Fit scaler on training set ONLY, transform both
    5. Save artifacts to models/ directory

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    df_prepared = standardize_columns(df)
    feature_cols = [c for c in ALL_FEATURES if c in df_prepared.columns]
    
    if len(feature_cols) != len(ALL_FEATURES):
        missing_feats = set(ALL_FEATURES) - set(feature_cols)
        raise ValueError(f"Missing expected features in prepare_data: {missing_feats}")

    X = df_prepared[feature_cols].copy()
    y = df_prepared[TARGET_COLUMN].copy()

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    label_map = dict(zip(le.classes_, [int(x) for x in le.transform(le.classes_)]))
    print(f"Label encoding mapping: {label_map}")

    # Stratified split ensures equal class representation in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    print(f"Saved scaler and label_encoder to {MODELS_DIR}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


def preprocess_single_input(input_dict, scaler=None, feature_cols=None):
    """
    Preprocess a single input sample dictionary for real-time prediction.
    """
    if scaler is None:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    # Convert dictionary to DataFrame and standardize column names
    df_input = pd.DataFrame([input_dict])
    df_input = standardize_columns(df_input)

    # Cast to numeric
    for col in df_input.columns:
        df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    # Engineer features if not already computed
    if "PerimeterToAreaRatio" not in df_input.columns:
        df_input = engineer_features(df_input)

    # Reorder columns exactly as expected by the scaler and model
    df_input = df_input[feature_cols]

    # Scale using the fitted scaler
    scaled_array = scaler.transform(df_input)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_cols)
    return scaled_df


if __name__ == "__main__":
    download_dataset()
    df_raw = load_data()
    inspect_data(df_raw)
    df_cleaned = clean_data(df_raw)
    df_engineered = engineer_features(df_cleaned)
    print(f"\nFinal dataset shape: {df_engineered.shape}")
    print(f"Total features: {len(ALL_FEATURES)}")
