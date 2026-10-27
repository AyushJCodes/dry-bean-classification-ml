"""
test_project.py — Comprehensive automated test suite for Dry Bean Classification.

Tests:
1. Data Loading & Integrity
2. Preprocessing & Data Cleaning
3. Feature Engineering & Leakage Prevention
4. Model Loading & Serialization
5. Prediction Pipeline & Probability Calibration
6. Input Validation & Error Handling

Run with:
    pytest tests/test_project.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_preprocessing import (
    download_dataset,
    load_data,
    clean_data,
    engineer_features,
    prepare_data,
    preprocess_single_input,
    ORIGINAL_FEATURES,
    ENGINEERED_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
    RAW_DATA_PATH,
    MODELS_DIR,
)
from predict import load_model, predict, validate_input, EXAMPLE_INPUTS


# --- Fixtures ----------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_dataset():
    """Load raw dataset."""
    download_dataset()
    return load_data()


@pytest.fixture(scope="session")
def cleaned_dataset(raw_dataset):
    """Cleaned dataset without duplicates."""
    return clean_data(raw_dataset)


@pytest.fixture(scope="session")
def engineered_dataset(cleaned_dataset):
    """Dataset with engineered features."""
    return engineer_features(cleaned_dataset)


@pytest.fixture(scope="session")
def trained_artifacts():
    """Load serialized model artifacts from models/."""
    model, scaler, le = load_model()
    return model, scaler, le


# --- 1. Data Loading Tests ---------------------------------------------------

class TestDataLoading:
    def test_dataset_file_exists(self):
        """Dataset file should be present on disk."""
        assert os.path.exists(RAW_DATA_PATH), f"Raw dataset missing at {RAW_DATA_PATH}"
        assert os.path.getsize(RAW_DATA_PATH) > 1000000, "Dataset file seems incomplete or empty"

    def test_dataset_dimensions(self, raw_dataset):
        """Dataset should have at least 13,000 samples and 17 columns."""
        assert raw_dataset.shape[0] >= 13000, f"Expected >=13,000 rows, got {raw_dataset.shape[0]}"
        assert raw_dataset.shape[1] >= 17, f"Expected >=17 columns, got {raw_dataset.shape[1]}"

    def test_target_column_present(self, raw_dataset):
        """Target column 'Class' must be present."""
        assert TARGET_COLUMN in raw_dataset.columns

    def test_all_original_features_present(self, raw_dataset):
        """All 16 original features must be present after loading."""
        for feat in ORIGINAL_FEATURES:
            assert feat in raw_dataset.columns, f"Missing original feature: {feat}"

    def test_target_classes_count(self, raw_dataset):
        """Target must contain exactly 7 distinct dry bean varieties."""
        unique_classes = raw_dataset[TARGET_COLUMN].dropna().unique()
        assert len(unique_classes) == 7, f"Expected 7 classes, found {len(unique_classes)}: {unique_classes}"

    def test_target_no_missing(self, raw_dataset):
        """Target variable must have zero missing values."""
        assert raw_dataset[TARGET_COLUMN].isnull().sum() == 0


# --- 2. Preprocessing & Cleaning Tests ---------------------------------------

class TestPreprocessing:
    def test_cleaning_removes_duplicates(self, raw_dataset):
        """Cleaning must detect and remove duplicate rows."""
        df_clean = clean_data(raw_dataset)
        assert df_clean.duplicated().sum() == 0
        assert len(df_clean) < len(raw_dataset), "Expected raw dataset to contain duplicates"

    def test_cleaned_has_no_missing_values(self, cleaned_dataset):
        """Cleaned dataset must contain zero NaN values."""
        assert cleaned_dataset.isnull().sum().sum() == 0

    def test_feature_engineering_adds_all_features(self, cleaned_dataset):
        """Feature engineering must create all planned domain ratios."""
        df_eng = engineer_features(cleaned_dataset)
        for col in ENGINEERED_FEATURES:
            assert col in df_eng.columns, f"Engineered column '{col}' missing"
            assert df_eng[col].isnull().sum() == 0, f"Engineered column '{col}' has NaNs"
            assert np.isinf(df_eng[col]).sum() == 0, f"Engineered column '{col}' has Infinities"

    def test_stratified_split_ratios(self, engineered_dataset):
        """Train-test split must respect the 80/20 proportion."""
        X_train, X_test, y_train, y_test, _, _ = prepare_data(engineered_dataset, test_size=0.2, random_state=42)
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.19 <= test_ratio <= 0.21, f"Expected test ratio ~0.20, got {test_ratio:.4f}"

    def test_no_data_leakage(self, engineered_dataset):
        """Train indices and test indices must be strictly disjoint."""
        X_train, X_test, y_train, y_test, _, _ = prepare_data(engineered_dataset, test_size=0.2, random_state=42)
        intersection = set(X_train.index).intersection(set(X_test.index))
        assert len(intersection) == 0, f"Data leakage detected: {len(intersection)} shared indices"

    def test_scaling_fitted_on_train_only(self, engineered_dataset):
        """Mean of training features after scaling should be approximately 0.0."""
        X_train, X_test, _, _, scaler, _ = prepare_data(engineered_dataset, test_size=0.2, random_state=42)
        train_means = np.abs(X_train.mean(axis=0))
        assert np.all(train_means < 0.05), "Training set was not centered near 0.0"


# --- 3. Model Artifact Tests -------------------------------------------------

class TestModelArtifacts:
    def test_model_files_exist(self):
        """Serialized joblib artifacts must exist in models/ directory."""
        for filename in ["best_model.joblib", "scaler.joblib", "label_encoder.joblib"]:
            file_path = os.path.join(MODELS_DIR, filename)
            assert os.path.exists(file_path), f"Artifact missing: {file_path}"
            assert os.path.getsize(file_path) > 100, f"Artifact empty: {file_path}"

    def test_model_callable_methods(self, trained_artifacts):
        """Loaded model must implement predict and predict_proba."""
        model, scaler, le = trained_artifacts
        assert hasattr(model, "predict"), "Model missing predict method"
        assert hasattr(model, "predict_proba"), "Model missing predict_proba method"

    def test_label_encoder_classes(self, trained_artifacts):
        """Label encoder must know the 7 target classes."""
        _, _, le = trained_artifacts
        assert len(le.classes_) == 7
        for expected in ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]:
            assert expected in le.classes_


# --- 4. Prediction Pipeline Tests --------------------------------------------

class TestPredictionPipeline:
    def test_seker_sample_prediction(self, trained_artifacts):
        """SEKER benchmark sample should classify into valid class."""
        result = predict(EXAMPLE_INPUTS["SEKER"])
        assert "predicted_class" in result
        assert result["predicted_class"] in ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
        assert "confidence" in result
        assert 0.0 < result["confidence"] <= 1.0

    def test_bombay_sample_prediction(self, trained_artifacts):
        """BOMBAY benchmark sample should classify into valid class."""
        result = predict(EXAMPLE_INPUTS["BOMBAY"])
        assert result["predicted_class"] == "BOMBAY"
        assert result["confidence"] > 0.80

    def test_probability_distribution_sums_to_one(self, trained_artifacts):
        """Probabilities across all 7 classes should sum to ~1.0."""
        result = predict(EXAMPLE_INPUTS["HOROZ"])
        probs = result["class_probabilities"]
        assert len(probs) == 7
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, expected 1.0"


# --- 5. Input Validation & Edge Case Tests -----------------------------------

class TestInputValidation:
    def test_valid_float_parsing(self):
        """Valid numeric string must convert to float."""
        assert validate_input("123.45", "Area") == 123.45
        assert validate_input(500, "Perimeter") == 500.0

    def test_negative_values_rejected(self):
        """Negative values must raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_input("-15.0", "Area")

    def test_non_numeric_rejected(self):
        """Non-numeric strings must raise ValueError."""
        with pytest.raises(ValueError, match="Must be a valid numeric value"):
            validate_input("abc", "Solidity")

    def test_nan_and_inf_rejected(self):
        """NaN and Infinity must raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_input(float("nan"), "Extent")
        with pytest.raises(ValueError, match="must be finite"):
            validate_input(float("inf"), "Extent")
