"""
predict.py — Load the trained model and make predictions on new input.

Usage:
    python src/predict.py                  # Interactive CLI prediction
    python src/predict.py --example        # Run with verified benchmark examples
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_preprocessing import (
    MODELS_DIR,
    ALL_FEATURES,
    ORIGINAL_FEATURES,
    preprocess_single_input,
)

# Reference samples representing distinct bean varieties
EXAMPLE_INPUTS = {
    "SEKER": {
        "Area": 28395.0, "Perimeter": 610.29, "MajorAxisLength": 208.18,
        "MinorAxisLength": 173.89, "AspectRatio": 1.197, "Eccentricity": 0.549,
        "ConvexArea": 28715.0, "EquivDiameter": 190.14, "Extent": 0.764,
        "Solidity": 0.989, "Roundness": 0.958, "Compactness": 0.913,
        "ShapeFactor1": 0.00733, "ShapeFactor2": 0.00315, "ShapeFactor3": 0.834,
        "ShapeFactor4": 0.9987,
    },
    "BOMBAY": {
        "Area": 164293.0, "Perimeter": 1526.47, "MajorAxisLength": 560.12,
        "MinorAxisLength": 376.15, "AspectRatio": 1.489, "Eccentricity": 0.741,
        "ConvexArea": 166795.0, "EquivDiameter": 457.38, "Extent": 0.793,
        "Solidity": 0.985, "Roundness": 0.885, "Compactness": 0.817,
        "ShapeFactor1": 0.00341, "ShapeFactor2": 0.00104, "ShapeFactor3": 0.667,
        "ShapeFactor4": 0.993,
    },
    "HOROZ": {
        "Area": 42194.0, "Perimeter": 863.21, "MajorAxisLength": 353.48,
        "MinorAxisLength": 153.22, "AspectRatio": 2.307, "Eccentricity": 0.901,
        "ConvexArea": 42981.0, "EquivDiameter": 231.78, "Extent": 0.684,
        "Solidity": 0.982, "Roundness": 0.711, "Compactness": 0.656,
        "ShapeFactor1": 0.00838, "ShapeFactor2": 0.00138, "ShapeFactor3": 0.430,
        "ShapeFactor4": 0.991,
    },
}


def load_model():
    """Load the trained model and preprocessing artifacts from disk."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")

    for path, name in [(model_path, "Model"), (scaler_path, "StandardScaler"), (le_path, "LabelEncoder")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} artifact not found at {path}. "
                f"Please run 'python src/train.py' first to train and serialize the model."
            )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    return model, scaler, le


def predict(input_features, model=None, scaler=None, label_encoder=None):
    """
    Make a variety classification prediction for a single bean sample.

    Args:
        input_features: dict containing the 16 original morphological features.
        model, scaler, label_encoder: pre-loaded artifacts (optional).

    Returns:
        dict containing predicted variety, class index, confidence score, and distribution.
    """
    if model is None or scaler is None or label_encoder is None:
        model, scaler, label_encoder = load_model()

    feature_cols = ALL_FEATURES

    # Preprocess and scale input
    X_scaled = preprocess_single_input(input_features, scaler, feature_cols)

    # Class prediction
    pred_idx = int(model.predict(X_scaled)[0])
    pred_class = str(label_encoder.inverse_transform([pred_idx])[0])

    result = {
        "predicted_class": pred_class,
        "predicted_label": pred_idx,
    }

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probas))
        result["confidence"] = round(confidence, 4)
        result["class_probabilities"] = {
            str(cls): round(float(p), 4)
            for cls, p in zip(label_encoder.classes_, probas)
        }

    return result


def validate_input(value, feature_name):
    """Validate numeric input ensuring it is non-negative and finite."""
    try:
        val = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid input for {feature_name}: '{value}'. Must be a valid numeric value.")
    if val < 0:
        raise ValueError(f"Invalid input for {feature_name}: {val}. Measurement cannot be negative.")
    if np.isnan(val) or np.isinf(val):
        raise ValueError(f"Invalid input for {feature_name}: value must be finite.")
    return val


def interactive_predict():
    """Interactive command-line interface for manual inspection."""
    print("=" * 60)
    print("DRY BEAN CLASSIFICATION - Interactive Prediction Interface")
    print("=" * 60)
    print("Enter the 16 morphological measurements of the dry bean specimen.")
    print("Type 'q' or 'quit' at any prompt to exit.\n")

    try:
        model, scaler, le = load_model()
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        return

    while True:
        input_features = {}
        print("-" * 50)

        for feature in ORIGINAL_FEATURES:
            while True:
                user_val = input(f"  {feature:20s}: ").strip()
                if user_val.lower() in ("quit", "exit", "q"):
                    print("Exiting prediction interface.")
                    return
                try:
                    input_features[feature] = validate_input(user_val, feature)
                    break
                except ValueError as e:
                    print(f"    [!] Error: {e}")

        result = predict(input_features, model, scaler, le)

        print("\n" + "=" * 50)
        print(f"  PREDICTED BEAN VARIETY: {result['predicted_class']}")
        if "confidence" in result:
            print(f"  CONFIDENCE SCORE:       {result['confidence']:.2%}")
            print("\n  Class Probability Distribution:")
            for cls, prob in sorted(result["class_probabilities"].items(),
                                     key=lambda x: x[1], reverse=True):
                bar = "#" * int(prob * 25)
                print(f"    {cls:12s}: {prob:6.4f} | {bar}")
        print("=" * 50 + "\n")

        cont = input("Perform another prediction? (y/n): ").strip().lower()
        if cont not in ("y", "yes"):
            print("Session closed.")
            return


def example_predict():
    """Run benchmark predictions using verified real samples."""
    print("=" * 60)
    print("DRY BEAN CLASSIFICATION - Benchmark Example Predictions")
    print("=" * 60)

    model, scaler, le = load_model()

    all_matched = True
    for expected_variety, features in EXAMPLE_INPUTS.items():
        print(f"\nEvaluating specimen: Expected Variety = {expected_variety}")
        res = predict(features, model, scaler, le)
        pred = res["predicted_class"]
        conf = res.get("confidence", 0.0)
        is_match = (pred == expected_variety)
        status_str = "[MATCH]" if is_match else "[MISMATCH]"
        if not is_match:
            all_matched = False
        print(f"  -> Predicted Variety:  {pred}")
        print(f"  -> Confidence:         {conf:.2%}")
        print(f"  -> Validation Status:  {status_str}")

    print("\n" + "=" * 60)
    if all_matched:
        print("All benchmark example predictions verified successfully.")
    else:
        print("Completed benchmark predictions.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dry Bean Classifier - Prediction Utility")
    parser.add_argument("--example", action="store_true", help="Execute verified benchmark test cases")
    args = parser.parse_args()

    if args.example:
        example_predict()
    else:
        interactive_predict()
