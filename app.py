"""
app.py — Streamlit web interface for the Dry Bean Classifier.

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd

# Add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_preprocessing import ORIGINAL_FEATURES, MODELS_DIR
from predict import load_model, predict, validate_input, EXAMPLE_INPUTS


# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

# ─── Feature descriptions and ranges for UI ─────────────────────────────────
FEATURE_INFO = {
    "Area": {"desc": "Area of the bean zone (pixels)", "min": 20000.0, "max": 255000.0, "default": 50000.0, "step": 100.0},
    "Perimeter": {"desc": "Length of the bean border (pixels)", "min": 500.0, "max": 2000.0, "default": 800.0, "step": 1.0},
    "MajorAxisLength": {"desc": "Longest line within the bean (pixels)", "min": 150.0, "max": 750.0, "default": 300.0, "step": 1.0},
    "MinorAxisLength": {"desc": "Shortest line within the bean (pixels)", "min": 130.0, "max": 500.0, "default": 220.0, "step": 1.0},
    "AspectRatio": {"desc": "Ratio of major to minor axis length", "min": 1.0, "max": 2.5, "default": 1.5, "step": 0.01},
    "Eccentricity": {"desc": "Eccentricity of the fitted ellipse", "min": 0.0, "max": 0.99, "default": 0.5, "step": 0.01},
    "ConvexArea": {"desc": "Pixels in smallest convex polygon", "min": 20000.0, "max": 260000.0, "default": 52000.0, "step": 100.0},
    "EquivDiameter": {"desc": "Diameter of circle with same area", "min": 150.0, "max": 580.0, "default": 250.0, "step": 1.0},
    "Extent": {"desc": "Ratio of bean to bounding box area", "min": 0.5, "max": 0.9, "default": 0.7, "step": 0.01},
    "Solidity": {"desc": "Ratio of bean to convex hull area", "min": 0.9, "max": 1.0, "default": 0.97, "step": 0.001},
    "Roundness": {"desc": "4πArea / Perimeter²", "min": 0.5, "max": 1.0, "default": 0.85, "step": 0.01},
    "Compactness": {"desc": "EquivDiameter / MajorAxisLength", "min": 0.5, "max": 1.0, "default": 0.8, "step": 0.01},
    "ShapeFactor1": {"desc": "Shape factor 1", "min": 0.001, "max": 0.02, "default": 0.006, "step": 0.001},
    "ShapeFactor2": {"desc": "Shape factor 2", "min": 0.0005, "max": 0.01, "default": 0.003, "step": 0.0005},
    "ShapeFactor3": {"desc": "Shape factor 3", "min": 0.4, "max": 1.0, "default": 0.75, "step": 0.01},
    "ShapeFactor4": {"desc": "Shape factor 4", "min": 0.95, "max": 1.0, "default": 0.99, "step": 0.001},
}

BEAN_DESCRIPTIONS = {
    "SEKER": "🟡 A small, round bean common in Turkish cuisine. Known for smooth texture.",
    "BARBUNYA": "🔴 A red-speckled bean popular in Mediterranean dishes. Medium-sized with kidney shape.",
    "BOMBAY": "⚪ A large, flat white bean. The biggest variety in this dataset.",
    "CALI": "🟤 A white kidney-shaped bean used widely in salads and stews.",
    "DERMASON": "⚪ A small white bean, one of the most common varieties in Turkey.",
    "HOROZ": "🟠 A long, slender bean with a distinctive elongated shape.",
    "SIRA": "🟡 A medium-sized, slightly elongated bean variety.",
}


def main():
    st.title("🫘 Dry Bean Classifier")
    st.markdown(
        "Classify dry bean varieties based on geometric shape measurements. "
        "Uses a **Random Forest** model trained on the "
        "[UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry-bean-dataset)."
    )

    # Check if model exists
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        st.error(
            "⚠️ Trained model not found. Please run training first:\n\n"
            "```bash\npython src/train.py\n```"
        )
        return

    # Load model
    try:
        model, scaler, le = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar — Example presets
    st.sidebar.header("📋 Example Presets")
    st.sidebar.markdown("Load example measurements for a known bean variety:")

    selected_example = st.sidebar.selectbox(
        "Choose a variety:",
        ["Custom Input"] + list(EXAMPLE_INPUTS.keys()),
    )

    # Main content — two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📐 Bean Measurements")
        st.markdown("Enter the 16 geometric features measured from the bean image:")

        input_features = {}
        cols = st.columns(2)

        for i, feature in enumerate(ORIGINAL_FEATURES):
            info = FEATURE_INFO[feature]
            col = cols[i % 2]

            # Use example values or defaults
            if selected_example != "Custom Input" and selected_example in EXAMPLE_INPUTS:
                default_val = EXAMPLE_INPUTS[selected_example][feature]
            else:
                default_val = info["default"]

            with col:
                val = st.number_input(
                    f"{feature}",
                    min_value=0.0,
                    value=float(default_val),
                    step=info["step"],
                    help=info["desc"],
                    key=feature,
                )
                input_features[feature] = val

    with col2:
        st.subheader("🔮 Prediction Result")

        if st.button("🫘 Classify Bean", type="primary", use_container_width=True):
            # Validate inputs
            errors = []
            for feat, val in input_features.items():
                try:
                    validate_input(val, feat)
                except ValueError as e:
                    errors.append(str(e))

            if errors:
                for err in errors:
                    st.error(err)
            else:
                with st.spinner("Classifying..."):
                    result = predict(input_features, model, scaler, le)

                # Display result
                predicted = result["predicted_class"]
                st.success(f"### Predicted Variety: **{predicted}**")

                if predicted in BEAN_DESCRIPTIONS:
                    st.info(BEAN_DESCRIPTIONS[predicted])

                if "confidence" in result:
                    st.metric("Confidence", f"{result['confidence']:.1%}")

                    # Probability chart
                    st.markdown("#### Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Variety": list(result["class_probabilities"].keys()),
                        "Probability": list(result["class_probabilities"].values()),
                    }).sort_values("Probability", ascending=True)

                    st.bar_chart(
                        prob_df.set_index("Variety"),
                        horizontal=True,
                    )

                # Disclaimer
                st.caption(
                    "⚠️ This classifier is a machine learning demonstration trained on the "
                    "UCI Dry Bean Dataset. It is not intended for production-grade quality "
                    "grading or commercial use."
                )

    # Bottom section — About
    st.divider()
    st.markdown(
        "**About this project:** CS3201T Machine Learning Mini Project — "
        "Dry Bean Classification using geometric features and Random Forest. "
        "Built by Ayush Vinod Jadhav."
    )


if __name__ == "__main__":
    main()
