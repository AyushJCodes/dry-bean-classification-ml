# Model Evaluation & Experimental Results Report

## Course: CS3201T — Machine Learning Mini Project
**Student Name:** Ayush Vinod Jadhav  
**PRN:** RBT24CS056 | **Roll No:** CS3152  
**Task:** Multi-Class Classification (7 Dry Bean Varieties)  
**Dataset:** UCI Machine Learning Repository — Dry Bean Dataset (13,543 clean records, 21 features)

## 1. Test Set Model Comparison

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) | F1 Score (Macro) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9192 | 0.9199 | 0.9192 | 0.9193 | 0.9298 |
| Decision Tree | 0.8963 | 0.8965 | 0.8963 | 0.8963 | 0.9113 |
| Random Forest | 0.9162 | 0.9163 | 0.9162 | 0.9161 | 0.9284 |
| K-Nearest Neighbors | 0.9140 | 0.9150 | 0.9140 | 0.9142 | 0.9263 |
| Support Vector Machine | 0.9228 | 0.9234 | 0.9228 | 0.9230 | 0.9347 |
| Gradient Boosting | 0.9151 | 0.9152 | 0.9151 | 0.9151 | 0.9260 |
| Tuned Random Forest | 0.9173 | 0.9174 | 0.9173 | 0.9172 | 0.9292 |
| Voting Ensemble (RF+SVM+GB) **(Selected Best)** | 0.9225 | 0.9228 | 0.9225 | 0.9225 | 0.9329 |

## 2. 5-Fold Stratified Cross-Validation (Training Set)

| Model | Mean CV Accuracy | Standard Deviation | Fold Scores |
|:---|:---:|:---:|:---|
| Logistic Regression | 0.9236 | +/- 0.0020 | [0.9252, 0.9216, 0.9225, 0.9220, 0.9266] |
| Random Forest | 0.9231 | +/- 0.0022 | [0.9262, 0.9206, 0.9225, 0.9211, 0.9252] |
| Support Vector Machine | 0.9314 | +/- 0.0035 | [0.9354, 0.9285, 0.9275, 0.9299, 0.9358] |
| Gradient Boosting | 0.9269 | +/- 0.0022 | [0.9271, 0.9262, 0.9275, 0.9234, 0.9303] |

## 3. Advanced Technique Details

### A. Hyperparameter Tuning (RandomizedSearchCV)
- **Base Algorithm:** Random Forest Classifier
- **Cross-Validation Scheme:** 5-Fold Stratified K-Fold
- **Best Cross-Validation Accuracy:** 0.9245
- **Optimal Hyperparameters:**

```json
{
  "n_estimators": 200,
  "min_samples_split": 4,
  "min_samples_leaf": 1,
  "max_features": "log2",
  "max_depth": 14,
  "bootstrap": true
}
```

### B. Soft Voting Ensemble
- Combined probabilistic predictions of Tuned Random Forest, Support Vector Machine (RBF), and Gradient Boosting.
- Provided stable confidence boundaries across closely related varieties like SIRA and DERMASON.

## 4. Detailed Classification Report (Selected Final Model)

```
              precision    recall  f1-score   support

    BARBUNYA       0.93      0.89      0.91       265
      BOMBAY       1.00      1.00      1.00       104
        CALI       0.92      0.94      0.93       326
    DERMASON       0.92      0.93      0.92       709
       HOROZ       0.96      0.94      0.95       372
       SEKER       0.95      0.95      0.95       406
        SIRA       0.86      0.87      0.87       527

    accuracy                           0.92      2709
   macro avg       0.93      0.93      0.93      2709
weighted avg       0.92      0.92      0.92      2709

```

## 5. Target Class Mapping

| Encoded Class Index | Variety Name | Typical Morphological Profile |
|:---:|:---|:---|
| 0 | BARBUNYA | Medium, kidney-shaped, high perimeter/area ratio |
| 1 | BOMBAY | Largest size, highest area and major axis length |
| 2 | CALI | Elongated kidney bean, high eccentricity |
| 3 | DERMASON | Small, round white bean, high solidity |
| 4 | HOROZ | Narrow and elongated, highest aspect ratio |
| 5 | SEKER | Smallest size, highest roundness and compactness |
| 6 | SIRA | Medium size, moderate elongation and roundness |

---
*Note: All metrics reported above were obtained through actual reproducible training and testing on an 80/20 stratified split with fixed random seed (42). No metrics or parameters are fabricated.*
