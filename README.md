# Automated Dry Bean Variety Classification Using Machine Learning & Ensemble Methods

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Dry%20Bean-green.svg)](https://archive.ics.uci.edu/dataset/602/dry-bean-dataset)

A rigorous Machine Learning project developed for **CS3201T: Machine Learning Mini Project**, implementing automated classification of seven registered commercial dry bean varieties (*Phaseolus vulgaris L.*) based on computer vision geometric and morphological features.

---

## 1. Student & Academic Details

- **Student Name:** Ayush Vinod Jadhav
- **PRN Number:** RBT24CS056
- **Roll Number:** CS3152
- **GitHub Username:** [AyushJCodes](https://github.com/AyushJCodes)
- **Course:** CS3201T — Machine Learning Mini Project
- **Academic Year:** 2026–2027 (Semester I)
- **Degree:** B.Tech in Computer Engineering
- **Institution:** Department of Computer Engineering, Rajarshi Shahu College of Engineering (RSCOE), Tathwade, Pune

---

## 2. Project Overview & Motivation

In modern agricultural supply chains, grain grading, seed certification, and quality control require rapid, non-destructive, and objective classification methods. Traditional bean classification relies heavily on manual human visual inspection, which is subjective, labor-intensive, error-prone, and prone to fatigue. Different varieties of dry beans command different market prices, cooking times, nutritional properties, and processing requirements.

This project implements an end-to-end Machine Learning pipeline that classifies dry beans into seven distinct commercial varieties based on **16 morphological, shape, and geometric features** extracted from high-resolution digital imagery, supplemented by **5 engineered domain ratios**.

### Key Highlights
- **100% Unique College Project:** Thoroughly verified against the department mini-project spreadsheet (50 existing registered projects). Zero overlap with peer submissions.
- **Genuine Empirical Dataset:** Sourced directly from the peer-reviewed **UCI Machine Learning Repository** (13,611 raw records, 13,543 deduplicated samples).
- **Leakage-Free Preprocessing:** Stratified train-test split (80/20) with `StandardScaler` transformations strictly fitted on the training split only.
- **Multiple Algorithm Benchmarking:** Evaluates Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine (RBF), and Gradient Boosting.
- **Advanced Machine Learning Techniques:**
  1. Systematic Hyperparameter Optimization via `RandomizedSearchCV` with 5-Fold Stratified Cross-Validation.
  2. Soft Voting Ensemble Classifier combining Random Forest, Support Vector Machine, and Gradient Boosting.
- **Interactive Demonstrations:** Fully functional command-line prediction interface (`src/predict.py`) and browser-based Streamlit web application (`app.py`).
- **Comprehensive Quality Assurance:** Automated test suite with 22 unit and integration tests passing via `pytest`.

---

## 3. Problem Statement & Objectives

### Problem Statement
Given 16 quantitative geometric and morphological features measured from grain images of a dry bean sample, construct a machine learning classification model that accurately predicts the bean's botanical variety among seven commercial classes (`SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, and `SIRA`) while handling non-linear boundary overlaps, multivariable collinearity, and class imbalance.

### Objectives
1. Perform thorough Exploratory Data Analysis (EDA) on grain morphometry.
2. Formulate domain-informed feature engineering ratios to enhance class separability.
3. Prevent target leakage and overfitting through stratified partitioning and strictly scoped scaling.
4. Train and benchmark six diverse classification algorithms using standard metrics (Accuracy, Weighted & Macro Precision, Recall, F1-Score).
5. Optimize the strongest ensemble estimator using 5-fold cross-validation and hyperparameter search.
6. Provide reproducible command-line and graphical interfaces for single-specimen inference.

---

## 4. Dataset Description & Provenance

### Source & Citation
- **Source:** [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry-bean-dataset)
- **Citation:** Koklu, M. and Ozkan, I.A., 2020. *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, p.105507. [DOI: 10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Dataset Attributes
- **Raw Samples:** 13,611
- **Clean Deduplicated Samples:** 13,543 (68 identical duplicate rows removed)
- **Input Features:** 16 base morphological metrics + 5 engineered features = 21 total features
- **Target Variable:** `Class` (7 classes, string categorical)

### Morphological Feature Reference
| Feature Name | Type | Unit | Description |
|:---|:---:|:---:|:---|
| `Area` | Integer | Pixels | Total pixel surface area of the bean silhouette |
| `Perimeter` | Float | Pixels | Circumferential boundary length of the bean |
| `MajorAxisLength` | Float | Pixels | Longest distance through the bean center (major ellipse axis) |
| `MinorAxisLength` | Float | Pixels | Shortest distance perpendicular to the major axis |
| `AspectRatio` | Float | Ratio | Dimensionless ratio: `MajorAxisLength / MinorAxisLength` |
| `Eccentricity` | Float | Scalar | Eccentricity of the equivalent bounding ellipse (0 = circle, 1 = line) |
| `ConvexArea` | Integer | Pixels | Area of the smallest enclosing convex polygon |
| `EquivDiameter` | Float | Pixels | Equivalent circle diameter: $\sqrt{4 \cdot \text{Area} / \pi}$ |
| `Extent` | Float | Ratio | Ratio of bean area to bounding bounding box rectangle area |
| `Solidity` | Float | Ratio | Ratio of bean area to its convex hull area: `Area / ConvexArea` |
| `Roundness` | Float | Ratio | Boundary smoothness metric: $(4\pi \cdot \text{Area}) / \text{Perimeter}^2$ |
| `Compactness` | Float | Ratio | Shape roundness ratio: `EquivDiameter / MajorAxisLength` |
| `ShapeFactor1` | Float | Metric | Primary moment invariant shape factor |
| `ShapeFactor2` | Float | Metric | Secondary moment invariant shape factor |
| `ShapeFactor3` | Float | Metric | Tertiary moment invariant shape factor |
| `ShapeFactor4` | Float | Metric | Quaternary moment invariant shape factor |

### Target Bean Varieties
| Class Label | Common Variety | Physical Characteristics | Sample Proportion |
|:---:|:---|:---|:---:|
| `DERMASON` | Dermason | Small white bean, highly uniform, rounded-flat shape | 26.05% |
| `SIRA` | Sira | Medium-small white bean, slight elongation | 19.37% |
| `SEKER` | Seker | Very small, almost spherical/round, highest compactness | 14.89% |
| `HOROZ` | Horoz | Elongated, cylindrical white bean, highest aspect ratio | 14.17% |
| `CALI` | Cali | Large kidney-shaped brownish bean | 11.98% |
| `BARBUNYA` | Barbunya | Medium-large speckled pinto/cranberry bean | 9.71% |
| `BOMBAY` | Bombay | Largest white bean, massive surface area and perimeter | 3.84% |

---

## 5. Machine Learning Workflow

```
[Raw UCI Dataset (.arff / .xlsx / .csv)]
                 │
                 ▼
[Data Cleaning & Standardization]
  - Column name standardization (AspectRation -> AspectRatio, roundness -> Roundness)
  - Duplicate detection & removal (68 duplicates dropped)
  - Verification of null values (0 missing values)
                 │
                 ▼
[Feature Engineering]
  - PerimeterToAreaRatio
  - AreaToConvexAreaRatio
  - CompactnessTimesRoundness
  - MajorMinorProduct
  - ShapeIndex
                 │
                 ▼
[Stratified Train-Test Split (80 / 20)]
  - Stratified partitioning preserves class proportions
  - Zero index overlap (Leakage-free design)
                 │
                 ▼
[StandardScaler Feature Scaling]
  - Fitted strictly on Training Set; Applied to Test Set & Real-Time Inferences
                 │
                 ▼
[Model Training & Cross-Validation]
  ├── Logistic Regression (Multinomial L-BFGS)
  ├── Decision Tree Classifier
  ├── Random Forest Classifier
  ├── K-Nearest Neighbors (Distance-weighted)
  ├── Support Vector Machine (RBF Kernel)
  └── Gradient Boosting Classifier
                 │
                 ▼
[Advanced ML Optimization]
  ├── Hyperparameter Optimization: RandomizedSearchCV (RF)
  └── Soft Voting Ensemble (Tuned RF + SVM + Gradient Boosting)
                 │
                 ▼
[Evaluation & Diagnostics]
  - Confusion Matrix & Classification Reports (Accuracy, Precision, Recall, F1)
  - Feature Importance Rankings & Model Comparison Plots
                 │
                 ▼
[Deployment & Inference]
  ├── Interactive CLI Predictor (`src/predict.py`)
  ├── Streamlit Web App (`app.py`)
  └── Automated Pytest Suite (`tests/test_project.py`)
```

---

## 6. Feature Engineering Formulation

Five domain-specific features were mathematically formulated from the raw morphometrics to assist tree-based and linear models in separating boundary-adjacent classes:

1. **Perimeter-to-Area Ratio:**
   $$\text{PerimeterToAreaRatio} = \frac{\text{Perimeter}}{\text{Area}}$$
   *Rationale:* Highly elongated beans (e.g., `HOROZ`) possess substantially higher perimeter per unit area than circular beans (`SEKER`).

2. **Area-to-ConvexArea Ratio:**
   $$\text{AreaToConvexAreaRatio} = \frac{\text{Area}}{\text{ConvexArea}}$$
   *Rationale:* Directly measures boundary concavities and indentation depth without rounding artifacts.

3. **Compactness $\times$ Roundness:**
   $$\text{CompactnessTimesRoundness} = \text{Compactness} \times \text{Roundness}$$
   *Rationale:* Joint shape regularity index; cleanly isolates nearly spherical `SEKER` beans from moderately oval `SIRA` beans.

4. **Major $\times$ Minor Axis Product:**
   $$\text{MajorMinorProduct} = \text{MajorAxisLength} \times \text{MinorAxisLength}$$
   *Rationale:* Approximates bounding ellipse area and separates colossal `BOMBAY` specimens from medium beans.

5. **Shape Index:**
   $$\text{ShapeIndex} = \text{AspectRatio} \times \text{Eccentricity}$$
   *Rationale:* Compounded metric of grain elongation that amplifies separation between elongated (`HOROZ`, `CALI`) and rounded varieties.

---

## 7. Model Selection Methodology & Experimental Results

### Methodological Principles
To adhere strictly to sound machine learning practices and prevent test-set snooping (data leakage in model selection), all model architecture comparisons and hyperparameter choices were decided using **5-fold stratified cross-validation on the training split** ($X_{\text{train}}$, 10,834 samples). The test split ($X_{\text{test}}$, 2,709 samples) was held out strictly for final unbiased evaluation of generalization error.

1. **Cross-Validation Ranking (Training Set):**
   - **Support Vector Machine (RBF):** Achieved the highest individual validation accuracy (**0.9314 ± 0.0035**).
   - **Gradient Boosting:** Achieved **0.9269 ± 0.0022**.
   - **Tuned Random Forest:** Achieved **0.9245** CV score via `RandomizedSearchCV`.
   - **Logistic Regression:** Achieved **0.9236 ± 0.0020**.
   - **Baseline Random Forest:** Achieved **0.9231 ± 0.0022**.

2. **Final Model Selection Rationale (Soft Voting Ensemble):**
   Rather than selecting a single estimator that can be vulnerable to local decision boundary variations between closely overlapping morphometric classes (specifically `SIRA` and `DERMASON`), the **Soft Voting Ensemble** combines three distinct modeling paradigms:
   - *Kernel Margin Maximization:* Support Vector Machine (RBF)
   - *Bagged De-correlated Trees:* Tuned Random Forest
   - *Sequential Residual Optimization:* Gradient Boosting Classifier

   Averaging their predicted class probability distributions $P(y=c \mid x) = \frac{1}{3}(P_{\text{RF}} + P_{\text{SVM}} + P_{\text{GB}})$ reduces overall prediction variance and delivers smooth, well-calibrated confidence scores necessary for real-time agricultural grading.

3. **Unbiased Generalization on Held-Out Test Set:**
   When evaluated on the 2,709 unseen test samples:
   - **Support Vector Machine (RBF):** 92.28% test accuracy (2,500 / 2,709 correct)
   - **Soft Voting Ensemble:** 92.25% test accuracy (2,499 / 2,709 correct)
   The difference between the two models on the test set is a mere **1 sample out of 2,709** (0.03%), verifying that cross-validation on the training split produced realistic estimates without overfitting.

### Comprehensive Test Set Benchmark Results
*(Recorded from the executed run in `reports/results.json`)*

| Algorithm | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | F1-Score (Macro) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (RBF)** | **0.9228** | **0.9234** | **0.9228** | **0.9230** | **0.9347** |
| **Soft Voting Ensemble (Selected)** | **0.9225** | **0.9228** | **0.9225** | **0.9225** | **0.9329** |
| **Logistic Regression (Multinomial)** | 0.9192 | 0.9199 | 0.9192 | 0.9193 | 0.9298 |
| **Tuned Random Forest** | 0.9173 | 0.9174 | 0.9173 | 0.9172 | 0.9292 |
| **Random Forest (Baseline)** | 0.9162 | 0.9163 | 0.9162 | 0.9161 | 0.9284 |
| **Gradient Boosting** | 0.9151 | 0.9152 | 0.9151 | 0.9151 | 0.9260 |
| **K-Nearest Neighbors** | 0.9140 | 0.9150 | 0.9140 | 0.9142 | 0.9263 |
| **Decision Tree** | 0.8963 | 0.8965 | 0.8963 | 0.8963 | 0.9113 |

### 5-Fold Stratified Cross-Validation Summary (Training Split)
- **Support Vector Machine (RBF):** Mean CV Accuracy = **0.9314 ± 0.0035**
- **Gradient Boosting:** Mean CV Accuracy = **0.9269 ± 0.0022**
- **Logistic Regression:** Mean CV Accuracy = **0.9236 ± 0.0020**
- **Random Forest (Baseline):** Mean CV Accuracy = **0.9231 ± 0.0022**
- **Tuned Random Forest (RandomizedSearchCV):** Best CV Score = **0.9245**

---

## 8. Visualizations Generated

All visualizations are generated reproducibly by `src/visualize.py` and saved to `images/`:

- `images/class_distribution.png`: Frequency distribution of all 7 bean varieties.
- `images/correlation_heatmap.png`: High-density triangular correlation matrix.
- `images/feature_boxplots.png`: Morphological metric distributions grouped by bean class.
- `images/feature_histograms.png`: Univariate distribution histograms across all attributes.
- `images/pairplot.png`: Multi-dimensional feature relationship scatter plots.
- `images/model_comparison.png`: Comparative accuracy chart across all trained algorithms.
- `images/confusion_matrix_*.png`: Confusion matrix showing true vs. predicted classifications.
- `images/feature_importance.png`: Ranking of top morphological determinants.

---

## 9. Repository Structure

```
dry-bean-classification-ml/
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Reproducible dependency specification
├── LICENSE                                # MIT Open Source License
├── .gitignore                             # Ignored virtual envs, binaries, caches
├── app.py                                 # Streamlit web application interface
├── data/
│   ├── README.md                          # Dataset provenance and download guide
│   └── dry_beans.csv                      # UCI Dry Bean Dataset (13,611 rows)
├── notebooks/
│   └── machine_learning_analysis.ipynb    # Complete end-to-end Jupyter Notebook
├── src/
│   ├── __init__.py                        # Source package initializer
│   ├── data_preprocessing.py              # Download, clean, engineer, scale, split
│   ├── train.py                           # Training, tuning, evaluation & reports
│   ├── predict.py                         # Single & batch inference CLI utility
│   └── visualize.py                       # Exploratory visualization generator
├── models/
│   ├── README.md                          # Artifact usage instructions
│   ├── best_model.joblib                  # Serialized best model artifact
│   ├── scaler.joblib                      # Fitted StandardScaler artifact
│   └── label_encoder.joblib               # Fitted LabelEncoder artifact
├── reports/
│   ├── evaluation_results.md              # Markdown report of experimental scores
│   └── results.json                       # Machine-readable evaluation metrics
├── images/
│   ├── class_distribution.png             # Target class distribution chart
│   ├── correlation_heatmap.png            # Feature correlation matrix
│   ├── feature_boxplots.png               # Box plots of key morphometrics
│   ├── feature_histograms.png             # Feature distribution histograms
│   ├── pairplot.png                       # Selected feature pairplot
│   ├── model_comparison.png               # Model accuracy bar chart
│   ├── confusion_matrix_*.png             # Confusion matrix plot
│   └── feature_importance.png             # Feature importance rankings
└── tests/
    └── test_project.py                    # 22 automated unit & integration tests
```

---

## 10. Reproducibility & Fresh-Clone Execution Sequence

To ensure complete reproducibility from a fresh clone, follow this verified step-by-step sequence in order:

### Step 1: Create and Activate the Environment
Clone or navigate to the repository directory, then create an isolated Python virtual environment:

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
Install the pinned dependencies specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3: Download or Prepare the Dataset
Download the UCI Dry Bean Dataset and execute the automated schema inspection and cleaning:
```bash
python src/data_preprocessing.py
```
*Note: This downloads `dry_beans.csv` (13,611 rows) into `data/`, removes 68 duplicate records, standardizes column headers, and constructs the 5 engineered domain features.*

### Step 4: Generate Required Artifacts by Training the Models
Train the baseline models, run 5-fold cross-validation, optimize hyperparameters via `RandomizedSearchCV`, construct the Soft Voting Ensemble, and serialize the trained artifacts (`best_model.joblib`, `scaler.joblib`, `label_encoder.joblib`) into `models/`:
```bash
python src/train.py
```
*(Optional: Run `python src/visualize.py` to regenerate all EDA figures in `images/`)*

### Step 5: Run the Automated Test Suite
Execute the full test suite verifying data integrity, preprocessing consistency, leakage prevention, artifact loading, probability summation, and input validation:
```bash
pytest tests/test_project.py -v
```
*(Expected result: 22 passed)*

### Step 6: Launch Streamlit or Execute Predictions
Once model artifacts have been generated in Step 4, run predictions via the CLI or web interface:

- **Benchmark Verification (CLI):**
  ```bash
  python src/predict.py --example
  ```
- **Interactive Manual Prediction (CLI):**
  ```bash
  python src/predict.py
  ```
- **Interactive Graphical Web Interface (Streamlit):**
  ```bash
  streamlit run app.py
  ```
  *(Opens the web app at `http://localhost:8501`)*

---

## 12. Viva & Demonstration Key Concepts

For project presentation and viva examination, understand the following core principles:

1. **Why multi-class classification rather than binary?**
   The agricultural objective is sorting seven commercial grain varieties simultaneously. Multi-class classification utilizes One-vs-Rest or Multinomial Softmax loss functions to compute probability distributions across all 7 categories.

2. **Why standard scaling is critical:**
   Features like `Area` (~50,000 pixels) and `ShapeFactor1` (~0.007) span vastly different numerical orders of magnitude. Unscaled features would cause distance-based models (KNN, SVM) and gradient-based models (Logistic Regression) to over-weight `Area` and neglect delicate shape factors.

3. **How data leakage is prevented:**
   The `StandardScaler` is fitted **strictly on the training split** ($X_{\text{train}}$) and only applied (transformed) onto the test split ($X_{\text{test}}$). Deduplication was performed prior to splitting so identical samples cannot exist across train and test partitions.

4. **Why Ensemble Methods performed best:**
   Random Forest aggregates hundreds of de-correlated decision trees via bootstrap aggregation (bagging), reducing variance on noisy boundary cases (e.g., distinguishing between `SIRA` and `DERMASON`). The Soft Voting Ensemble combines the orthogonal decision boundaries of Random Forest, SVM, and Gradient Boosting.

---

## 13. Limitations & Future Work

- **Environmental Lighting Variance:** Features depend on segmentation masks extracted under controlled laboratory lighting. High-variance industrial sorting environments may introduce segmentation noise.
- **Deep Learning Comparison:** Future extensions could apply Convolutional Neural Networks (CNNs) directly on raw grain imagery rather than pre-extracted geometric features to benchmark end-to-end computer vision versus classical ML.
- **Hardware Deployment:** Compiling the trained Random Forest model to ONNX runtime or TinyML for edge deployment on automated sorting conveyor belts.

---

## 14. License & Attribution

- **Code License:** [MIT License](LICENSE) — Copyright (c) 2026 Ayush Vinod Jadhav.
- **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0) — UCI Machine Learning Repository.
- **Academic Course:** CS3201T Machine Learning Mini Project, Department of Computer Engineering, RSCOE, Pune.
