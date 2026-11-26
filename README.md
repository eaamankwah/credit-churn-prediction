# Credit Church Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Pylint](https://img.shields.io/badge/Pylint-10.00%2F10-brightgreen?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-Passing-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A production-ready machine learning pipeline that identifies credit card customers most likely to churn — engineered to professional software standards.**

[Overview](#-overview) · [Quick Start](#-quick-start) · [Architecture](#-architecture) · [Usage](#-usage) · [Testing](#-testing--logging) · [Results](#-results) · [API Reference](#-api-reference)

</div>

---

##  Overview

Customer churn is one of the most costly and preventable challenges in banking. This project transforms an exploratory Jupyter notebook into a **modular, testable, CLI-executable Python package** that predicts which credit card customers are most likely to leave — enabling data-driven, proactive retention campaigns.

###  Key Highlights

| Capability | Detail |
|---|---|
|  **Prediction accuracy** | Random Forest AUC ≈ 0.99 · F1 ≈ 0.93 on churn class |
|  **Code quality** | Pylint **10.00 / 10** on both Python files |
|  **Test coverage** | Unit tests for every public function with assertions |
|  **Logging** | Structured, timestamped log output written to file |
|  **Reproducible** | Fixed random seed · CLI-executable · fully documented |
|  **Production-ready** | Persisted `.pkl` models loadable in any inference pipeline |

---
A full analysis and discussions of the project: [REPORT](doc/Churn_Prediction_Report.pdf)
---

##  Project Structure

```
credict-churn-prediction/
├── churn_library.py                    #  Core ML pipeline library (7 public functions)
├── churn_script_logging_and_tests.py   #  Unit tests with integrated file logging
├── README.md                           #  This file
├── requirements.txt                    #  Pinned Python dependencies
│
├── data/
│   └── bank_data.csv                   #  Source dataset — 10,127 customers × 23 features
├── doc/
│   └── Churn_Prediction_Report.pdf     #  Project Report
│
├── images/
│   ├── eda/                            #   EDA visualisations (auto-generated)
│   │   ├── churn_distribution.png      #     Univariate — target class balance
│   │   ├── customer_age.png            #     Univariate — age histogram
│   │   ├── marital_status.png          #     Univariate — categorical bar chart
│   │   ├── total_trans_ct.png          #     Univariate — KDE transaction count
│   │   └── heatmap.png                 #     Bivariate — feature correlation heatmap
│   │
│   └── results/                        #  Model evaluation plots (auto-generated)
│       ├── roc_curve_result.png        #     ROC curves — RF vs LR comparison
│       ├── feature_importances.png     #     Ranked feature importance bar chart
│       ├── nn_confusion_matrices.png   #     Confusion matrix
│       ├── nn_metrics_comparison.png   #     Comparing matrices
│       ├── nn_radar_chart.png          #     Radar chart
│       ├── nn_roc_comparison.png       #     Comparing rocs
│       ├── nn_training_curves.png      #     Training  curves
│       ├── rf_results.png              #     Random Forest classification report
│       └── logistic_results.png        #     Logistic Regression classification report
│
├── logs/
│   └── churn_library.log               #  Timestamped test & pipeline log (auto-generated)
│
└── models/
    ├── rfc_model.pkl                   #  Best Random Forest (GridSearchCV tuned)
    └── logistic_model.pkl              #  Logistic Regression (StandardScaler pipeline)
```

---

## ⚡ Quick Start

### 1 — Clone and install dependencies

```bash
git clone https://github.com/your-org/credit-churn-prediction.git
cd credit-churn-prediction
pip install -r requirements.txt
```

### 2 — Run the full ML pipeline

```bash
python churn_library.py
```

### 3 — Run tests and view the log

```bash
python churn_script_logging_and_tests.py
cat logs/churn_library.log
```

> **That's it.** All EDA plots, model artefacts, result images, and log files are created automatically in the correct directories.

---

##  Architecture

### Pipeline Sequence

The pipeline executes the following stages in strict order when `churn_library.py` is run as a script. Each stage is an independent, testable function — outputs are passed explicitly, with no shared global state.

```
import_data ──► perform_eda ──► encoder_helper ──► perform_feature_engineering ──► train_models
```

---

###  Data Science Flow Diagram

```mermaid
flowchart TD

    %% ═══════════════════════════════════════════════════════════
    %%  NODE DEFINITIONS
    %% ═══════════════════════════════════════════════════════════

    %% ── GROUP 1 · DATA INGESTION (deep navy) ───────────────────
    A1([📂 bank_data.csv\n10,127 rows × 23 cols])
    A2[/import_data pth\nRead CSV · Create Churn column/]
    A3[(Raw DataFrame\n+ Binary Target)]

    %% ── GROUP 2 · EDA (emerald green) ──────────────────────────
    B1[/perform_eda df\nGenerate 5 diagnostic plots/]
    B2([🖼 churn_distribution.png\nUnivariate · Quantitative])
    B3([🖼 customer_age.png\nUnivariate · Quantitative])
    B4([🖼 marital_status.png\nUnivariate · Categorical])
    B5([🖼 total_trans_ct.png\nKDE · Transaction Count])
    B6([🖼 heatmap.png\nBivariate · Correlation])

    %% ── GROUP 3 · FEATURE ENGINEERING (amber) ──────────────────
    C1[/encoder_helper df cat_lst response\nMean-encode 5 categorical columns/]
    C2[/perform_feature_engineering df response\nSelect 19 features · 70/30 split/]
    C3[(X_train · X_test\ny_train · y_test)]

    %% ── GROUP 4 · MODEL TRAINING (violet) ──────────────────────
    D1[/train_models\nOrchestrate full training cycle/]
    D2[[🌲 Random Forest\nGridSearchCV 5-fold CV\nn_estimators 200 500\nmax_depth 4 5 100\ncriterion gini entropy]]
    D3[[📈 Logistic Regression\nPipeline StandardScaler LR\nsolver lbfgs · max_iter 3000]]

    %% ── GROUP 5 · PREDICTION (rose) ────────────────────────────
    E1{Predict\nTrain + Test}
    E2([y_train_preds_rf\ny_test_preds_rf])
    E3([y_train_preds_lr\ny_test_preds_lr])

    %% ── GROUP 6 · EVALUATION (sky blue) ────────────────────────
    F1[/classification_report_image\nPrecision · Recall · F1 · Support/]
    F2[/feature_importance_plot\nRanked impurity-based importances/]
    F3([🖼 rf_results.png\nRF Classification Report])
    F4([🖼 logistic_results.png\nLR Classification Report])
    F5([🖼 roc_curve_result.png\nROC Curves RF vs LR])
    F6([🖼 feature_importances.png\nTop Predictors Ranked])

    %% ── GROUP 7 · PERSISTENCE (slate) ──────────────────────────
    G1[(🤖 rfc_model.pkl\nBest RF Estimator)]
    G2[(🤖 logistic_model.pkl\nLR Pipeline)]

    %% ── GROUP 8 · TESTING & LOGGING (orange) ───────────────────
    H1([⚙️ churn_script_logging_and_tests.py\npython · ipython · pytest])
    H2[/test_import\nShape · Churn column/]
    H3[/test_eda\n5 PNG files exist/]
    H4[/test_encoder_helper\nFloat cols · no NaN/]
    H5[/test_feature_engineering\nSplit ratio · KEEP_COLS/]
    H6[/test_train_models\nPKL files · result PNGs/]
    H7[(📝 logs/churn_library.log\nTimestamped INFO + ERROR)]

    %% ═══════════════════════════════════════════════════════════
    %%  EDGES
    %% ═══════════════════════════════════════════════════════════

    A1 --> A2 --> A3
    A3 --> B1
    B1 --> B2 & B3 & B4 & B5 & B6

    A3 --> C1 --> C2 --> C3

    C3 --> D1
    D1 --> D2 & D3
    D2 & D3 --> E1
    E1 --> E2 & E3

    E2 & E3 --> F1
    D2 --> F2
    F1 --> F3 & F4
    D2 & D3 -.->|RocCurveDisplay| F5
    F2 --> F6

    D2 -->|joblib.dump| G1
    D3 -->|joblib.dump| G2

    H1 --> H2 & H3 & H4 & H5 & H6
    H2 & H3 & H4 & H5 & H6 --> H7

    %% ═══════════════════════════════════════════════════════════
    %%  STYLES — each group gets its own distinct color palette
    %% ═══════════════════════════════════════════════════════════

    %% GROUP 1 · Data Ingestion — deep navy blue
    style A1 fill:#1e3a5f,stroke:#0d2137,color:#ffffff
    style A2 fill:#2c5282,stroke:#1e3a5f,color:#ffffff
    style A3 fill:#2b4c7e,stroke:#1e3a5f,color:#ffffff

    %% GROUP 2 · EDA — emerald / teal green
    style B1 fill:#065f46,stroke:#022c22,color:#ffffff
    style B2 fill:#059669,stroke:#065f46,color:#ffffff
    style B3 fill:#059669,stroke:#065f46,color:#ffffff
    style B4 fill:#059669,stroke:#065f46,color:#ffffff
    style B5 fill:#059669,stroke:#065f46,color:#ffffff
    style B6 fill:#059669,stroke:#065f46,color:#ffffff

    %% GROUP 3 · Feature Engineering — amber / gold
    style C1 fill:#92400e,stroke:#451a03,color:#ffffff
    style C2 fill:#b45309,stroke:#92400e,color:#ffffff
    style C3 fill:#d97706,stroke:#92400e,color:#000000

    %% GROUP 4 · Model Training — violet / purple
    style D1 fill:#4c1d95,stroke:#2e1065,color:#ffffff
    style D2 fill:#6d28d9,stroke:#4c1d95,color:#ffffff
    style D3 fill:#7c3aed,stroke:#4c1d95,color:#ffffff

    %% GROUP 5 · Prediction — rose / crimson
    style E1 fill:#881337,stroke:#4c0519,color:#ffffff
    style E2 fill:#be123c,stroke:#881337,color:#ffffff
    style E3 fill:#be123c,stroke:#881337,color:#ffffff

    %% GROUP 6 · Evaluation — sky blue / cyan
    style F1 fill:#075985,stroke:#0c4a6e,color:#ffffff
    style F2 fill:#0369a1,stroke:#075985,color:#ffffff
    style F3 fill:#0284c7,stroke:#075985,color:#ffffff
    style F4 fill:#0284c7,stroke:#075985,color:#ffffff
    style F5 fill:#0ea5e9,stroke:#075985,color:#000000
    style F6 fill:#0ea5e9,stroke:#075985,color:#000000

    %% GROUP 7 · Persistence — slate / charcoal
    style G1 fill:#1e293b,stroke:#0f172a,color:#ffffff
    style G2 fill:#334155,stroke:#1e293b,color:#ffffff

    %% GROUP 8 · Testing & Logging — burnt orange
    style H1 fill:#7c2d12,stroke:#431407,color:#ffffff
    style H2 fill:#c2410c,stroke:#7c2d12,color:#ffffff
    style H3 fill:#c2410c,stroke:#7c2d12,color:#ffffff
    style H4 fill:#c2410c,stroke:#7c2d12,color:#ffffff
    style H5 fill:#c2410c,stroke:#7c2d12,color:#ffffff
    style H6 fill:#c2410c,stroke:#7c2d12,color:#ffffff
    style H7 fill:#ea580c,stroke:#7c2d12,color:#ffffff
```

### Color Legend

| Color | Group | Nodes |
|---|---|---|
| 🟦 **Deep Navy** | Data Ingestion | `import_data`, Raw CSV, DataFrame |
| 🟩 **Emerald Green** | EDA | `perform_eda`, all 5 EDA plots |
| 🟨 **Amber / Gold** | Feature Engineering | `encoder_helper`, `perform_feature_engineering`, split data |
| 🟪 **Violet / Purple** | Model Training | `train_models`, Random Forest, Logistic Regression |
| 🟥 **Rose / Crimson** | Prediction | Predict decision node, prediction arrays |
| 🔵 **Sky Blue** | Evaluation | `classification_report_image`, `feature_importance_plot`, result PNGs |
| ⬛ **Slate / Charcoal** | Persistence | `.pkl` model files on disk |
| 🟧 **Burnt Orange** | Testing & Logging | Test script, 5 test functions, log file |

---

### Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn — `RandomForestClassifier`, `LogisticRegression`, `GridSearchCV`, `Pipeline`, `StandardScaler` |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Model Persistence | joblib (`.pkl` format) |
| Testing | pytest + custom `__main__` CLI runner |
| Logging | Python `logging` — `FileHandler` |
| Code Quality | pylint (10.00/10), autopep8 |
| Style Standard | PEP 8 |

---

##  Dataset

**Source:** `data/bank_data.csv` — anonymised credit card customer data

| Attribute | Value |
|---|---|
| Records | 10,127 customers |
| Raw features | 23 columns |
| Model input features | 19 (engineered) |
| Target variable | `Churn` — `0` = Retained · `1` = Churned |
| Class distribution | 83.9% Retained · 16.1% Churned |
| Missing values | None |
| Train / Test split | 70% / 30% (`random_state=42`) |

### Categorical Features — Mean-Encoded

| Feature | Description |
|---|---|
| `Gender` | Customer gender |
| `Education_Level` | Highest education attained |
| `Marital_Status` | Married / Single / Divorced / Unknown |
| `Income_Category` | Annual income bracket |
| `Card_Category` | Card tier — Blue / Silver / Gold / Platinum |

### Quantitative Features — Used As-Is

| Feature | Description |
|---|---|
| `Customer_Age` | Age in years |
| `Dependent_count` | Number of dependents |
| `Months_on_book` | Months as a customer |
| `Total_Relationship_Count` | Number of products held |
| `Months_Inactive_12_mon` | Months inactive in past 12 months |
| `Contacts_Count_12_mon` | Customer service contacts (past 12 months) |
| `Credit_Limit` | Credit card limit ($) |
| `Total_Revolving_Bal` | Revolving balance ($) |
| `Avg_Open_To_Buy` | Average open-to-buy credit ($) |
| `Total_Amt_Chng_Q4_Q1` | Change in transaction amount Q4 vs Q1 |
| `Total_Trans_Amt` | Total transaction amount — past 12 months |
| `Total_Trans_Ct` | Total transaction count — past 12 months ⭐ |
| `Total_Ct_Chng_Q4_Q1` | Change in transaction count Q4 vs Q1 |
| `Avg_Utilization_Ratio` | Average card utilisation ratio |

>  `Total_Trans_Ct` is the **single most important churn predictor** per Random Forest feature importances.

---

##  Usage

### Run the full ML pipeline

```bash
python churn_library.py
```

**Stages executed, in order:**

1. Load `./data/bank_data.csv` → create binary `Churn` target from `Attrition_Flag`
2. Generate 5 EDA plots → save to `./images/eda/`
3. Mean-encode 5 categorical columns via `encoder_helper`
4. Select 19 features · perform 70/30 train-test split
5. Train **Random Forest** with 5-fold `GridSearchCV` over `PARAM_GRID`
6. Train **Logistic Regression** via `StandardScaler → LR` scikit-learn Pipeline
7. Generate ROC curves, classification reports, feature importances → `./images/results/`
8. Persist both models → `./models/*.pkl`

---

### Run tests and generate the log file

```bash
# Option 1 — direct CLI runner (writes to logs/churn_library.log)
python churn_script_logging_and_tests.py

# Option 2 — pytest for richer terminal output
pytest churn_script_logging_and_tests.py -v

# Option 3 — ipython
ipython churn_script_logging_and_tests.py
```

**View the log after running:**

```bash
cat logs/churn_library.log
```

**Expected log output:**

```
2026-05-16 12:00:01 - churn_tests - INFO - ===== Starting churn library tests =====
2026-05-16 12:00:01 - root        - INFO - test_import: SUCCESS – data loaded
2026-05-16 12:00:01 - root        - INFO - test_import: dataframe shape (10127, 23), 'Churn' column present
2026-05-16 12:00:03 - root        - INFO - test_eda: perform_eda ran without exception
2026-05-16 12:00:03 - root        - INFO - test_eda: SUCCESS – all 5 EDA plots saved
2026-05-16 12:00:03 - root        - INFO - test_encoder_helper: SUCCESS – 5 encoded columns created
2026-05-16 12:00:03 - root        - INFO - test_perform_feature_engineering: SUCCESS – train (7088, 19), test (3039, 19)
2026-05-16 12:02:45 - root        - INFO - test_train_models: train_models ran without exception
2026-05-16 12:02:45 - root        - INFO - test_train_models: SUCCESS – 2 model files saved
2026-05-16 12:02:45 - root        - INFO - test_train_models: SUCCESS – 4 result images saved
2026-05-16 12:02:45 - churn_tests - INFO - test_train_models passed
2026-05-16 12:02:45 - churn_tests - INFO - ===== All tests passed =====
```

---

### Code quality checks

```bash
# Auto-format with autopep8
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py

# Lint — both files achieve 10.00/10
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

---

### Load a saved model for inference

```python
import joblib
import pandas as pd

# Load the persisted Random Forest
rfc_model = joblib.load('./models/rfc_model.pkl')

# Score new customers — must have the same 19 feature columns as KEEP_COLS
X_new = pd.DataFrame([...])
churn_proba = rfc_model.predict_proba(X_new)[:, 1]   # probability of churn
churn_pred  = rfc_model.predict(X_new)                # binary prediction
```

---

## Testing & Logging

### Test suite

| Test function | Function under test | Key assertions |
|---|---|---|
| `test_import` | `import_data` | File loads · shape > 0 · `Churn` column created |
| `test_eda` | `perform_eda` | All 5 EDA PNGs exist in `images/eda/` |
| `test_encoder_helper` | `encoder_helper` | 5 new `*_Churn` float columns · zero NaNs |
| `test_perform_feature_engineering` | `perform_feature_engineering` | 4 non-empty splits · ≈70/30 ratio · all 19 `KEEP_COLS` present |
| `test_train_models` | `train_models` | 2 `.pkl` files in `models/` · 4 result PNGs in `images/results/` |

### Logging design

- **Handler:** `FileHandler` attached to the root logger **before** `churn_library` is imported — ensures all library-level log messages are captured
- **Format:** `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Level:** `INFO` and above
- **Location:** `./logs/churn_library.log` (overwritten each run with `filemode='w'`)
- Both `INFO` (success paths) and `ERROR` (caught exceptions with tracebacks) are logged

---

## Results

### Model Performance Summary

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| AUC (ROC) | ~0.99 | ~0.92 |
| F1 — Churn class (test) | ~0.93 | ~0.74 |
| Recall — Churn class (test) | ~0.92 | ~0.67 |
| Precision — Churn class (test) | ~0.95 | ~0.83 |
| Macro-avg F1 (test) | ~0.96 | ~0.87 |

### Top 5 Churn Predictors (Random Forest)

| Rank | Feature | Business Interpretation |
|---|---|---|
| 1 | `Total_Trans_Ct` | Declining transaction count is the clearest disengagement signal |
| 2 | `Total_Trans_Amt` | Reduced spending precedes account closure |
| 3 | `Total_Revolving_Bal` | Paying down balance signals intent to close |
| 4 | `Total_Ct_Chng_Q4_Q1` | Quarter-over-quarter drop flags accelerating disengagement |
| 5 | `Avg_Utilization_Ratio` | Low utilisation indicates the customer is stepping back |

---

## API Reference

### `churn_library.py` — Public Functions

---

#### `import_data(pth)`

Reads the CSV at `pth` and creates a binary `Churn` column from `Attrition_Flag`.

```python
df = import_data('./data/bank_data.csv')
```

| Param | Type | Description |
|---|---|---|
| `pth` | `str` | Path to the source CSV file |

**Returns:** `pd.DataFrame` with `Churn` column added (`0` = Retained, `1` = Churned).

---

#### `perform_eda(df)`

Generates and saves 5 diagnostic plots to `./images/eda/`.

```python
perform_eda(df)
```

| Param | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Raw dataframe from `import_data` |

**Returns:** `None`. Side effect: saves 5 PNG files to `images/eda/`.

---

#### `encoder_helper(df, category_lst, response)`

Applies mean target encoding to each column in `category_lst`, appending a new `<col>_Churn` column for each.

```python
df = encoder_helper(df, CAT_COLUMNS, response='Churn')
```

| Param | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Input dataframe |
| `category_lst` | `list[str]` | Column names to encode |
| `response` | `str` | Target column name |

**Returns:** `pd.DataFrame` with new mean-encoded columns appended.

---

#### `perform_feature_engineering(df, response)`

Encodes categorical features, selects the 19-column `KEEP_COLS` feature set, and performs a 70/30 train-test split.

```python
X_train, X_test, y_train, y_test = perform_feature_engineering(df, response='Churn')
```

| Param | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Input dataframe |
| `response` | `str` | Target column name |

**Returns:** `tuple` — `(X_train, X_test, y_train, y_test)`.

---

#### `train_models(X_train, X_test, y_train, y_test)`

Trains both models, generates all evaluation plots, and persists `.pkl` artefacts.

```python
train_models(X_train, X_test, y_train, y_test)
```

| Param | Type | Description |
|---|---|---|
| `X_train` | `pd.DataFrame` | Training features |
| `X_test` | `pd.DataFrame` | Test features |
| `y_train` | `pd.Series` | Training labels |
| `y_test` | `pd.Series` | Test labels |

**Returns:** `None`. Side effects: saves models to `models/` and plots to `images/results/`.

---

#### `classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)`

Saves precision/recall/F1 classification reports for both models as PNG images.

**Returns:** `None`. Saves `rf_results.png` and `logistic_results.png` to `images/results/`.

---

#### `feature_importance_plot(model, x_data, output_pth)`

Saves a ranked feature importance bar chart derived from `model.feature_importances_`.

```python
feature_importance_plot(best_rfc, X_all, './images/results/feature_importances.png')
```

| Param | Type | Description |
|---|---|---|
| `model` | estimator | Fitted model with `feature_importances_` attribute |
| `x_data` | `pd.DataFrame` | Feature dataframe (column names used as x-axis labels) |
| `output_pth` | `str` | Full file path for the output PNG |

**Returns:** `None`.

---

## Constants Reference

All constants are defined at module level in `churn_library.py` — modify here to adapt the pipeline to new data.

```python
CAT_COLUMNS = [
    'Gender', 'Education_Level', 'Marital_Status',
    'Income_Category', 'Card_Category',
]

KEEP_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn',
]

EDA_PATH     = './images/eda'
RESULTS_PATH = './images/results'
MODELS_PATH  = './models'

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt'],
    'max_depth'   : [4, 5, 100],
    'criterion'   : ['gini', 'entropy'],
}
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

```
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
matplotlib>=3.4
seaborn>=0.11
joblib>=1.1
pytest>=7.0
pylint>=2.12
autopep8>=1.6
```

**System requirements:** Python 3.10+ · No GPU required · ~500 MB disk space

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: bank_data.csv` | Wrong working directory | `cd` to project root before running |
| `ModuleNotFoundError` | Missing packages | `pip install -r requirements.txt` |
| Log file empty or not created | `basicConfig` called before `FileHandler` set | Run `churn_script_logging_and_tests.py` directly, not as an import |
| EDA plots missing before tests | `perform_eda` not yet called | Run `python churn_library.py` first |
| Pylint score below 10 | Formatting issues | Run `autopep8 --in-place --aggressive --aggressive <file>.py` |
| `ConvergenceWarning` from LR | Feature scale mismatch | Already fixed — `StandardScaler` is applied in the Pipeline |

---

## Author

**Edward Amankwah**  
Project: ML DevOps Engineer Nanodegree with Clean Code Principles
Date: May 2026

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

** If this project helped you, please give it a star!**

Made with  Python ·  scikit-learn ·  matplotlib ·  pytest

[Report link](doc/Churn_Prediction_Report.pdf)
</div>
