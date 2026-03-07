# ASD (Autism Spectrum Disorder) Predictive Modeling with Explainable AI (XAI)

This project builds upon the theoretical framework and Exploratory Data Analysis (EDA) to develop predictive models for the dataset. The objective is to gain a deeper understanding of algorithm selection, perform rigorous model evaluation, and apply Explainable AI techniques to interpret model predictions.
---

## Project Structure

```
asd-poject_asd assignment2/
├── data/                         # Dataset (not tracked because raw data was not uploaded)
│   └── asd.csv
├── src/
│   └── asd_modeling.py           # Main modeling script
├── notebooks/
│   └── asd_modeling.ipynb        # Colab/Jupyter
├── outputs/                      # Generated plots and figures
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Overview

Building upon our theoretical framework and Exploratory Data Analysis (EDA), our group
will now train predictive models. The goal of this phase is to build a deep intuition of algorithm
selection , evaluate models rigorously against class imbalance using PR-AUC , and use
Explainable AI (SHAP) to translate black-box predictions into actionable business or theoretical
insights.
The pipeline covers:

- **Data Preprocessing** — missing value imputation, label encoding, feature scaling
- **Model Training** — two models trained with class-imbalance handling
- **Advanced Evaluation** — PR-AUC, Confusion Matrix, F1-Score
- **Explainability** — SHAP global and local explanations

---

## Models

| Model | Imbalance Strategy |
|---|---|
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` (neg/pos ratio) |

---

## Evaluation Metrics

**Primary metric: PR-AUC** (Precision-Recall Area Under Curve)

Standard accuracy is misleading for imbalanced datasets yet PR-AUC focuses
specifically on the minority (ASD=Yes) class, making it the right metric
for clinical screening tasks where missed diagnoses carry high real-world cost.

---

## Explainability (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain both global model
behavior and individual predictions:

- **Beeswarm Plot** — feature importance with direction of influence
- **Bar Plot** — mean absolute SHAP value ranking
- **Waterfall Plot (ASD)** — why the model flagged a specific child
- **Waterfall Plot (No ASD)** — what protected a child from being flagged
- **Dependence Plot** — relationship between top feature and its SHAP contribution

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/asd-project_asd_assignment2.git
cd asd-prediction-xai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place `asd.csv` inside the `data/` folder.

### 4. Run the script
```bash
cd src
python asd_modeling.py
```

Output plots will be saved to the `outputs/` folder.

---

## Dataset Note

The dataset (`asd.csv`) is **not tracked** in this repository due to data
privacy considerations. Please contact the project team or refer to the
original data source to obtain the dataset.

---

## Team

| Name | Student ID |
|------|-------|
|ABENAITWE ALLAN | 2025/MSIS/025/PS |
|AKANKUNDA SOPHIA NYANGOMA| 2025/MSIS/002/PS|
| AUGUSTINE BEILEL | 2025/MSIS/008/PS |
| NANTUMBWE DOROTHY | 2025/MSIS/042/PS |
| TIKO JOY | 2025/MSIS/021/PS|
| TURYASINGURA KENNEDY | 2025/MSIS/023/PS|

---

## Course
DATA SCIENCE

**Assignment:** 
Part 1: Algorithm Selection & Training
Part 2: Advanced Evaluation Metrics
Part 3: Model Interpretability (Explainable AI)

**Platform:** Python 3 (VS Code / Google Colab compatible/PyCharm)
