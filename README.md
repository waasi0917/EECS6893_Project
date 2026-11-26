# Medicare Fraud Detection at Scale

A big-data pipeline that flags potentially fraudulent Medicare billing in the **CMS Medicare Physician & Other Practitioners 2022** dataset (about 10 million provider–service records). Built with **PySpark on Google Cloud** and orchestrated end to end with **Apache Airflow**.

*Final project for EECS 6893: Big Data Analytics, Columbia University (Fall 2024).*

---

## Problem

Medicare fraud costs U.S. taxpayers billions each year, but the claims data has **no fraud labels**. This project builds an unsupervised-to-supervised pipeline: statistical anomaly detection first generates fraud labels, then classifiers learn to predict them from provider and billing features.

## Pipeline

```
 CMS CSV (GCS)
     │
     ▼
 ┌──────────────┐   ┌──────────┐   ┌───────────────────┐   ┌──────────────┐
 │ Data prep    │──▶│   EDA    │──▶│ Outlier detection │──▶│ Feature      │
 │ clean, impute│   │ profiling│   │ & fraud labelling │   │ encoding     │
 └──────────────┘   └──────────┘   └───────────────────┘   └──────┬───────┘
                                                                  │
                                         ┌────────────────────────┼────────────────────────┐
                                         ▼                                                 ▼
                                 ┌────────────────┐                               ┌────────────────┐
                                 │ Logistic Reg.  │                               │ Random Forest  │
                                 └───────┬────────┘                               └───────┬────────┘
                                         └──────────────────┬─────────────────────────────┘
                                                            ▼
                                                   ┌────────────────┐
                                                   │ Metrics report │
                                                   └────────────────┘
```

Each stage is a PySpark job that reads from and writes Parquet to a GCS bucket. The Airflow DAG (`fraud_detection_ml_airflow`) chains them with retries.

### 1. Data preparation (`data_prep.py`)
- Renames cryptic CMS columns (e.g. `Avg_Sbmtd_Chrg` → `Average_Submitted_Charge`)
- Drops records missing critical keys (provider NPI, HCPCS procedure code)
- Imputes missing numeric values with the **approximate median** (`approxQuantile`), which is robust to outliers and scales to 10M rows
- Normalizes categorical text and engineers a `Valid_Charge_Ratio` (submitted charge ÷ Medicare-allowed amount)

### 2. Exploratory analysis (`data_eda.py`)
Schema and missing-value checks, distribution summaries, average charges by provider type, provider counts by state, and high-charge filtering.

### 3. Anomaly detection and labelling (`data_outlier_detection.py`)
A record is labelled `is_fraud = 1` if either holds:
- It is a **Z-score or IQR outlier** on submitted charge, Medicare-allowed amount or Medicare payment
- Its submitted charge is **more than 10× the Medicare-allowed amount**

### 4. Feature engineering (`data_labelling.py`)
Spark ML pipeline: `StringIndexer` → `OneHotEncoder` for provider credentials, type, state and participation; `VectorAssembler` with the numeric and outlier-flag features; `StandardScaler`.

### 5. Modelling (`train_logistic_regression.py`, `train_random_forest.py`)
Logistic Regression and Random Forest (100 trees, depth 5) on a 70/30 split, evaluated with Spark's `BinaryClassificationEvaluator` (area under ROC). Metrics from both models are consolidated into a single report.

## Repository structure

```
src/
  fraud-detection-scripts/   # Phase 1: standalone PySpark scripts for each stage
  airflow/                   # Phase 2: modular stage functions + Airflow DAG
  visualisation-scripts/     # BigQuery queries for dashboards
dev/                         # development notebook
docs/Approach and code.docx  # project write-up
```

## Running it

Requirements: Python 3.8+, PySpark, Apache Airflow 2.x, a GCP project with a GCS bucket (or Dataproc cluster).

1. Download the 2022 **Medicare Physician & Other Practitioners – by Provider and Service** CSV from data.cms.gov and upload it to your bucket.
2. Replace the `gs://6893_waj/...` paths in `src/airflow/*.py` with your bucket.
3. Copy `src/airflow/` into your Airflow `dags/` folder and trigger `fraud_detection_ml_airflow`.

## Limitations and next steps

- Labels come from rule-based anomaly flags, and some of those flags are also model features. The classifiers therefore partly learn the labelling rules. A natural next step is validating against confirmed fraud cases, such as the **OIG LEIE exclusion list**.
- Add precision/recall at the provider level, since fraud is rare and AUC alone can flatter results.
- Try unsupervised detectors (Isolation Forest, autoencoders) as an alternative to rule-based labels.

## Team and contributions

- **Waasi Jagirdar** ([@waasi1709](https://github.com/waasi1709)): data preparation and cleaning pipeline, EDA, anomaly-detection fraud-labelling logic, and the feature-engineering / ML pipeline (Phase 1 scripts)
- **Gautam V.** (gv2359): restructured the code into modular stages, Airflow DAG orchestration, model training and report consolidation (Phase 2)

## Tech

PySpark · Spark ML · Apache Airflow · Google Cloud Storage · BigQuery · Python
