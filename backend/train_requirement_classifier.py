"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 5: Requirement Classifier Model Retraining (train_requirement_classifier.py)

Retrains the Requirement Sentence Classification model (Sentence -> YES / NO)
using the upgraded 4,000-row dataset ARIP_requirement_classification_FINAL_4000_dataset(1).csv.

Model Architecture:
  TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=2500)
          ↓
  LogisticRegression(C=2.0, max_iter=1000, random_state=42)

Artifacts Saved:
  - backend/model/requirement_classifier/classifier_pipeline.joblib
  - backend/model/requirement_classifier/requirement_classifier_metadata.json
  - backend/model/requirement_classifier/error_analysis_test_set.csv
  - backend/model/requirement_classifier/classifier_pipeline_backup.joblib
"""

import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, Tuple, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Environment & Directory Setup
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Primary Dataset Path
DATASET_PATH: str = os.path.join(BASE_DIR, "ARIP_requirement_classification_FINAL_4000_dataset(1).csv")
if not os.path.exists(DATASET_PATH):
    # Fallback search path in data directory
    DATASET_PATH = os.path.join(BASE_DIR, "data", "ARIP_requirement_classification_FINAL_4000_dataset(1).csv")

MODEL_DIR: str = os.path.join(BASE_DIR, "model", "requirement_classifier")
MODEL_PATH: str = os.path.join(MODEL_DIR, "classifier_pipeline.joblib")
BACKUP_PATH: str = os.path.join(MODEL_DIR, "classifier_pipeline_backup.joblib")
METADATA_PATH: str = os.path.join(MODEL_DIR, "requirement_classifier_metadata.json")
METRICS_JSON_PATH: str = os.path.join(MODEL_DIR, "metrics.json")
ERROR_ANALYSIS_PATH: str = os.path.join(MODEL_DIR, "error_analysis_test_set.csv")

LABEL_MAP: Dict[str, int] = {"NO": 0, "YES": 1}
ID_TO_LABEL: Dict[int, str] = {0: "NO", 1: "YES"}
RANDOM_SEED: int = 42


def inspect_and_clean_dataset(csv_path: str) -> pd.DataFrame:
    """
    Programmatically inspects and cleans the 4,000-row requirement dataset.
    Performs validation, missing value checks, duplicate sentence checks,
    and label conflict detection.
    """
    print("=" * 70)
    print("1. DATASET INSPECTION & CLEANING")
    print("=" * 70)
    print(f"[+] Loading dataset from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ["Sentence", "Requirement"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    initial_row_count = len(df)
    print(f"[+] Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Column names: {df.columns.tolist()}")

    # Check for missing / null values
    null_counts = df[required_cols].isnull().sum().to_dict()
    print(f"[+] Missing values in required columns: {null_counts}")

    # Clean sentence text
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    # Normalize excessive internal whitespace
    df["Sentence"] = df["Sentence"].str.replace(r'\s+', ' ', regex=True)

    # Filter out empty or whitespace-only sentences
    df = df[df["Sentence"] != ""].copy()
    cleaned_row_count = len(df)
    if initial_row_count != cleaned_row_count:
        print(f"[!] Removed {initial_row_count - cleaned_row_count} empty/whitespace rows.")

    # Normalize target labels
    df["Requirement"] = df["Requirement"].astype(str).str.strip().str.upper()
    valid_labels = set(LABEL_MAP.keys())
    invalid_labels = set(df["Requirement"].unique()) - valid_labels
    if invalid_labels:
        raise ValueError(f"Dataset contains invalid target labels: {invalid_labels}")

    df["label_id"] = df["Requirement"].map(LABEL_MAP)

    # Duplicate & Label Conflict Checks
    exact_duplicates = df.duplicated(subset=["Sentence", "Requirement"]).sum()
    duplicate_sentences = df.duplicated(subset=["Sentence"]).sum()

    grouped_conflicts = df.groupby("Sentence")["Requirement"].nunique()
    label_conflicts = grouped_conflicts[grouped_conflicts > 1]

    print(f"[+] Exact duplicate rows (Sentence + Label): {exact_duplicates}")
    print(f"[+] Duplicate sentence strings: {duplicate_sentences}")
    print(f"[+] Sentence label conflicts (Same sentence, different labels): {len(label_conflicts)}")

    if len(label_conflicts) > 0:
        print("[!] Warning: Found label conflicts for sentences:")
        for sentence_text in label_conflicts.index[:5]:
            conflicting_labels = df[df["Sentence"] == sentence_text]["Requirement"].unique().tolist()
            print(f"    - '{sentence_text[:60]}...' => Labels: {conflicting_labels}")

    # Deduplicate exact duplicate sentences if present, keeping first
    if duplicate_sentences > 0:
        print(f"[+] Deduplicating {duplicate_sentences} duplicate sentence entries...")
        df = df.drop_duplicates(subset=["Sentence"], keep="first").copy()

    # Target distribution
    target_counts = df["Requirement"].value_counts().to_dict()
    print(f"[+] Target label distribution: {target_counts}")
    print(f"[+] Cleaned dataset final row count: {len(df)}")
    print("-" * 70 + "\n")

    return df


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates a stratified 80% Train / 10% Validation / 10% Test split.
    Prevents exact duplicate leakage across splits.
    """
    print("=" * 70)
    print("2. TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    # Stage 1: Split into 80% Train and 20% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label_id"]
    )

    # Stage 2: Split 20% Temp into 50% Val (10% total) and 50% Test (10% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label_id"]
    )

    print(f"[+] Train set size      : {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    - Target YES: {sum(train_df['label_id']==1)}, NO: {sum(train_df['label_id']==0)}")
    print(f"[+] Validation set size : {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    - Target YES: {sum(val_df['label_id']==1)}, NO: {sum(val_df['label_id']==0)}")
    print(f"[+] Test set size       : {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    - Target YES: {sum(test_df['label_id']==1)}, NO: {sum(test_df['label_id']==0)}")

    # Data leakage check between splits
    train_sents = set(train_df["Sentence"])
    val_sents = set(val_df["Sentence"])
    test_sents = set(test_df["Sentence"])

    tv_overlap = len(train_sents.intersection(val_sents))
    tt_overlap = len(train_sents.intersection(test_sents))
    vt_overlap = len(val_sents.intersection(test_sents))

    print(f"[+] Leakage check overlap (Train-Val): {tv_overlap}")
    print(f"[+] Leakage check overlap (Train-Test): {tt_overlap}")
    print(f"[+] Leakage check overlap (Val-Test): {vt_overlap}")
    if tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0:
        print("[+] Verified ZERO duplicate sentence leakage across Train/Val/Test splits.")

    print("-" * 70 + "\n")
    return train_df, val_df, test_df


def train_model(train_df: pd.DataFrame) -> Pipeline:
    """
    Builds and trains TF-IDF + Logistic Regression sklearn Pipeline.
    """
    print("=" * 70)
    print("3. MODEL TRAINING (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 70)

    tfidf_config = {
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.95,
        "max_features": 2500
    }
    print(f"[+] TF-IDF Vectorizer configuration: {tfidf_config}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_config)),
        ("clf", LogisticRegression(
            C=2.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight=None  # 50/50 balanced target dataset
        ))
    ])

    X_train = train_df["Sentence"]
    y_train = train_df["label_id"]

    print("[+] Fitting pipeline on training split...")
    pipeline.fit(X_train, y_train)
    print("[+] Model training completed successfully.")
    print("-" * 70 + "\n")

    return pipeline


def evaluate_set(pipeline: Pipeline, df_subset: pd.DataFrame, set_name: str) -> Dict[str, Any]:
    """
    Evaluates model pipeline on a dataset subset and returns comprehensive metrics.
    """
    X = df_subset["Sentence"]
    y_true = df_subset["label_id"]

    y_pred = pipeline.predict(X)

    acc = float(accuracy_score(y_true, y_pred))
    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    # Per class metrics
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report_str = classification_report(y_true, y_pred, target_names=["NO", "YES"], digits=4)

    return {
        "set_name": set_name,
        "accuracy": round(acc, 4),
        "binary_precision": round(float(prec_bin), 4),
        "binary_recall": round(float(rec_bin), 4),
        "binary_f1": round(float(f1_bin), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "class_NO": {
            "precision": round(float(prec_per[0]), 4),
            "recall": round(float(rec_per[0]), 4),
            "f1_score": round(float(f1_per[0]), 4)
        },
        "class_YES": {
            "precision": round(float(prec_per[1]), 4),
            "recall": round(float(rec_per[1]), 4),
            "f1_score": round(float(f1_per[1]), 4)
        },
        "confusion_matrix": cm,
        "classification_report_str": report_str,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist()
    }


def compare_with_old_model(test_df: pd.DataFrame, new_test_metrics: Dict[str, Any]) -> None:
    """
    Compares the newly trained model with the old classifier model artifact on the SAME test split.
    """
    print("=" * 70)
    print("5. MODEL COMPARISON (OLD MODEL vs NEW MODEL ON TEST SET)")
    print("=" * 70)

    old_model_file = BACKUP_PATH if os.path.exists(BACKUP_PATH) else MODEL_PATH

    if os.path.exists(old_model_file):
        try:
            old_pipeline = joblib.load(old_model_file)
            old_metrics = evaluate_set(old_pipeline, test_df, "Old Model Test Set")

            print(f"Metrics Comparison (Test Set N={len(test_df)}):")
            print(f"{'Metric':<20} | {'Old Classifier':<15} | {'New Retrained Classifier':<25}")
            print("-" * 65)
            print(f"{'Accuracy':<20} | {old_metrics['accuracy']*100:>13.2f}% | {new_test_metrics['accuracy']*100:>23.2f}%")
            print(f"{'Macro F1':<20} | {old_metrics['macro_f1']*100:>13.2f}% | {new_test_metrics['macro_f1']*100:>23.2f}%")
            print(f"{'Weighted F1':<20} | {old_metrics['weighted_f1']*100:>13.2f}% | {new_test_metrics['weighted_f1']*100:>23.2f}%")
            print(f"{'YES Precision':<20} | {old_metrics['class_YES']['precision']*100:>13.2f}% | {new_test_metrics['class_YES']['precision']*100:>23.2f}%")
            print(f"{'YES Recall':<20} | {old_metrics['class_YES']['recall']*100:>13.2f}% | {new_test_metrics['class_YES']['recall']*100:>23.2f}%")
            print(f"{'NO Precision':<20} | {old_metrics['class_NO']['precision']*100:>13.2f}% | {new_test_metrics['class_NO']['precision']*100:>23.2f}%")
            print(f"{'NO Recall':<20} | {old_metrics['class_NO']['recall']*100:>13.2f}% | {new_test_metrics['class_NO']['recall']*100:>23.2f}%")
        except Exception as e:
            print(f"[!] Could not load old model for direct comparison: {e}")
    else:
        print("[!] No pre-existing classifier model found for direct comparison.")

    print("-" * 70 + "\n")


def perform_error_analysis(pipeline: Pipeline, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies misclassified examples on the test set and exports error analysis CSV.
    """
    print("=" * 70)
    print("6. ERROR ANALYSIS & MISCLASSIFIED EXAMPLES")
    print("=" * 70)

    X_test = test_df["Sentence"]
    y_test = test_df["label_id"]

    y_pred = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)

    errors = []
    for idx, (sentence, actual, pred, proba) in enumerate(zip(X_test, y_test, y_pred, probas)):
        if actual != pred:
            conf = float(proba[pred])
            errors.append({
                "test_index": idx,
                "sentence": sentence,
                "actual_label": ID_TO_LABEL[actual],
                "predicted_label": ID_TO_LABEL[pred],
                "confidence": round(conf * 100, 2),
                "prob_NO": round(float(proba[0]), 4),
                "prob_YES": round(float(proba[1]), 4)
            })

    error_df = pd.DataFrame(errors)
    print(f"[+] Total misclassified test samples: {len(error_df)} / {len(test_df)} ({len(error_df)/len(test_df)*100:.2f}%)")

    if len(error_df) > 0:
        print("\n--- SAMPLE MISCLASSIFIED EXAMPLES ---")
        for i, row in error_df.head(10).iterrows():
            print(f"[{i+1}] Actual: {row['actual_label']} | Predicted: {row['predicted_label']} | Conf: {row['confidence']}%")
            print(f"    Sentence: \"{row['sentence']}\"\n")

        error_df.to_csv(ERROR_ANALYSIS_PATH, index=False, encoding="utf-8")
        print(f"[+] Saved misclassifications analysis report to: {ERROR_ANALYSIS_PATH}")

    print("-" * 70 + "\n")
    return error_df


def test_domain_generalization(pipeline: Pipeline) -> None:
    """
    Tests model on diverse domain requirements and tricky non-requirements.
    """
    print("=" * 70)
    print("7. DOMAIN GENERALIZATION & SAMPLE TEST CASES")
    print("=" * 70)

    test_samples = [
        # CLEAR REQUIREMENTS (Diverse Domains)
        ("The system shall allow customers to reset their password.", "YES", "Security / Auth"),
        ("The application shall store transaction records.", "YES", "Finance / Banking"),
        ("The system shall generate a monthly attendance report.", "YES", "Education / HR"),
        ("The hospital system shall allow doctors to view patient records.", "YES", "Healthcare"),
        ("The e-commerce engine shall calculate sales tax automatically at checkout.", "YES", "E-Commerce"),
        ("Station managers shall be able to update platform assignments for delayed trains.", "YES", "Transportation"),
        ("The library catalog service shall reserve books for registered members.", "YES", "Library Management"),

        # NON-REQUIREMENTS (Document Noise, Section Titles, Discussion)
        ("The project team will discuss the architecture next week.", "NO", "Project Discussion"),
        ("This document describes the purpose of the system.", "NO", "Document Scope"),
        ("Future work may include integration with external platforms.", "NO", "Future Work"),
        ("Glossary: authentication means the process of verifying identity.", "NO", "Glossary / Definition"),
        ("Figure 4.4 illustrates the architectural diagram of network topology.", "NO", "Figure Reference"),
        ("2.1 Product Perspective and Context", "NO", "Heading / Section Title"),
        ("Appendix S: citation material for hardware requirements.", "NO", "Appendix Citation")
    ]

    print(f"{'Text Snippet':<65} | {'Expected':<8} | {'Pred':<8} | {'Confidence':<10}")
    print("-" * 100)

    correct_count = 0
    for text, expected, category in test_samples:
        pred_id = int(pipeline.predict([text])[0])
        probas = pipeline.predict_proba([text])[0]
        pred_label = ID_TO_LABEL[pred_id]
        conf = float(probas[pred_id]) * 100

        is_correct = (pred_label == expected)
        if is_correct:
            correct_count += 1

        mark = "[OK]  " if is_correct else "[FAIL]"
        print(f"{mark} {text[:60]:<61} | {expected:<8} | {pred_label:<8} | {conf:>6.1f}%")

    print("-" * 100)
    print(f"[+] Sample Test Accuracy: {correct_count} / {len(test_samples)} ({correct_count/len(test_samples)*100:.1f}%)")
    print("-" * 70 + "\n")


def backup_and_save_artifacts(
    pipeline: Pipeline,
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any]
) -> None:
    """
    Backs up existing model artifact and saves newly trained joblib pipeline & JSON metadata.
    """
    print("=" * 70)
    print("8. ARTIFACT SAVING & METADATA GENERATION")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Backup existing artifact if present
    if os.path.exists(MODEL_PATH):
        try:
            shutil.copy(MODEL_PATH, BACKUP_PATH)
            print(f"[+] Backed up existing model artifact to: {BACKUP_PATH}")
        except Exception as e:
            print(f"[!] Warning: Could not create model backup: {e}")

    # Save joblib pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[+] Saved retrained requirement classifier pipeline to: {MODEL_PATH}")

    # Save comprehensive JSON metadata
    metadata = {
        "model_name": "TF-IDF + Logistic Regression Requirement Classifier",
        "dataset_name": "ARIP_requirement_classification_FINAL_4000_dataset(1).csv",
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": RANDOM_SEED,
        "input_feature": "Sentence",
        "target_label": "Requirement",
        "classes": ["NO", "YES"],
        "dataset_size": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        },
        "target_distribution": df["Requirement"].value_counts().to_dict(),
        "tfidf_configuration": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 2500
        },
        "classifier_configuration": {
            "algorithm": "LogisticRegression",
            "C": 2.0,
            "max_iter": 1000,
            "class_weight": "None"
        },
        "training_metrics": {
            "accuracy": train_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"],
            "weighted_f1": train_metrics["weighted_f1"]
        },
        "validation_metrics": {
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "confusion_matrix": val_metrics["confusion_matrix"]
        },
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "binary_precision": test_metrics["binary_precision"],
            "binary_recall": test_metrics["binary_recall"],
            "binary_f1": test_metrics["binary_f1"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "class_NO": test_metrics["class_NO"],
            "class_YES": test_metrics["class_YES"],
            "confusion_matrix": test_metrics["confusion_matrix"]
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved metadata report to: {METADATA_PATH}")

    # Also update backward-compatible metrics.json
    metrics_legacy = {
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["binary_precision"],
        "recall": test_metrics["binary_recall"],
        "f1_score": test_metrics["binary_f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "model_type": "TF-IDF + Logistic Regression (Requirement Classifier 4000)"
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_legacy, f, indent=4)
    print(f"[+] Updated legacy metrics file: {METRICS_JSON_PATH}")

    print("-" * 70 + "\n")


def main() -> None:
    """
    Main training execution workflow.
    """
    print("\n" + "=" * 70)
    print("  ARIP REQUIREMENT SENTENCE CLASSIFIER MODEL RETRAINING (4,000 DATASET)")
    print("=" * 70 + "\n")

    # Step 1: Inspect and clean dataset
    df = inspect_and_clean_dataset(DATASET_PATH)

    # Step 2: Stratified split into Train (80%), Val (10%), Test (10%)
    train_df, val_df, test_df = split_dataset(df)

    # Step 3: Train model pipeline
    pipeline = train_model(train_df)

    # Step 4: Validation and Evaluation
    print("=" * 70)
    print("4. MODEL EVALUATION & METRICS REPORT")
    print("=" * 70)

    train_metrics = evaluate_set(pipeline, train_df, "Training Set")
    val_metrics = evaluate_set(pipeline, val_df, "Validation Set")
    test_metrics = evaluate_set(pipeline, test_df, "Test Set")

    print(f"[+] Training Set Accuracy   : {train_metrics['accuracy'] * 100:.2f}% | Macro F1: {train_metrics['macro_f1'] * 100:.2f}%")
    print(f"[+] Validation Set Accuracy : {val_metrics['accuracy'] * 100:.2f}% | Macro F1: {val_metrics['macro_f1'] * 100:.2f}%")
    print(f"[+] Final Test Set Accuracy : {test_metrics['accuracy'] * 100:.2f}% | Macro F1: {test_metrics['macro_f1'] * 100:.2f}%\n")

    print("--- FINAL TEST SET CLASSIFICATION REPORT ---")
    print(test_metrics["classification_report_str"])

    print("--- FINAL TEST SET CONFUSION MATRIX ---")
    print("                   Pred NO        Pred YES")
    print(f" True NO        :     {test_metrics['confusion_matrix'][0][0]:<10} {test_metrics['confusion_matrix'][0][1]:<10}")
    print(f" True YES       :     {test_metrics['confusion_matrix'][1][0]:<10} {test_metrics['confusion_matrix'][1][1]:<10}")
    print("-" * 70 + "\n")

    # Step 5: Model comparison with old model
    compare_with_old_model(test_df, test_metrics)

    # Step 6: Error Analysis on test split
    perform_error_analysis(pipeline, test_df)

    # Step 7: Domain generalization test
    test_domain_generalization(pipeline)

    # Step 8: Backup and save artifacts
    backup_and_save_artifacts(
        pipeline, df, train_df, val_df, test_df,
        train_metrics, val_metrics, test_metrics
    )

    print("=" * 70)
    print("   REQUIREMENT CLASSIFIER MODEL RETRAINING COMPLETE (SUCCESS)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
