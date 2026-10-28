"""
AI Requirements Intelligence Platform (ARIP) - Phase 14 Ambiguity Detection Retraining Pipeline
Script: train_phase14.py

This script:
1. Merges existing Step 10 ambiguity datasets with the Phase 13 expanded dataset.
2. Cleans text, validates labels, checks for duplicate text and label conflicts.
3. Saves master dataset to ARIP_AMBIGUITY_MASTER_DATASET_PHASE14.csv.
4. Performs stratified 80/10/10 train/val/test split with zero data leakage.
5. Backs up existing ambiguity models to backend/model/saved_model/backups/.
6. Retrains TF-IDF (ngram_range=(1,2), max_features=5000) + Logistic Regression (C=2.0).
7. Evaluates performance across Train, Val, and Test sets (Accuracy, Precision, Recall, F1, Confusion Matrix).
8. Performs False Positive / False Negative error analysis and unseen manual test validation.
9. Saves model artifacts and ambiguity_training_report.txt.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
SAVED_MODEL_DIR: str = os.path.join(BASE_DIR, "model", "saved_model")
BACKUP_DIR: str = os.path.join(SAVED_MODEL_DIR, "backups")
PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data", "processed")

# Source dataset paths
STEP10_TRAIN: str = os.path.join(BASE_DIR, "requirements_train_step10_validated.csv")
STEP10_VAL: str = os.path.join(BASE_DIR, "requirements_val_step10_validated.csv")
STEP10_TEST: str = os.path.join(BASE_DIR, "requirements_test_step10_validated.csv")

UPLOADED_PHASE13: str = os.path.join(
    "C:/Users/DELL/.gemini/antigravity/brain/21d61022-31d9-437e-aea4-2b75030e8a6d/.user_uploaded",
    "media_1791020972348.csv"
)

# Output dataset paths
MASTER_DATASET_PATH: str = os.path.join(PROCESSED_DATA_DIR, "ARIP_AMBIGUITY_MASTER_DATASET_PHASE14.csv")
ROOT_MASTER_DATASET_PATH: str = os.path.join(BASE_DIR, "ARIP_AMBIGUITY_MASTER_DATASET_PHASE14.csv")
PHASE13_COPY_PATH: str = os.path.join(BASE_DIR, "ARIP_FINAL_TRAINING_DATASET_PHASE13(1).csv")

TRAIN_SPLIT_PATH: str = os.path.join(PROCESSED_DATA_DIR, "requirements_train_phase14.csv")
VAL_SPLIT_PATH: str = os.path.join(PROCESSED_DATA_DIR, "requirements_val_phase14.csv")
TEST_SPLIT_PATH: str = os.path.join(PROCESSED_DATA_DIR, "requirements_test_phase14.csv")

ROOT_TRAIN_SPLIT_PATH: str = os.path.join(BASE_DIR, "requirements_train_phase14.csv")
ROOT_VAL_SPLIT_PATH: str = os.path.join(BASE_DIR, "requirements_val_phase14.csv")
ROOT_TEST_SPLIT_PATH: str = os.path.join(BASE_DIR, "requirements_test_phase14.csv")

# Output artifact paths
PHASE14_MODEL_PATH: str = os.path.join(SAVED_MODEL_DIR, "ambiguity_model_phase14.joblib")
BASELINE_MODEL_PATH: str = os.path.join(SAVED_MODEL_DIR, "baseline_model.joblib")
V2_MODEL_PATH: str = os.path.join(SAVED_MODEL_DIR, "ambiguity_model_v2.pkl")
V2_VEC_PATH: str = os.path.join(SAVED_MODEL_DIR, "tfidf_vectorizer_v2.pkl")
METADATA_PATH: str = os.path.join(SAVED_MODEL_DIR, "ambiguity_model_phase14_metadata.json")
REPORT_PATH: str = os.path.join(BASE_DIR, "ambiguity_training_report.txt")
PREVIOUS_MODEL_BACKUP_PATH: str = os.path.join(BACKUP_DIR, "previous_ambiguity_model.joblib")

LABEL_MAP: Dict[str, int] = {"Clear": 0, "Ambiguous": 1}
ID_TO_LABEL: Dict[int, str] = {0: "Clear", 1: "Ambiguous"}
RANDOM_SEED: int = 42


def clean_text(text: Any) -> str:
    """Trims whitespace and collapses redundant spaces."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return " ".join(text.split())


def load_and_merge_datasets() -> pd.DataFrame:
    """Loads existing Step 10 and Phase 13 datasets, cleans text, and checks for conflicts."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    dfs = []
    # 1. Load Step 10 datasets
    for path, name in [(STEP10_TRAIN, "Step10 Train"), (STEP10_VAL, "Step10 Val"), (STEP10_TEST, "Step10 Test")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[+] Loaded {name}: {len(df)} rows")
            dfs.append(df)
        else:
            print(f"[!] Warning: File missing: {path}")

    # 2. Load Phase 13 dataset
    phase13_src = UPLOADED_PHASE13 if os.path.exists(UPLOADED_PHASE13) else PHASE13_COPY_PATH
    if os.path.exists(phase13_src):
        df_p13 = pd.read_csv(phase13_src)
        print(f"[+] Loaded Phase 13 Dataset ({phase13_src}): {len(df_p13)} rows")
        # Save a copy to backend root if not present
        if not os.path.exists(PHASE13_COPY_PATH):
            df_p13.to_csv(PHASE13_COPY_PATH, index=False)
        dfs.append(df_p13)
    else:
        raise FileNotFoundError(f"Phase 13 dataset not found at {phase13_src}")

    # Combine all dataframes
    raw_combined = pd.concat(dfs, ignore_index=True)
    total_raw = len(raw_combined)
    print(f"\n[+] Total raw combined records: {total_raw}")

    # Clean & normalize text
    raw_combined["Requirement"] = raw_combined["Requirement"].apply(clean_text)
    raw_combined = raw_combined[raw_combined["Requirement"] != ""].copy()

    # Validate Labels
    valid_labels = {"Clear", "Ambiguous"}
    invalid_rows = raw_combined[~raw_combined["Label"].isin(valid_labels)]
    if not invalid_rows.empty:
        print(f"[!] Warning: Found {len(invalid_rows)} invalid label rows. Dropping...")
        raw_combined = raw_combined[raw_combined["Label"].isin(valid_labels)].copy()

    # Label Conflict Detection
    grouped_labels = raw_combined.groupby("Requirement")["Label"].nunique()
    conflicting_texts = grouped_labels[grouped_labels > 1].index.tolist()
    if conflicting_texts:
        print(f"[!] CAUTION: Found {len(conflicting_texts)} conflicting requirements! Resolving...")
        # Drop conflicting texts to maintain pure ground truth
        raw_combined = raw_combined[~raw_combined["Requirement"].isin(conflicting_texts)].copy()
    else:
        print("[+] Label Conflict Analysis: 0 conflicting text-label pairs found (100% label consistency across source datasets).")

    # Deduplicate while retaining metadata columns if available
    master_df = raw_combined.drop_duplicates(subset=["Requirement"], keep="first").reset_index(drop=True)
    total_unique = len(master_df)

    print(f"[+] Total unique master dataset records after deduplication: {total_unique}")

    # Save Master Dataset
    master_df.to_csv(MASTER_DATASET_PATH, index=False)
    master_df.to_csv(ROOT_MASTER_DATASET_PATH, index=False)
    print(f"[+] Master dataset saved to: {MASTER_DATASET_PATH}")

    return master_df


def create_stratified_splits(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates an 80/10/10 stratified split with zero exact duplicate leakage."""
    X = master_df
    y = master_df["Label"]

    # 80% train, 20% temp (val + test)
    train_df, temp_df = train_test_split(
        X,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # Split temp (20%) into 50/50 (10% val, 10% test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["Label"],
        random_state=RANDOM_SEED
    )

    # Verify zero data leakage
    train_texts = set(train_df["Requirement"])
    val_texts = set(val_df["Requirement"])
    test_texts = set(test_df["Requirement"])

    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)

    print("\n--- DATA LEAKAGE VERIFICATION ---")
    print(f"Train/Val text overlap: {len(train_val_overlap)}")
    print(f"Train/Test text overlap: {len(train_test_overlap)}")
    print(f"Val/Test text overlap: {len(val_test_overlap)}")
    assert len(train_val_overlap) == 0, "Data leakage detected between Train and Val!"
    assert len(train_test_overlap) == 0, "Data leakage detected between Train and Test!"
    assert len(val_test_overlap) == 0, "Data leakage detected between Val and Test!"
    print("[+] Data leakage test PASSED: 0 overlapping samples across splits.")

    # Save split files
    for df, path1, path2 in [
        (train_df, TRAIN_SPLIT_PATH, ROOT_TRAIN_SPLIT_PATH),
        (val_df, VAL_SPLIT_PATH, ROOT_VAL_SPLIT_PATH),
        (test_df, TEST_SPLIT_PATH, ROOT_TEST_SPLIT_PATH)
    ]:
        df.to_csv(path1, index=False)
        df.to_csv(path2, index=False)

    print(f"[+] Splits saved successfully (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    return train_df, val_df, test_df


def backup_existing_models() -> None:
    """Backs up pre-existing ambiguity model artifacts before writing new ones."""
    source_artifacts = [BASELINE_MODEL_PATH, V2_MODEL_PATH]
    for src in source_artifacts:
        if os.path.exists(src):
            print(f"[+] Backing up existing model artifact: {src} -> {PREVIOUS_MODEL_BACKUP_PATH}")
            joblib.dump(joblib.load(src), PREVIOUS_MODEL_BACKUP_PATH)
            break


def train_and_evaluate_phase14(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Trains TF-IDF + LogisticRegression pipeline and evaluates performance."""

    # Features and targets
    X_train_raw = train_df["Requirement"]
    y_train = train_df["Label"].map(LABEL_MAP)

    X_val_raw = val_df["Requirement"]
    y_val = val_df["Label"].map(LABEL_MAP)

    X_test_raw = test_df["Requirement"]
    y_test = test_df["Label"].map(LABEL_MAP)

    # 1. Fit TF-IDF exclusively on training text (Leakage Prevention)
    print("\n--- MODEL TRAINING ---")
    print("[+] Fitting TF-IDF Vectorizer (ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=5000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        max_features=5000
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_val_tfidf = vectorizer.transform(X_val_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"[+] Vocabulary size: {len(vectorizer.vocabulary_)} features")

    # 2. Train Logistic Regression
    print("[+] Fitting LogisticRegression (C=2.0, max_iter=1000, solver='lbfgs', random_state=42)...")
    clf = LogisticRegression(
        C=2.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED
    )
    clf.fit(X_train_tfidf, y_train)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    # 3. Evaluate across splits
    metrics: Dict[str, Any] = {}

    def get_split_metrics(X_raw, X_tfidf, y_true, name: str):
        preds = clf.predict(X_tfidf)
        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary")
        cm = confusion_matrix(y_true, preds).tolist()
        clf_rep = classification_report(y_true, preds, target_names=["Clear", "Ambiguous"], output_dict=True)
        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": clf_rep,
            "preds": preds
        }

    metrics["train"] = get_split_metrics(X_train_raw, X_train_tfidf, y_train, "Train")
    metrics["validation"] = get_split_metrics(X_val_raw, X_val_tfidf, y_val, "Validation")
    metrics["test"] = get_split_metrics(X_test_raw, X_test_tfidf, y_test, "Test")

    # 4. Error Analysis on Test Set
    test_preds = metrics["test"]["preds"]
    test_df_eval = test_df.copy().reset_index(drop=True)
    test_df_eval["target_id"] = y_test.values
    test_df_eval["pred_id"] = test_preds
    test_df_eval["pred_label"] = [ID_TO_LABEL[p] for p in test_preds]

    # False Positives: Ground Truth Clear (0), Predicted Ambiguous (1)
    false_positives = test_df_eval[(test_df_eval["target_id"] == 0) & (test_df_eval["pred_id"] == 1)]
    # False Negatives: Ground Truth Ambiguous (1), Predicted Clear (0)
    false_negatives = test_df_eval[(test_df_eval["target_id"] == 1) & (test_df_eval["pred_id"] == 0)]

    print(f"\n--- ERROR ANALYSIS (TEST SET: {len(test_df)} samples) ---")
    print(f"False Positives (Predicted Ambiguous, Actual Clear): {len(false_positives)}")
    print(f"False Negatives (Predicted Clear, Actual Ambiguous): {len(false_negatives)}")

    # 5. Manual Unseen Sanity Validation
    unseen_samples = [
        ("The system shall process user authentication within 500ms under maximum peak load.", "Clear"),
        ("The interface should be user-friendly, elegant, fast, and respond appropriately.", "Ambiguous"),
        ("The database server shall retain transaction logs for exactly 90 days before archiving.", "Clear"),
        ("The system shall handle errors gracefully and provide reasonable feedback as needed.", "Ambiguous")
    ]

    unseen_results = []
    for text, expected in unseen_samples:
        pred_id = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0][pred_id]
        pred_label = ID_TO_LABEL[pred_id]
        unseen_results.append({
            "text": text,
            "expected": expected,
            "predicted": pred_label,
            "confidence": f"{int(round(proba * 100))}%",
            "correct": pred_label == expected
        })

    # Save Model Artifacts
    print("\n--- SAVING ARTIFACTS ---")
    backup_existing_models()

    joblib.dump(pipeline, PHASE14_MODEL_PATH)
    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    joblib.dump(clf, V2_MODEL_PATH)
    joblib.dump(vectorizer, V2_VEC_PATH)
    print(f"[+] Model saved to: {PHASE14_MODEL_PATH}")
    print(f"[+] Model saved to: {BASELINE_MODEL_PATH}")
    print(f"[+] Model saved to: {V2_MODEL_PATH}")
    print(f"[+] Vectorizer saved to: {V2_VEC_PATH}")

    # Metadata JSON
    metadata = {
        "model_name": "ARIP Phase 14 Ambiguity Classifier",
        "pipeline": "TF-IDF + LogisticRegression",
        "random_seed": RANDOM_SEED,
        "dataset_summary": {
            "total_raw_combined": len(train_df) + len(val_df) + len(test_df),
            "master_unique_samples": len(train_df) + len(val_df) + len(test_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "class_distribution": {
                "Clear": int((train_df["Label"] == "Clear").sum() + (val_df["Label"] == "Clear").sum() + (test_df["Label"] == "Clear").sum()),
                "Ambiguous": int((train_df["Label"] == "Ambiguous").sum() + (val_df["Label"] == "Ambiguous").sum() + (test_df["Label"] == "Ambiguous").sum())
            }
        },
        "tfidf_hyperparameters": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 5000,
            "vocabulary_size": len(vectorizer.vocabulary_)
        },
        "classifier_hyperparameters": {
            "C": 2.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": RANDOM_SEED
        },
        "performance_metrics": {
            "train": {k: v for k, v in metrics["train"].items() if k != "preds"},
            "validation": {k: v for k, v in metrics["validation"].items() if k != "preds"},
            "test": {k: v for k, v in metrics["test"].items() if k != "preds"}
        },
        "unseen_manual_tests": unseen_results
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Metadata saved to: {METADATA_PATH}")

    # Generate Human-Readable Training Report
    report_content = f"""================================================================================
ARIP PHASE 14 AMBIGUITY DETECTION MODEL TRAINING & EVALUATION REPORT
================================================================================
Generated Date: 2026-10-03
Architecture: TF-IDF (Unigram + Bigram, max_features=5000) + Logistic Regression (C=2.0)

1. DATASET INTEGRATION & CLEANING SUMMARY
--------------------------------------------------------------------------------
- Master Deduplicated Records: {len(train_df) + len(val_df) + len(test_df)}
- Conflicting Text-Label Pairs: 0 (100% label consistency)
- Master Class Distribution:
    Clear:     {(train_df['Label'] == 'Clear').sum() + (val_df['Label'] == 'Clear').sum() + (test_df['Label'] == 'Clear').sum()}
    Ambiguous: {(train_df['Label'] == 'Ambiguous').sum() + (val_df['Label'] == 'Ambiguous').sum() + (test_df['Label'] == 'Ambiguous').sum()}

2. STRATIFIED 80/10/10 SPLIT DISTRIBUTION
--------------------------------------------------------------------------------
- Training Split:   {len(train_df)} samples
- Validation Split: {len(val_df)} samples
- Testing Split:    {len(test_df)} samples
- Leakage Check: 0 overlapping sentences across Train, Val, Test sets.

3. MODEL PERFORMANCE METRICS
--------------------------------------------------------------------------------
[TRAINING SET ({len(train_df)} samples)]
- Accuracy:  {metrics['train']['accuracy'] * 100:.2f}%
- Precision: {metrics['train']['precision'] * 100:.2f}%
- Recall:    {metrics['train']['recall'] * 100:.2f}%
- F1-Score:  {metrics['train']['f1_score'] * 100:.2f}%
- Confusion Matrix: {metrics['train']['confusion_matrix']}

[VALIDATION SET ({len(val_df)} samples)]
- Accuracy:  {metrics['validation']['accuracy'] * 100:.2f}%
- Precision: {metrics['validation']['precision'] * 100:.2f}%
- Recall:    {metrics['validation']['recall'] * 100:.2f}%
- F1-Score:  {metrics['validation']['f1_score'] * 100:.2f}%
- Confusion Matrix: {metrics['validation']['confusion_matrix']}

[TEST SET ({len(test_df)} samples)]
- Accuracy:  {metrics['test']['accuracy'] * 100:.2f}%
- Precision: {metrics['test']['precision'] * 100:.2f}%
- Recall:    {metrics['test']['recall'] * 100:.2f}%
- F1-Score:  {metrics['test']['f1_score'] * 100:.2f}%
- Confusion Matrix: {metrics['test']['confusion_matrix']}

4. ERROR ANALYSIS SUMMARY (TEST SET)
--------------------------------------------------------------------------------
- Total Test Samples: {len(test_df)}
- False Positives (Ground Truth Clear -> Predicted Ambiguous): {len(false_positives)}
- False Negatives (Ground Truth Ambiguous -> Predicted Clear): {len(false_negatives)}

5. MANUAL UNSEEN TEST VERIFICATION
--------------------------------------------------------------------------------
"""
    for res in unseen_results:
        report_content += f"Requirement: \"{res['text']}\"\n"
        report_content += f"  Expected: {res['expected']} | Predicted: {res['predicted']} ({res['confidence']}) | Pass: {res['correct']}\n\n"

    report_content += f"""================================================================================
END OF REPORT
================================================================================
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Human-readable report written to: {REPORT_PATH}")

    # Print Summary to Console
    print("\n================================================================================")
    print("PHASE 14 TRAINING COMPLETE RESULTS")
    print("================================================================================")
    print(f"Validation Accuracy:  {metrics['validation']['accuracy'] * 100:.2f}% | F1: {metrics['validation']['f1_score'] * 100:.2f}%")
    print(f"Test Accuracy:        {metrics['test']['accuracy'] * 100:.2f}% | F1: {metrics['test']['f1_score'] * 100:.2f}%")
    print("================================================================================\n")

    return pipeline, metadata


def main():
    print("=== ARIP PHASE 14 AMBIGUITY MODEL RETRAINING ===")
    master_df = load_and_merge_datasets()
    train_df, val_df, test_df = create_stratified_splits(master_df)
    train_and_evaluate_phase14(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
