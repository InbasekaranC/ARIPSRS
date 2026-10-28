"""
AI Requirements Intelligence Platform (ARIP) - Module 1: Ambiguity Detection
Step 11 Requirement Ambiguity Model Training Pipeline (train.py)

Trains TF-IDF + Logistic Regression ambiguity classification model strictly using the three
validated Step 10 dataset splits (Train: 1696, Val: 274, Test: 275).
Prevents data leakage by fitting vectorizer exclusively on training text.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Paths to fixed Step 10 datasets
TRAIN_PATH: str = os.path.join(BASE_DIR, "requirements_train_step10_validated.csv")
VAL_PATH: str = os.path.join(BASE_DIR, "requirements_val_step10_validated.csv")
TEST_PATH: str = os.path.join(BASE_DIR, "requirements_test_step10_validated.csv")

# Model artifact directories
MODEL_DIR: str = os.path.join(BASE_DIR, "model", "saved_model")
LABEL_MAPPING_PATH: str = os.path.join(MODEL_DIR, "label_mapping.json")
BASELINE_MODEL_PATH: str = os.path.join(MODEL_DIR, "baseline_model.joblib")
EVAL_REPORT_PATH: str = os.path.join(MODEL_DIR, "model_evaluation.json")
EVAL_DIR: str = os.path.join(BASE_DIR, "evaluation")
CM_REPORT_PATH: str = os.path.join(EVAL_DIR, "ambiguity_confusion_matrices.json")

LABEL_MAP: Dict[str, int] = {"Clear": 0, "Ambiguous": 1}
ID_TO_LABEL: Dict[int, str] = {0: "Clear", 1: "Ambiguous"}
RANDOM_SEED: int = 42


def preprocess_text(text: str) -> str:
    """Normalizes input requirement text without destroying ambiguity trigger keywords."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def validate_and_load_split(csv_path: str, expected_name: str) -> pd.DataFrame:
    """Loads and validates a Step 10 dataset split."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Step 10 dataset split missing: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Requirement" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"{expected_name} split missing 'Requirement' or 'Label' columns.")

    df["Requirement"] = df["Requirement"].apply(preprocess_text)
    df = df[df["Requirement"] != ""].copy()

    df["label_id"] = df["Label"].map(LABEL_MAP)
    if df["label_id"].isnull().any():
        raise ValueError(f"{expected_name} split contains unrecognized label values (must be 'Clear' or 'Ambiguous').")

    return df


def train_and_evaluate() -> Tuple[Pipeline, TfidfVectorizer, LogisticRegression, Dict[str, Any]]:
    """Trains TF-IDF + Logistic Regression ambiguity classifier and evaluates on val and test splits."""
    # 1. Load fixed splits
    train_df = validate_and_load_split(TRAIN_PATH, "Training")
    val_df = validate_and_load_split(VAL_PATH, "Validation")
    test_df = validate_and_load_split(TEST_PATH, "Testing")

    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)

    train_clear = sum(train_df["Label"] == "Clear")
    train_amb = sum(train_df["Label"] == "Ambiguous")
    val_clear = sum(val_df["Label"] == "Clear")
    val_amb = sum(val_df["Label"] == "Ambiguous")
    test_clear = sum(test_df["Label"] == "Clear")
    test_amb = sum(test_df["Label"] == "Ambiguous")

    # 2. Extract features and targets
    X_train_raw = train_df["Requirement"]
    y_train = train_df["label_id"]

    X_val_raw = val_df["Requirement"]
    y_val = val_df["label_id"]

    X_test_raw = test_df["Requirement"]
    y_test = test_df["label_id"]

    # 3. Fit TF-IDF exclusively on training set (Leakage Prevention)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_raw)

    # 4. Transform validation and test sets
    X_val = vectorizer.transform(X_val_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 5. Train Logistic Regression classifier
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    # Construct pipeline wrapper for backward-compatibility with predict.py
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf)
    ])

    # 6. Evaluate on Validation split
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="binary")
    val_cm = confusion_matrix(y_val, val_preds).tolist()

    # 7. Evaluate on Test split
    test_preds = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="binary")
    test_cm = confusion_matrix(y_test, test_preds).tolist()

    # 8. Save Label Mapping
    os.makedirs(MODEL_DIR, exist_ok=True)
    mapping_payload = {
        "label2id": LABEL_MAP,
        "id2label": {str(k): v for k, v in ID_TO_LABEL.items()}
    }
    with open(LABEL_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=4)

    # 9. Save Model Artifacts
    joblib.dump(pipeline, BASELINE_MODEL_PATH)

    # Save vectorizer and model separately for explicit step requirement
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer_v2.pkl")
    model_path = os.path.join(MODEL_DIR, "ambiguity_model_v2.pkl")
    joblib.dump(vectorizer, vec_path)
    joblib.dump(clf, model_path)

    # 10. Save Evaluation Reports
    eval_metrics = {
        "training_samples": n_train,
        "validation_samples": n_val,
        "test_samples": n_test,
        "class_counts": {
            "training": {"Clear": train_clear, "Ambiguous": train_amb},
            "validation": {"Clear": val_clear, "Ambiguous": val_amb},
            "testing": {"Clear": test_clear, "Ambiguous": test_amb}
        },
        "model_type": "TF-IDF + Logistic Regression",
        "tf_idf_params": {"ngram_range": [1, 2], "max_features": 1500, "sublinear_tf": True},
        "validation_metrics": {
            "accuracy": round(float(val_acc), 4),
            "precision": round(float(val_prec), 4),
            "recall": round(float(val_rec), 4),
            "f1_score": round(float(val_f1), 4),
            "confusion_matrix": val_cm
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall": round(float(test_rec), 4),
            "f1_score": round(float(test_f1), 4),
            "confusion_matrix": test_cm
        },
        "model_path": BASELINE_MODEL_PATH,
        "vectorizer_path": vec_path
    }

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=4)

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(CM_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "validation_confusion_matrix": val_cm,
            "test_confusion_matrix": test_cm
        }, f, indent=4)

    # 11. Print Step 14 Formatted Output
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"Test samples: {n_test}\n")

    print(f"Training:      Clear: {train_clear:<5} Ambiguous: {train_amb:<5}")
    print(f"Validation:    Clear: {val_clear:<5} Ambiguous: {val_amb:<5}")
    print(f"Test:          Clear: {test_clear:<5} Ambiguous: {test_amb:<5}\n")

    print("Training completed.\n")

    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Validation Precision: {val_prec * 100:.2f}%")
    print(f"Validation Recall: {val_rec * 100:.2f}%")
    print(f"Validation F1: {val_f1 * 100:.2f}%\n")

    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Precision: {test_prec * 100:.2f}%")
    print(f"Test Recall: {test_rec * 100:.2f}%")
    print(f"Test F1: {test_f1 * 100:.2f}%\n")

    print(f"Model saved to: {BASELINE_MODEL_PATH}")
    print(f"Vectorizer saved to: {vec_path}")

    return pipeline, vectorizer, clf, eval_metrics


if __name__ == "__main__":
    train_and_evaluate()
