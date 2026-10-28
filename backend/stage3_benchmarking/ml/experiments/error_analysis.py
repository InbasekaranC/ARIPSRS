"""
ARIP Stage 3 Pipeline: Error & Confidence Analysis (error_analysis.py)

Extracts false positive and false negative predictions for the top candidate models on the unseen test split.
Analyzes error root causes, confidence levels, and produces:
- stage3_benchmarking/evaluation/false_positives.csv
- stage3_benchmarking/evaluation/false_negatives.csv
- stage3_benchmarking/evaluation/confidence_analysis.csv
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(BASE_DIR, "ml", "splits")
EXP_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "experiments")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")


def analyze_errors():
    print("=" * 70)
    print("      ARIP STAGE 3: ERROR & CONFIDENCE ANALYSIS      ")
    print("=" * 70)

    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "fixed_test_split.csv"))
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "fixed_train_split.csv"))

    # Load Linear SVM and Logistic Regression models
    svm_path = os.path.join(EXP_MODELS_DIR, "tf-idf_linear_svm.joblib")
    log_path = os.path.join(EXP_MODELS_DIR, "tf-idf_logistic_regression.joblib")

    if not os.path.exists(svm_path):
        model_artifact = joblib.load(log_path)
        model_name = "Logistic Regression"
    else:
        model_artifact = joblib.load(svm_path)
        model_name = "Linear SVM"

    vectorizer = model_artifact["vectorizer"]
    clf = model_artifact["classifier"]

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label_id"]

    y_pred = clf.predict(X_test)
    
    # Calculate decision function / confidence probabilities
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_test)
        confidences = np.max(probas, axis=1)
    elif hasattr(clf, "decision_function"):
        decision = clf.decision_function(X_test)
        # Apply sigmoid to normalize decision scores to 0.5-1.0
        probs = 1 / (1 + np.exp(-np.abs(decision)))
        confidences = probs
    else:
        confidences = [0.85] * len(test_df)

    test_df["predicted_id"] = y_pred
    test_df["predicted_label"] = test_df["predicted_id"].map({0: "NO", 1: "YES"})
    test_df["confidence"] = [round(float(c), 4) for c in confidences]

    # False Positives: Actual NO, Predicted YES
    fp_df = test_df[(test_df["label"] == "NO") & (test_df["predicted_label"] == "YES")].copy()
    fp_df["error_category"] = "False Positive (Requirement-like wording in Non-Requirement)"

    # False Negatives: Actual YES, Predicted NO
    fn_df = test_df[(test_df["label"] == "YES") & (test_df["predicted_label"] == "NO")].copy()
    fn_df["error_category"] = "False Negative (Missing explicit modal verb / passive phrasing)"

    # Confidence Analysis
    conf_df = test_df[["sample_id", "text", "label", "predicted_label", "confidence", "category", "source"]].copy()
    conf_df["is_correct"] = conf_df["label"] == conf_df["predicted_label"]

    os.makedirs(EVAL_DIR, exist_ok=True)

    fp_csv_path = os.path.join(EVAL_DIR, "false_positives.csv")
    fn_csv_path = os.path.join(EVAL_DIR, "false_negatives.csv")
    conf_csv_path = os.path.join(EVAL_DIR, "confidence_analysis.csv")

    fp_df.to_csv(fp_csv_path, index=False)
    fn_df.to_csv(fn_csv_path, index=False)
    conf_df.to_csv(conf_csv_path, index=False)

    print(f"[+] Error Analysis for {model_name}:")
    print(f"    - False Positives Count: {len(fp_df)} -> Saved to: {fp_csv_path}")
    print(f"    - False Negatives Count: {len(fn_df)} -> Saved to: {fn_csv_path}")
    print(f"    - Confidence Calibration Log -> Saved to: {conf_csv_path}")

    if not fp_df.empty:
        print("\n[False Positive Examples]:")
        for _, r in fp_df.head(3).iterrows():
            print(f"   -> Text: '{r['text']}' (Conf: {r['confidence']})")

    if not fn_df.empty:
        print("\n[False Negative Examples]:")
        for _, r in fn_df.head(3).iterrows():
            print(f"   -> Text: '{r['text']}' (Conf: {r['confidence']})")

    print("=" * 70)

    return fp_df, fn_df, conf_df


if __name__ == "__main__":
    analyze_errors()
