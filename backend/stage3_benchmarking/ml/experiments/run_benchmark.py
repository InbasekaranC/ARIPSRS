"""
ARIP Stage 3 Pipeline: Master Benchmarking Orchestrator (run_benchmark.py)

Trains and benchmarks 5 candidate requirement-classification models on the fixed train split
and evaluates on the unseen test split with strict data leakage prevention.

Outputs:
- stage3_benchmarking/evaluation/model_comparison.csv
- stage3_benchmarking/evaluation/confusion_matrices/
- stage3_benchmarking/ml/models/experiments/*.joblib
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(BASE_DIR, "ml", "splits")
EXP_MODELS_DIR = os.path.join(BASE_DIR, "ml", "models", "experiments")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
CM_DIR = os.path.join(EVAL_DIR, "confusion_matrices")

RANDOM_SEED = 42


def run_benchmark():
    print("=" * 70)
    print("      ARIP STAGE 3: CANDIDATE MODEL TRAINING & BENCHMARKING      ")
    print("=" * 70)

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "fixed_train_split.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "fixed_validation_split.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "fixed_test_split.csv"))

    print(f"[+] Splits Loaded: Train ({len(train_df)}), Val ({len(val_df)}), Unseen Test ({len(test_df)})")

    # STAGE 3 LEAKAGE PREVENTION: Fit vectorizer strictly on training data ONLY
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label_id"]

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label_id"]

    candidate_models = {
        "TF-IDF + Logistic Regression": LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=RANDOM_SEED),
        "TF-IDF + Linear SVM": LinearSVC(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
        "TF-IDF + Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "TF-IDF + Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_SEED),
        "TF-IDF + Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_SEED)
    }

    results = []
    os.makedirs(EXP_MODELS_DIR, exist_ok=True)
    os.makedirs(CM_DIR, exist_ok=True)

    print("\n[+] Benchmarking Candidate Models against Unseen Test Split...\n")

    for model_name, clf in candidate_models.items():
        # Measure training time
        t0 = time.time()
        clf.fit(X_train, y_train)
        t_train = (time.time() - t0) * 1000  # ms

        # Measure inference time
        t0 = time.time()
        y_pred = clf.predict(X_test)
        t_infer = (time.time() - t0) * 1000  # ms

        # Save model pipeline artifact
        model_filename = model_name.lower().replace(" + ", "_").replace(" ", "_") + ".joblib"
        model_path = os.path.join(EXP_MODELS_DIR, model_filename)
        joblib.dump({"vectorizer": vectorizer, "classifier": clf}, model_path)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()

        req_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # YES recall
        no_rejection = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # NO recall (rejection rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        cm_payload = {
            "model_name": model_name,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        }

        cm_json_path = os.path.join(CM_DIR, f"cm_{model_filename.replace('.joblib', '')}.json")
        with open(cm_json_path, "w") as f:
            json.dump(cm_payload, f, indent=4)

        results.append({
            "Model": model_name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "Requirement_Recall": round(float(req_recall), 4),
            "NonReq_Rejection_Rate": round(float(no_rejection), 4),
            "False_Positive_Rate": round(float(fpr), 4),
            "False_Negative_Rate": round(float(fnr), 4),
            "FP_Count": int(fp),
            "FN_Count": int(fn),
            "Train_Time_ms": round(float(t_train), 2),
            "Inference_Time_ms": round(float(t_infer), 2)
        })

    results_df = pd.DataFrame(results)
    results_df.sort_values(by="F1_Score", ascending=False, inplace=True)

    csv_path = os.path.join(EVAL_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)

    print(results_df.to_string(index=False))
    print(f"\n[+] Model comparison saved to: {csv_path}")
    print("=" * 70)

    return results_df


if __name__ == "__main__":
    run_benchmark()
