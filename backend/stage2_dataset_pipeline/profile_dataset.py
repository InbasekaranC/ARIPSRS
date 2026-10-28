"""
ARIP Stage 2 Pipeline: Part 1 — Dataset Profiling (profile_dataset.py)

Performs a read-only technical audit and profiling of the raw dataset.
Calculates statistical metrics, duplicate counts, text length distribution,
and outputs data/reports/dataset_statistics.json.
"""

import os
import json
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "requirement_classification_data.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
STATS_JSON_PATH = os.path.join(REPORTS_DIR, "dataset_statistics.json")


def normalize_text_for_comparison(text: str) -> str:
    """Normalizes text for duplicate and conflict detection."""
    if not isinstance(text, str):
        return ""
    # Unicode / whitespace / lowercase / punctuation trimming
    clean = text.strip().lower()
    clean = re.sub(r'\s+', ' ', clean)
    clean = re.sub(r'[^\w\s]', '', clean)
    return clean


def profile_raw_dataset() -> dict:
    print("=" * 70)
    print("      ARIP STAGE 2: PART 1 — DATASET PROFILING      ")
    print("=" * 70)
    print(f"[+] Loading raw dataset from: {RAW_CSV_PATH}")

    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Raw dataset backup not found at {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)

    total_rows = len(df)
    cols = df.columns.tolist()
    text_col = "Sentence" if "Sentence" in cols else cols[0]
    label_col = "Requirement" if "Requirement" in cols else cols[1]

    # Clean missing / empty check
    missing_text_count = int(df[text_col].isnull().sum())
    missing_label_count = int(df[label_col].isnull().sum())
    empty_whitespace_count = int(df[text_col].fillna("").astype(str).str.strip().eq("").sum())

    # Raw Class Distribution
    raw_class_dist = df[label_col].value_counts().to_dict()

    # Exact duplicate rows
    exact_duplicate_rows = int(df.duplicated().sum())

    # Duplicate sentences (text column only)
    duplicate_sentences_count = int(df[text_col].duplicated().sum())

    # Normalized duplicates
    df["norm_text"] = df[text_col].fillna("").astype(str).apply(normalize_text_for_comparison)
    norm_duplicate_count = int(df["norm_text"].duplicated().sum())

    # Conflicting labels (same sentence with different labels)
    conflicts = df.groupby(text_col)[label_col].nunique()
    conflicting_sentences_count = int((conflicts > 1).sum())

    # Unique sample counts
    unique_sentences_count = int(df[text_col].nunique())
    unique_by_class = df.groupby(label_col)[text_col].nunique().to_dict()

    # Text length statistics
    text_lens = df[text_col].fillna("").astype(str).str.len()
    min_len = int(text_lens.min())
    max_len = int(text_lens.max())
    mean_len = round(float(text_lens.mean()), 2)
    median_len = float(text_lens.median())

    # Suspicious / Malformed samples (extremely short < 15 chars or extremely long > 200 chars)
    short_samples = df[text_lens < 15][text_col].tolist()
    long_samples = df[text_lens > 200][text_col].tolist()

    stats_report = {
        "dataset_name": "requirement_classification_data.csv",
        "total_rows": total_rows,
        "columns": cols,
        "text_column": text_col,
        "label_column": label_col,
        "raw_class_distribution": raw_class_dist,
        "missing_values": {
            "text_missing": missing_text_count,
            "label_missing": missing_label_count,
            "empty_or_whitespace": empty_whitespace_count
        },
        "duplication_metrics": {
            "exact_duplicate_rows": exact_duplicate_rows,
            "duplicate_sentence_text": duplicate_sentences_count,
            "normalized_duplicate_count": norm_duplicate_count,
            "duplication_percentage": round((exact_duplicate_rows / total_rows) * 100, 2)
        },
        "label_conflict_metrics": {
            "conflicting_sentences_count": conflicting_sentences_count
        },
        "uniqueness_metrics": {
            "unique_sentences_count": unique_sentences_count,
            "unique_samples_per_class": unique_by_class
        },
        "text_length_statistics": {
            "min_length_chars": min_len,
            "max_length_chars": max_len,
            "mean_length_chars": mean_len,
            "median_length_chars": median_len
        },
        "suspicious_samples": {
            "short_samples_count": len(short_samples),
            "short_sample_examples": short_samples[:5],
            "long_samples_count": len(long_samples),
            "long_sample_examples": long_samples[:5]
        }
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(STATS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=4)

    print(f"[+] Profiling complete. Report saved to: {STATS_JSON_PATH}")
    print(f"    - Total Rows: {total_rows}")
    print(f"    - Unique Sentences: {unique_sentences_count}")
    print(f"    - Exact Duplicate Rows: {exact_duplicate_rows} ({stats_report['duplication_metrics']['duplication_percentage']}%)")
    print(f"    - Conflicting Labels: {conflicting_sentences_count}")
    print("=" * 70)

    return stats_report


if __name__ == "__main__":
    profile_raw_dataset()
