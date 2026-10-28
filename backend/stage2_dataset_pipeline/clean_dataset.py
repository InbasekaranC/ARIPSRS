"""
ARIP Stage 2 Pipeline: Part 2, 3, 4 — Dataset Cleaning & Deduplication (clean_dataset.py)

Applies exact, normalized, cross-label conflict, and semantic deduplication.
Preserves original text, normalized text, label, and metadata fields.
Outputs:
- data/processed/cleaned_dataset.csv
- data/rejected/conflicting_samples.csv
"""

import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "requirement_classification_data.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REJECTED_DIR = os.path.join(BASE_DIR, "data", "rejected")

CLEANED_CSV_PATH = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")
CONFLICTING_CSV_PATH = os.path.join(REJECTED_DIR, "conflicting_samples.csv")

# Explicit Configurable Semantic Duplicate Threshold
SEMANTIC_DUPLICATE_THRESHOLD: float = 0.90


def normalize_text(text: str) -> str:
    """
    Normalizes text strictly for duplicate and conflict detection.
    Preserves original text untouched in output.
    """
    if not isinstance(text, str):
        return ""
    clean = text.strip().lower()
    clean = re.sub(r'\s+', ' ', clean)
    clean = re.sub(r'[^\w\s]', '', clean)
    return clean


def clean_and_deduplicate() -> pd.DataFrame:
    print("=" * 70)
    print("      ARIP STAGE 2: PARTS 2, 3, 4 — CLEANING & DEDUPLICATION      ")
    print("=" * 70)
    print(f"[+] Loading raw dataset from: {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)
    initial_count = len(df)

    text_col = "Sentence"
    label_col = "Requirement"

    # Part 2: Normalization
    df["original_text"] = df[text_col].astype(str).str.strip()
    df["normalized_text"] = df["original_text"].apply(normalize_text)
    df["label"] = df[label_col].astype(str).str.strip()
    df["source"] = "raw_requirement_classification_data"

    # Filter out blank rows
    df = df[df["normalized_text"] != ""].copy()

    # Part 3: Identify Cross-Label Conflicts
    conflicts = df.groupby("normalized_text")["label"].nunique()
    conflicting_norms = set(conflicts[conflicts > 1].index)

    conflicting_df = df[df["normalized_text"].isin(conflicting_norms)].copy()
    if not conflicting_df.empty:
        os.makedirs(REJECTED_DIR, exist_ok=True)
        conflicting_df[["original_text", "normalized_text", "label", "source"]].to_csv(CONFLICTING_CSV_PATH, index=False)
        print(f"[!] Found {len(conflicting_df)} cross-label conflicting rows. Saved to: {CONFLICTING_CSV_PATH}")
    else:
        print("[+] Zero cross-label conflicts found.")

    # Remove conflicting rows from clean pipeline
    df = df[~df["normalized_text"].isin(conflicting_norms)].copy()

    # Part 3: Exact & Normalized Duplicate Removal (Keep first instance)
    df = df.drop_duplicates(subset=["normalized_text"], keep="first").copy()
    exact_dedup_count = len(df)
    print(f"[+] Exact/Normalized Deduplication: {initial_count} -> {exact_dedup_count} samples.")

    # Part 4: Semantic Duplicate Detection (TF-IDF + Cosine Similarity)
    print(f"[+] Performing Semantic Near-Duplicate Filtering (Threshold: {SEMANTIC_DUPLICATE_THRESHOLD})...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(df["normalized_text"])

    sim_matrix = cosine_similarity(tfidf_matrix)
    num_samples = len(df)
    keep_indices = []
    dropped_semantic_count = 0

    indices = list(range(num_samples))
    dropped_set = set()

    for i in range(num_samples):
        if i in dropped_set:
            continue
        keep_indices.append(i)
        # Find candidates with similarity >= threshold and same label
        label_i = df.iloc[i]["label"]
        for j in range(i + 1, num_samples):
            if j in dropped_set:
                continue
            if df.iloc[j]["label"] == label_i and sim_matrix[i, j] >= SEMANTIC_DUPLICATE_THRESHOLD:
                dropped_set.add(j)
                dropped_semantic_count += 1

    cleaned_df = df.iloc[keep_indices].copy().reset_index(drop=True)
    cleaned_df["sample_id"] = [f"CLEAN_{i+1:04d}" for i in range(len(cleaned_df))]

    # Reorder columns
    final_cols = ["sample_id", "original_text", "normalized_text", "label", "source"]
    cleaned_df = cleaned_df[final_cols]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)

    print(f"[+] Semantic Deduplication: Removed {dropped_semantic_count} near-duplicate samples.")
    print(f"[+] Cleaned baseline dataset saved to: {CLEANED_CSV_PATH}")
    print(f"    - Final Cleaned Count: {len(cleaned_df)} unique samples")
    print(f"    - YES count: {sum(cleaned_df['label'] == 'YES')}")
    print(f"    - NO count: {sum(cleaned_df['label'] == 'NO')}")
    print("=" * 70)

    return cleaned_df


if __name__ == "__main__":
    clean_and_deduplicate()
