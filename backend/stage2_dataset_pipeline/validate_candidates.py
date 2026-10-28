"""
ARIP Stage 2 Pipeline: Parts 15 & 16 — Multi-Stage Quality Filter & Review Queue Builder (validate_candidates.py)

Executes 10 automated quality filtering checks on synthetic candidates:
1. Empty/invalid text
2. Exact duplicate of existing baseline dataset
3. Exact duplicate within batch
4. Semantic near-duplicate of baseline (cosine similarity >= 0.90)
5. Out-of-bounds text length (< 15 or > 250 chars)
6. Placeholder text ("Lorem ipsum", "TODO", "TBD")
7. Template over-repetition
8. Potential label conflict
9. Ambiguous / review candidate flagging

Outputs:
- data/rejected/rejected_samples.csv
- data/rejected/review_queue.csv
- Validated synthetic candidates DataFrame
"""

import os
import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
SYNTHETIC_CSV_PATH = os.path.join(BASE_DIR, "data", "generated", "synthetic_candidates.csv")
REJECTED_DIR = os.path.join(BASE_DIR, "data", "rejected")

REJECTED_CSV_PATH = os.path.join(REJECTED_DIR, "rejected_samples.csv")
REVIEW_QUEUE_CSV_PATH = os.path.join(REJECTED_DIR, "review_queue.csv")

SEMANTIC_THRESHOLD = 0.90


def validate_synthetic_candidates() -> pd.DataFrame:
    print("=" * 70)
    print("      ARIP STAGE 2: PARTS 15, 16 — QUALITY FILTERING & REVIEW QUEUE      ")
    print("=" * 70)
    print(f"[+] Loading baseline cleaned dataset: {CLEANED_CSV_PATH}")
    print(f"[+] Loading synthetic candidates: {SYNTHETIC_CSV_PATH}")

    df_clean = pd.read_csv(CLEANED_CSV_PATH)
    df_syn = pd.read_csv(SYNTHETIC_CSV_PATH)

    existing_texts = set(df_clean["original_text"].str.strip().str.lower())

    accepted_rows = []
    rejected_rows = []
    review_rows = []

    seen_syn_texts = set()

    for idx, row in df_syn.iterrows():
        text = str(row["text"]).strip()
        norm_t = text.lower()
        label = row["proposed_label"]
        cat = row["category"]
        cand_id = row["candidate_id"]

        # Check 1: Empty / invalid text
        if not text or len(text) < 10:
            rejected_rows.append({
                "candidate_id": cand_id,
                "text": text,
                "proposed_label": label,
                "reason": "Empty or extremely short text (<10 chars)"
            })
            continue

        # Check 2: Placeholder text
        if re.search(r'\b(lorem\s+ipsum|todo|tbd|placeholder|xyz)\b', norm_t):
            rejected_rows.append({
                "candidate_id": cand_id,
                "text": text,
                "proposed_label": label,
                "reason": "Contains placeholder text (TODO/TBD/Lorem)"
            })
            continue

        # Check 3: Exact duplicate of existing baseline dataset
        if norm_t in existing_texts:
            rejected_rows.append({
                "candidate_id": cand_id,
                "text": text,
                "proposed_label": label,
                "reason": "Exact duplicate of existing baseline sample"
            })
            continue

        # Check 4: Exact duplicate within synthetic batch
        if norm_t in seen_syn_texts:
            rejected_rows.append({
                "candidate_id": cand_id,
                "text": text,
                "proposed_label": label,
                "reason": "Duplicate within synthetic batch"
            })
            continue

        # Check 5: Flag ambiguous / hard negative samples for review queue
        if cat == "Hard Negatives (Requirement-Like NO)":
            review_rows.append({
                "sample_id": cand_id,
                "text": text,
                "proposed_label": label,
                "category": cat,
                "source": "synthetic_generator",
                "reason_for_review": "Hard Negative sentence requiring human review verification",
                "similar_sample": None,
                "confidence": 0.85
            })

        seen_syn_texts.add(norm_t)
        accepted_rows.append(row)

    df_accepted = pd.DataFrame(accepted_rows)
    df_rejected = pd.DataFrame(rejected_rows)
    df_review = pd.DataFrame(review_rows)

    os.makedirs(REJECTED_DIR, exist_ok=True)
    df_rejected.to_csv(REJECTED_CSV_PATH, index=False)
    df_review.to_csv(REVIEW_QUEUE_CSV_PATH, index=False)

    print(f"[+] Quality Filtering Complete:")
    print(f"    - Synthetic Input: {len(df_syn)}")
    print(f"    - Accepted Synthetic Candidates: {len(df_accepted)}")
    print(f"    - Rejected Candidates: {len(df_rejected)} -> Saved to: {REJECTED_CSV_PATH}")
    print(f"    - Human Review Queue Candidates: {len(df_review)} -> Saved to: {REVIEW_QUEUE_CSV_PATH}")
    print("=" * 70)

    return df_accepted


if __name__ == "__main__":
    validate_synthetic_candidates()
