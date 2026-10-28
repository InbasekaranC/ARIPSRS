"""
ARIP Stage 3 Pipeline: Fixed Data Splitter (create_splits.py)

Creates fixed 70% Train, 15% Validation, and 15% Test splits from Stage 2 final_dataset.csv.
Enforces stratification and verifies zero sample overlap across splits.

Outputs:
- stage3_benchmarking/ml/splits/fixed_train_split.csv
- stage3_benchmarking/ml/splits/fixed_validation_split.csv
- stage3_benchmarking/ml/splits/fixed_test_split.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "final_dataset.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "ml", "splits")

RANDOM_SEED = 42


def create_fixed_splits():
    print("=" * 70)
    print("      ARIP STAGE 3: CREATING FIXED 70/15/15 DATA SPLITS      ")
    print("=" * 70)
    print(f"[+] Loading final dataset: {FINAL_DATASET_PATH}")

    df = pd.read_csv(FINAL_DATASET_PATH)
    total_len = len(df)
    print(f"[+] Total samples: {total_len} (YES: {sum(df['label']=='YES')}, NO: {sum(df['label']=='NO')})")

    # Map labels to numeric
    df["label_id"] = df["label"].map({"NO": 0, "YES": 1})

    # First split: 70% train vs 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["label_id"]
    )

    # Second split: 50% of temp for val (15% total), 50% for test (15% total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label_id"]
    )

    # Verify zero overlap
    train_ids = set(train_df["sample_id"])
    val_ids = set(val_df["sample_id"])
    test_ids = set(test_df["sample_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train-Val overlap detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train-Test overlap detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val-Test overlap detected!"

    os.makedirs(SPLITS_DIR, exist_ok=True)

    train_path = os.path.join(SPLITS_DIR, "fixed_train_split.csv")
    val_path = os.path.join(SPLITS_DIR, "fixed_validation_split.csv")
    test_path = os.path.join(SPLITS_DIR, "fixed_test_split.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[+] Fixed splits saved to {SPLITS_DIR}:")
    print(f"    - Train Split: {len(train_df)} samples ({len(train_df)/total_len*100:.1f}%)")
    print(f"    - Validation Split: {len(val_df)} samples ({len(val_df)/total_len*100:.1f}%)")
    print(f"    - Unseen Test Split: {len(test_df)} samples ({len(test_df)/total_len*100:.1f}%)")
    print("=" * 70)

    return train_df, val_df, test_df


if __name__ == "__main__":
    create_fixed_splits()
