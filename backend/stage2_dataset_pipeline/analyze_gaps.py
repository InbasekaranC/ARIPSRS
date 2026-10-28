"""
ARIP Stage 2 Pipeline: Parts 5, 6, 7, 8 — Diversity Analysis & Dataset Gap Report (analyze_gaps.py)

Categorizes YES requirements into 18 functional/NFR categories and NO sentences into
12 non-requirement categories. Identifies category representation gaps and outputs
data/reports/dataset_gap_report.json.
"""

import os
import json
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_dataset.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "data", "reports")
GAP_REPORT_PATH = os.path.join(REPORTS_DIR, "dataset_gap_report.json")

# Target Distribution Goals (~3,000 YES / ~3,000 NO = ~6,000 total)
TARGET_YES_TOTAL = 3000
TARGET_NO_TOTAL = 3000

# Category Target Allocations for Expansion
YES_CATEGORY_TARGETS = {
    "Functional": 300,
    "Non-Functional": 200,
    "Security": 250,
    "Performance": 200,
    "Usability": 150,
    "Reliability": 150,
    "Availability": 150,
    "Data": 150,
    "API/Interface": 200,
    "Business Rules": 200,
    "Access Control": 150,
    "Authentication": 150,
    "Authorization": 150,
    "Reporting": 150,
    "Notification": 150,
    "Validation": 150,
    "Workflow": 100,
    "Integration": 100
}

NO_CATEGORY_TARGETS = {
    "Section Heading": 250,
    "Introduction": 200,
    "Background": 200,
    "System Overview": 250,
    "Figure/Table Caption": 200,
    "Glossary/Reference": 200,
    "Future Work": 150,
    "Implementation/Design Description": 350,
    "General/Historical Discussion": 250,
    "Document Metadata": 250,
    "Hard Negatives (Requirement-Like NO)": 500
}


def categorize_yes_sample(text: str) -> str:
    t = text.lower()
    if re.search(r'\b(authenticat|login|passkey|mfa|otp|biometric|token|jwt)\b', t):
        return "Authentication"
    elif re.search(r'\b(authoriz|role|permission|privilege|access\s+control|grant|restrict)\b', t):
        return "Access Control"
    elif re.search(r'\b(encrypt|hash|ssl|tls|cipher|security|audit\s+log|vulnerability|firewall)\b', t):
        return "Security"
    elif re.search(r'\b(second|ms|millisecond|minute|hour|per\s+second|throughput|latency|benchmark|cpu|memory|load)\b', t):
        return "Performance"
    elif re.search(r'\b(uptime|availability|99\.|24/7|fault\s+tolerant)\b', t):
        return "Availability"
    elif re.search(r'\b(backup|restore|disaster\s+recovery|failover|resilien|reliab)\b', t):
        return "Reliability"
    elif re.search(r'\b(ui|ux|screen|interface|accessibility|jaws|nvda|wcag|mobile\s+view|responsive)\b', t):
        return "Usability"
    elif re.search(r'\b(api|rest|json|xml|http|endpoint|webhook|soap|payload)\b', t):
        return "API/Interface"
    elif re.search(r'\b(report|dashboard|analytics|export|chart|csv|pdf\s+report|statistics)\b', t):
        return "Reporting"
    elif re.search(r'\b(notif|email|sms|alert|push|reminder|message)\b', t):
        return "Notification"
    elif re.search(r'\b(database|field|record|table|schema|storage|data|catalog|checksum)\b', t):
        return "Data"
    elif re.search(r'\b(calculat|fee|tax|interest|penalty|discount|business\s+rule|policy|rate|score|tier)\b', t):
        return "Business Rules"
    elif re.search(r'\b(validat|check|verify|format|regex|require)\b', t):
        return "Validation"
    elif re.search(r'\b(workflow|step|pipeline|stage|approval|process)\b', t):
        return "Workflow"
    elif re.search(r'\b(integrat|third-party|external|gateway|payment)\b', t):
        return "Integration"
    elif re.search(r'\b(shall|must|should|can|will|allow|provide|enable|support)\b', t):
        return "Functional"
    else:
        return "Non-Functional"


def categorize_no_sample(text: str) -> str:
    t = text.lower()
    if re.search(r'^(?:section|\d+\.|\d+\.\d+|1\.0|2\.0|3\.0|[A-Z0-9\s\_]{3,30}$)', text):
        return "Section Heading"
    elif re.search(r'\b(introduction|purpose|scope|overview of the document)\b', t):
        return "Introduction"
    elif re.search(r'\b(background|historical|existing\s+system|traditional|problem\s+statement)\b', t):
        return "Background"
    elif re.search(r'\b(figure\s+\d+|table\s+\d+|caption|shown\s+in\s+figure)\b', t):
        return "Figure/Table Caption"
    elif re.search(r'\b(glossary|definition|acronym|reference|appendix)\b', t):
        return "Glossary/Reference"
    elif re.search(r'\b(future\s+work|next\s+phase|planned|roadmap)\b', t):
        return "Future Work"
    elif re.search(r'\b(was\s+implemented|built\s+using|designed\s+with|developed\s+in|architecture|framework|uses\s+react|uses\s+python|uses\s+java|backend\s+module)\b', t):
        return "Implementation/Design Description"
    elif re.search(r'\b(author|guided\s+by|submitted\s+by|version\s*\d|be-\d+|b\.e\.|page\s+\d+)\b', t):
        return "Document Metadata"
    elif re.search(r'\b(is\s+an\s+important\s+feature|provides\s+functionality|allows\s+various|helps\s+users)\b', t):
        return "Hard Negatives (Requirement-Like NO)"
    else:
        return "General/Historical Discussion"


def analyze_gaps() -> dict:
    print("=" * 70)
    print("      ARIP STAGE 2: PARTS 5, 6, 7, 8 — DIVERSITY & GAP ANALYSIS      ")
    print("=" * 70)
    print(f"[+] Loading baseline cleaned dataset from: {CLEANED_CSV_PATH}")

    df = pd.read_csv(CLEANED_CSV_PATH)

    yes_df = df[df["label"] == "YES"].copy()
    no_df = df[df["label"] == "NO"].copy()

    yes_df["category"] = yes_df["original_text"].apply(categorize_yes_sample)
    no_df["category"] = no_df["original_text"].apply(categorize_no_sample)

    yes_counts = yes_df["category"].value_counts().to_dict()
    no_counts = no_df["category"].value_counts().to_dict()

    yes_gap_summary = {}
    for cat, target in YES_CATEGORY_TARGETS.items():
        curr = yes_counts.get(cat, 0)
        gap = max(0, target - curr)
        yes_gap_summary[cat] = {
            "current_count": curr,
            "target_count": target,
            "expansion_gap": gap
        }

    no_gap_summary = {}
    for cat, target in NO_CATEGORY_TARGETS.items():
        curr = no_counts.get(cat, 0)
        gap = max(0, target - curr)
        no_gap_summary[cat] = {
            "current_count": curr,
            "target_count": target,
            "expansion_gap": gap
        }

    gap_report = {
        "summary": {
            "total_cleaned_baseline": len(df),
            "current_yes_count": len(yes_df),
            "target_yes_count": TARGET_YES_TOTAL,
            "total_yes_gap": sum(v["expansion_gap"] for v in yes_gap_summary.values()),
            "current_no_count": len(no_df),
            "target_no_count": TARGET_NO_TOTAL,
            "total_no_gap": sum(v["expansion_gap"] for v in no_gap_summary.values()),
        },
        "yes_categories_gap": yes_gap_summary,
        "no_categories_gap": no_gap_summary
    }

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(GAP_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(gap_report, f, indent=4)

    print(f"[+] Gap analysis complete. Saved report to: {GAP_REPORT_PATH}")
    print(f"    - Current Cleaned Baseline: {len(df)} (YES: {len(yes_df)}, NO: {len(no_df)})")
    print(f"    - Total Expansion Gap Required: YES +{gap_report['summary']['total_yes_gap']} / NO +{gap_report['summary']['total_no_gap']}")
    print("=" * 70)

    return gap_report


if __name__ == "__main__":
    analyze_gaps()
