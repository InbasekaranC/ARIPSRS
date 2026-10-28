"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 7, 8, 9: Requirement Classifier & Candidate Filtering Module (requirement_classifier.py)

Evaluates candidate sentences using the trained TF-IDF + Logistic Regression model
in combination with requirement language obligation cues. Filters YES candidates into
rich internal VerifiedRequirement structures.
"""

import os
import re
from typing import List, Any, Optional, Tuple
from pydantic import BaseModel
import joblib
from sentence_splitter import SentenceCandidate

# Paths to trained model artifacts
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(BASE_DIR, "model", "requirement_classifier")
MODEL_PATH: str = os.path.join(MODEL_DIR, "classifier_pipeline.joblib")

# Global singleton model pipeline
_CLASSIFIER_PIPELINE: Optional[Any] = None


class VerifiedRequirement(BaseModel):
    """Rich internal requirement representation for downstream pipeline stages & future modules."""
    id: Optional[str] = None
    text: str  # Original clean requirement sentence text (NOT normalized)
    section_name: str
    page_num: int
    classification: str = "YES"
    confidence: float
    raw_sentence: str


def _load_classifier_model() -> Any:
    """
    Singleton loader for trained requirement classifier pipeline.
    """
    global _CLASSIFIER_PIPELINE
    if _CLASSIFIER_PIPELINE is not None:
        return _CLASSIFIER_PIPELINE

    if not os.path.exists(MODEL_PATH):
        print(f"[!] Classifier model artifact missing at {MODEL_PATH}. Executing fallback model training...")
        import train_requirement_classifier
        train_requirement_classifier.main()

    try:
        _CLASSIFIER_PIPELINE = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"[!] Exception loading classifier model: {e}")
        _CLASSIFIER_PIPELINE = None

    return _CLASSIFIER_PIPELINE


# Load model at module import time
_load_classifier_model()


def predict_sentence_requirement(sentence: str) -> Tuple[bool, float]:
    """
    Evaluates candidate sentence using trained ML classifier model combined with explicit requirement obligation cues.

    Args:
        sentence (str): Candidate sentence.

    Returns:
        Tuple[bool, float]: (is_requirement, confidence_score)
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return False, 0.0

    clean_sent = sentence.strip()
    pipeline = _load_classifier_model()

    # Check for explicit requirement obligation phrases
    has_explicit_obligation = bool(re.search(
        r'\b(shall|should|must|may|has\s+to|have\s+to|is\s+required\s+to|are\s+required\s+to|needs?\s+to)\b',
        clean_sent,
        re.IGNORECASE
    ))

    if pipeline is not None:
        try:
            pred_id = int(pipeline.predict([clean_sent])[0])
            probas = pipeline.predict_proba([clean_sent])[0]
            conf = float(probas[pred_id])
            if pred_id == 1:
                return True, conf
            elif has_explicit_obligation:
                # High-confidence explicit obligation phrase override
                return True, max(conf, 0.85)
            else:
                return False, conf
        except Exception as e:
            print(f"[!] Warning: ML prediction error: {e}")

    if has_explicit_obligation:
        return True, 0.85

    return False, 0.0


def classify_requirement(sentence: str) -> bool:
    """
    Classifies whether candidate sentence is a software requirement statement (YES/True) or not (NO/False).

    Args:
        sentence (str): Candidate sentence.

    Returns:
        bool: True if classified as a software requirement statement.
    """
    is_req, _ = predict_sentence_requirement(sentence)
    return is_req


def filter_candidates(candidates: List[SentenceCandidate]) -> List[VerifiedRequirement]:
    """
    Filters candidate sentence objects, retaining only those classified as requirements (YES).

    Args:
        candidates (List[SentenceCandidate]): List of sentence candidates from sentence_splitter.

    Returns:
        List[VerifiedRequirement]: Verified requirement objects.
    """
    if not candidates:
        return []

    verified: List[VerifiedRequirement] = []

    for cand in candidates:
        is_req, conf = predict_sentence_requirement(cand.text)
        if is_req:
            verified.append(
                VerifiedRequirement(
                    id=cand.id,
                    text=cand.text,
                    section_name=cand.section_name,
                    page_num=cand.page_num,
                    classification="YES",
                    confidence=conf,
                    raw_sentence=cand.raw_sentence
                )
            )

    return verified


def filter_requirements(sentences: List[str]) -> List[str]:
    """
    Backward-compatible string filter helper.

    Args:
        sentences (List[str]): Input string sentences.

    Returns:
        List[str]: Filtered requirement string sentences.
    """
    if not sentences:
        return []

    return [s.strip() for s in sentences if classify_requirement(s)]
