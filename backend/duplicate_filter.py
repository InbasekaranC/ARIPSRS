"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 10: Conservative Duplicate Removal Module (duplicate_filter.py)

Filters exact and near-exact duplicate requirement statements using RapidFuzz fuzzy similarity,
conservatively preserving requirement ordering and semantically distinct requirements.
"""

from typing import List
from rapidfuzz import fuzz
from requirement_classifier import VerifiedRequirement

# Conservative tunable similarity threshold constant (0-100)
# High threshold (90.0) ensures only exact/near-exact duplicates are removed
DUPLICATE_SIMILARITY_THRESHOLD: float = 90.0


def deduplicate_verified_requirements(requirements: List[VerifiedRequirement]) -> List[VerifiedRequirement]:
    """
    Deduplicates VerifiedRequirement objects conservatively, preserving original order.

    Args:
        requirements (List[VerifiedRequirement]): Verified requirement objects.

    Returns:
        List[VerifiedRequirement]: Deduplicated requirement objects.
    """
    if not requirements:
        return []

    unique_reqs: List[VerifiedRequirement] = []

    for req in requirements:
        text_clean = req.text.strip()
        if not text_clean:
            continue

        is_duplicate = False
        for existing in unique_reqs:
            # Token set ratio similarity
            score = fuzz.token_set_ratio(text_clean.lower(), existing.text.lower())
            if score >= DUPLICATE_SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_reqs.append(req)

    return unique_reqs


def remove_duplicate_requirements(requirements: List[str]) -> List[str]:
    """
    Backward-compatible deduplication helper for string lists.

    Args:
        requirements (List[str]): List of candidate requirement strings.

    Returns:
        List[str]: Deduplicated list of strings.
    """
    if not requirements:
        return []

    unique_strings: List[str] = []

    for req in requirements:
        req_clean = req.strip()
        if not req_clean:
            continue

        is_duplicate = False
        for existing in unique_strings:
            score = fuzz.token_set_ratio(req_clean.lower(), existing.lower())
            if score >= DUPLICATE_SIMILARITY_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_strings.append(req_clean)

    return unique_strings
