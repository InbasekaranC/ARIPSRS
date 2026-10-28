"""
AI Requirements Intelligence Platform (ARIP)
Requirement Extraction Module (requirement_extractor.py)

This module parses cleaned text to identify and extract software requirement statements.
It prioritizes sentences containing IEEE SRS modal verbs (shall, should, must, may) and
applies section-skipping heuristics to ignore non-requirement sections (Table of Contents,
Introduction, Scope, References, Glossary, Revision History, Appendices).
"""

import re
from typing import List

# IEEE SRS Modal Verbs pattern (word boundary matched, case-insensitive)
MODAL_VERBS_REGEX = re.compile(r'\b(shall|should|must|may)\b', re.IGNORECASE)

# Keywords indicating non-requirement sections to skip
NON_REQUIREMENT_KEYWORDS = [
    r'table\s+of\s+contents',
    r'introduction',
    r'purpose',
    r'scope',
    r'references',
    r'glossary',
    r'definitions',
    r'revision\s+history',
    r'document\s+history',
    r'appendix',
    r'appendices'
]

# Keywords indicating explicit requirement sections
REQUIREMENT_KEYWORDS = [
    r'requirements',
    r'functional\s+requirements',
    r'non-functional\s+requirements',
    r'system\s+features',
    r'specific\s+requirements',
    r'system\s+requirements',
    r'user\s+stories'
]

# Robust section heading matching pattern supporting section numbering like "1.", "1.1.", "A.1"
NON_REQ_HEADING_REGEX = re.compile(
    r'^(?:[\d.]+\s+)?(?:' + '|'.join(NON_REQUIREMENT_KEYWORDS) + r')\b',
    re.IGNORECASE
)

REQ_HEADING_REGEX = re.compile(
    r'^(?:[\d.]+\s+)?(?:' + '|'.join(REQUIREMENT_KEYWORDS) + r')\b',
    re.IGNORECASE
)


def extract_requirements(cleaned_text: str) -> List[str]:
    """
    Extracts software requirement statements from cleaned SRS text.

    Uses section-skipping heuristics to filter out non-requirement document sections
    (TOC, Introduction, Scope, etc.) and searches for candidate requirement sentences
    containing IEEE modal verbs ('shall', 'should', 'must', 'may').

    Args:
        cleaned_text (str): Cleaned multi-line text from document.

    Returns:
        List[str]: List of extracted software requirement strings.
    """
    if not isinstance(cleaned_text, str) or not cleaned_text.strip():
        return []

    lines = cleaned_text.splitlines()
    extracted_requirements: List[str] = []
    seen_requirements = set()

    # Section tracking heuristic state
    # Note for Viva: Section detection uses keyword-matching heuristics on line headers.
    # It is not a full semantic document parser, but effectively skips obvious non-requirement sections.
    in_non_requirement_section = False

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check if current line is a non-requirement section heading
        if NON_REQ_HEADING_REGEX.match(line_clean):
            in_non_requirement_section = True
            continue

        # Check if current line switches back to a requirement section heading
        if REQ_HEADING_REGEX.match(line_clean):
            in_non_requirement_section = False
            continue

        # If currently inside a non-requirement section, skip processing line
        if in_non_requirement_section:
            continue

        # Split line into candidate sentences if multiple sentences exist in a single line
        # e.g., "1.1 System performance. The system shall process requests within 2s."
        sentences = re.split(r'(?<=[.!?])\s+', line_clean)

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            # Filter for IEEE modal verbs: shall, should, must, may (word-boundary matched)
            if MODAL_VERBS_REGEX.search(sentence_clean):
                # Clean prefix artifacts like "1.1 ", "REQ-01: ", "* ", "- "
                cleaned_stmt = re.sub(r'^(?:REQ-\d+:?|[\d.]+|\*|-)\s*', '', sentence_clean, flags=re.IGNORECASE).strip()
                if not cleaned_stmt:
                    cleaned_stmt = sentence_clean

                # Deduplicate and append statement to extracted requirements
                normalized_stmt = cleaned_stmt.lower()
                if normalized_stmt not in seen_requirements:
                    seen_requirements.add(normalized_stmt)
                    extracted_requirements.append(cleaned_stmt)

    return extracted_requirements
