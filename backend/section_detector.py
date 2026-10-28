"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 3, 4, 5: Robust 3-Level Section Detector & Boundary Manager (section_detector.py)

Parses cleaned layout blocks to identify section headings using fuzzy matching (rapidfuzz),
classifies sections into 3 levels (INCLUDE, REVIEW, EXCLUDE), and determines exact section boundaries.
"""

import re
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel
from rapidfuzz import fuzz
from text_cleaner import CleanedBlock

# -----------------------------------------------------------------------------
# CONFIGURABLE FUZZY MATCHING THRESHOLD & KEYWORD LIST CONSTANTS
# -----------------------------------------------------------------------------
FUZZY_HEADING_THRESHOLD: float = 75.0

# Level 1: INCLUDE Sections (Pass straight to sentence splitting)
INCLUDE_SECTION_KEYWORDS: List[str] = [
    "functional requirement",
    "functional requirements",
    "functional specification",
    "functional specifications",
    "non-functional requirement",
    "non-functional requirements",
    "nonfunctional requirement",
    "nonfunctional requirements",
    "specific requirement",
    "specific requirements",
    "detailed requirement",
    "detailed requirements",
    "system requirement",
    "system requirements",
    "software requirement",
    "software requirements",
    "business rule",
    "business rules",
    "system feature",
    "system features",
    "system features specification",
    "system function",
    "system functions",
    "system functionality",
    "use case",
    "use cases",
    "use case requirements",
    "requirements specification",
    "requirements analysis",
    "business requirements",
    "user requirement",
    "user requirements",
    "product requirement",
    "product requirements",
    "interface requirement",
    "interface requirements",
    "performance requirement",
    "performance requirements",
    "security requirement",
    "security requirements",
    "external interface requirements",
    "features and requirements",
    "system capabilities"
]

# Level 2: REVIEW Sections (Pass to sentence splitter and requirement validator)
REVIEW_SECTION_KEYWORDS: List[str] = [
    "product features",
    "user stories",
    "user story",
    "external interfaces",
    "constraints"
]

# Level 3: EXCLUDE Sections (Drop entirely - introduction, background, problem statements, metadata, glossary, appendices)
EXCLUDE_SECTION_KEYWORDS: List[str] = [
    "introduction",
    "purpose",
    "scope",
    "scope and overview",
    "background",
    "problem statement",
    "motivation",
    "existing system",
    "proposed system",
    "feasibility study",
    "document conventions",
    "intended audience",
    "user characteristics",
    "stakeholders",
    "system overview",
    "general description",
    "organization of document",
    "project overview",
    "software overview",
    "references",
    "glossary",
    "definitions",
    "acronyms",
    "abbreviations",
    "acknowledgement",
    "acknowledgements",
    "revision history",
    "document history",
    "table of contents",
    "list of figures",
    "list of tables",
    "preface",
    "appendix",
    "appendices"
]


class ExtractedSection(BaseModel):
    """Section metadata and constituent text blocks."""
    section_name: str
    classification: str  # "INCLUDE", "REVIEW", "EXCLUDE"
    start_page: int
    blocks: List[CleanedBlock]


def is_heading_candidate(line: str) -> bool:
    """
    Determines whether a block text line fits section heading formatting heuristics.
    Headings are concise lines (<= 8 words and <= 90 characters).

    Args:
        line (str): Text string.

    Returns:
        bool: True if line resembles a section heading.
    """
    clean = line.strip()
    if not clean or len(clean) > 90:
        return False

    words = clean.split()
    if len(words) > 8:
        return False

    # Standard section numbering regex (e.g. "1.", "3.1", "SECTION 3", "3.1.2", "FR-1")
    if re.match(r'^(?:section\s+)?\d+(?:\.\d+)*\.?\s+[A-Z]', clean, re.IGNORECASE):
        return True

    # Check for short title-case or uppercase lines
    if clean.isupper() or clean.istitle():
        return True

    return False


def classify_heading_3level(line: str) -> Tuple[str, str]:
    """
    Classifies candidate heading into 3 decision levels (INCLUDE, REVIEW, EXCLUDE) using fuzzy matching.

    Args:
        line (str): Candidate heading text line.

    Returns:
        Tuple[str, str]: (classification_level, matched_keyword)
    """
    # Normalize heading text by stripping section numbering
    clean_line = re.sub(r'^(?:section\s+)?\d+(?:\.\d+)*\.?\s*', '', line.strip(), flags=re.IGNORECASE).lower()

    # Check EXCLUDE keywords first
    for kw in EXCLUDE_SECTION_KEYWORDS:
        if fuzz.token_set_ratio(clean_line, kw) >= FUZZY_HEADING_THRESHOLD:
            return "EXCLUDE", kw

    # Check INCLUDE keywords
    for kw in INCLUDE_SECTION_KEYWORDS:
        if fuzz.token_set_ratio(clean_line, kw) >= FUZZY_HEADING_THRESHOLD:
            return "INCLUDE", kw

    # Check REVIEW keywords
    for kw in REVIEW_SECTION_KEYWORDS:
        if fuzz.token_set_ratio(clean_line, kw) >= FUZZY_HEADING_THRESHOLD:
            return "REVIEW", kw

    # Default unrecognized heading to REVIEW to avoid accidental data loss
    return "REVIEW", "unknown"


def extract_sections_from_layout(blocks: List[CleanedBlock]) -> List[ExtractedSection]:
    """
    Parses layout blocks, checks every line for section headings, identifies section boundaries,
    and assigns 3-level section decisions (INCLUDE, REVIEW, EXCLUDE).

    Args:
        blocks (List[CleanedBlock]): Cleaned layout blocks from text_cleaner.

    Returns:
        List[ExtractedSection]: Partitioned section objects.
    """
    if not blocks:
        return []

    sections: List[ExtractedSection] = []
    current_name = "Document Body"
    current_classification = "REVIEW"
    current_start_page = 1
    current_blocks: List[CleanedBlock] = []
    headings_found = 0

    for block in blocks:
        lines = [l.strip() for l in block.text.splitlines() if l.strip()]
        if not lines:
            continue

        accumulated_lines: List[str] = []

        for line in lines:
            if is_heading_candidate(line):
                # Flush accumulated lines before this heading into the current section
                if accumulated_lines:
                    current_blocks.append(
                        CleanedBlock(
                            text="\n".join(accumulated_lines),
                            page_num=block.page_num,
                            bbox=block.bbox
                        )
                    )
                    accumulated_lines = []

                # Save current section if it contains blocks
                if current_blocks:
                    sections.append(
                        ExtractedSection(
                            section_name=current_name,
                            classification=current_classification,
                            start_page=current_start_page,
                            blocks=list(current_blocks)
                        )
                    )
                    current_blocks = []

                # Start new section
                classification, matched_kw = classify_heading_3level(line)
                headings_found += 1
                current_name = line
                current_classification = classification
                current_start_page = block.page_num
            else:
                accumulated_lines.append(line)

        if accumulated_lines:
            current_blocks.append(
                CleanedBlock(
                    text="\n".join(accumulated_lines),
                    page_num=block.page_num,
                    bbox=block.bbox
                )
            )

    # Save remaining trailing section
    if current_blocks:
        sections.append(
            ExtractedSection(
                section_name=current_name,
                classification=current_classification,
                start_page=current_start_page,
                blocks=current_blocks
            )
        )

    # Safe Fallback: If no explicit headings were detected, label section as UNSECTIONED_DOCUMENT
    # marked as REVIEW so downstream validator performs sentence-level requirement filtering.
    if headings_found == 0 or not sections:
        return [
            ExtractedSection(
                section_name="UNSECTIONED_DOCUMENT",
                classification="REVIEW",
                start_page=1,
                blocks=blocks
            )
        ]

    return sections


def extract_requirement_sections(cleaned_text: str) -> str:
    """
    Backward-compatible string helper for section extraction.

    Args:
        cleaned_text (str): Cleaned text.

    Returns:
        str: Section text.
    """
    if not isinstance(cleaned_text, str) or not cleaned_text.strip():
        return ""

    lines = cleaned_text.splitlines()
    req_lines = []
    in_section = True
    headers_found = 0

    for line in lines:
        clean = line.strip()
        if not clean:
            continue
        if is_heading_candidate(clean):
            classification, _ = classify_heading_3level(clean)
            headers_found += 1
            if classification == "EXCLUDE":
                in_section = False
            else:
                in_section = True
            continue
        if in_section:
            req_lines.append(clean)

    if not req_lines:
        return cleaned_text

    return "\n".join(req_lines)
