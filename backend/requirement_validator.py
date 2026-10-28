"""
AI Requirements Intelligence Platform (ARIP)
Rule-Based Requirement Validation & Noise Filtering Module (requirement_validator.py)

Executes precision rule-based filtering before ML requirement classification:
1. Heading-Description Separation: Splits glued feature titles from trailing requirement text.
2. Metadata Filter: Rejects academic metadata, project descriptions, author lines, and TOC entries.
3. Feature Heading Filter: Rejects standalone feature headers (e.g. "RESERVE SEATS -", "CANCEL RESERVATION-").
4. Narrative & Overview Filter: Rejects transition discourse sentences, problem statements, and overview paragraphs.
5. Fragment / Incompleteness Filter: Rejects noun phrase fragments lacking verb/predicate structure.
"""

import re
from typing import Tuple, Optional, List
from pydantic import BaseModel
# Load spaCy NLP model for grammatical parsing (with serverless fallback)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


class ValidationResult(BaseModel):
    """Result object detailing validation status, cleaned text, and explainability reason."""
    is_valid: bool
    clean_text: str
    rejected_category: Optional[str] = None
    reason: str


# -----------------------------------------------------------------------------
# DATA-DRIVEN REGEX PATTERNS & CONFIGURABLE REJECTION LISTS
# -----------------------------------------------------------------------------

# Academic metadata, institution tags, semester labels, and author lines
ACADEMIC_METADATA_PATTERNS_CASE_SENSITIVE = [
    re.compile(r'\b(?:B\.E\.|B\.Tech|M\.Tech|B\.Sc|M\.Sc|BE-\d+|BE\s*\d+|CE-\d+|IT-\d+)\b'),
]

ACADEMIC_METADATA_PATTERNS_CASE_INSENSITIVE = [
    re.compile(r'\b(?:Bachelor\s+of|Master\s+of|Semester\s*\d+|Department\s+of\s+Computer|Computer\s+Engineering|Faculty\s+of|University|College\s+of|Institute\s+of)\b', re.IGNORECASE),
    re.compile(r'^(?:submitted\s+by|guided\s+by|prepared\s+by|author|version\s*\d|academic\s*year|project\s+report|mini\s+project|major\s+project|under\s+the\s+guidance|guided\s+by:)', re.IGNORECASE),
    re.compile(r'^(?:course|branch|subject|roll\s*no|enrollment\s*no|student\s*id|class:)', re.IGNORECASE)
]

# Project overview descriptions, generic intro statements, and textbook narrative discussions
PROJECT_DESCRIPTION_PATTERNS = [
    re.compile(r'^(?:a\s+software\s+has\s+to\s+be\s+developed|a\s+system\s+is\s+to\s+be\s+developed|this\s+project\s+aims\s+to|this\s+document\s+(?:describes|specifies|outlines)|the\s+objective\s+of\s+this\s+(?:project|document)|this\s+software\s+is\s+developed\s+for)', re.IGNORECASE),
    re.compile(r'^(?:the\s+main\s+aim\s+of\s+this\s+project|the\s+proposed\s+system\s+is\s+designed\s+to\s+automate|this\s+system\s+helps\s+in\s+automating)', re.IGNORECASE)
]

# Narrative transition words, discourse markers, and general overview discussion patterns
NARRATIVE_DISCOURSE_PATTERNS = [
    re.compile(r'^(?:thus|finally|hence|therefore|however|furthermore|moreover|for\s+example|in\s+summary|a\s+few\s+factors|the\s+following\s+points|as\s+(?:discussed|mentioned)\s+above|in\s+this\s+section|somehow\s+the|there\s+are\s+three\s+major|a\s+few\s+factors\s+that\s+direct)\b', re.IGNORECASE),
    re.compile(r'\b(?:developer\s+does\s+not\s+understand|parties\s+interested\s+in\s+a\s+new\s+system|direct\s+us\s+to\s+develop|client,\s*the\s*user,\s*and\s*the\s*developer|concerns\s+of\s+the\s+users\s+have\s+to\s+be|manual\s+system\s+automation)\b', re.IGNORECASE)
]

# Document navigation, TOC, table headers, and reference patterns
TOC_DOCUMENT_PATTERNS = [
    re.compile(r'^(?:table\s+of\s+contents|figure\s+\d+|table\s+\d+|page\s+\d+|references?|appendix\s+[a-z]|glossary|revision\s+history)', re.IGNORECASE),
    re.compile(r'^\d+(\.\d+)*\s*$', re.IGNORECASE)
]

# Requirement obligation modal verbs and action cues
MODAL_OBLIGATION_PATTERNS = re.compile(
    r'\b(shall|should|must|may|can|will|has\s+to|have\s+to|is\s+required\s+to|are\s+required\s+to|needs?\s+to)\b',
    re.IGNORECASE
)


def separate_heading_from_description(text: str) -> Tuple[Optional[str], str]:
    """
    Separates a glued feature heading prefix from trailing requirement text.
    Example: "VIEW RESERVATION STATUS-The user need to enter the PIN number printed on ticket."
    Returns: ("VIEW RESERVATION STATUS", "The user need to enter the PIN number printed on ticket.")

    Args:
        text (str): Input sentence string.

    Returns:
        Tuple[Optional[str], str]: (extracted_heading, clean_requirement_text)
    """
    clean_raw = text.strip()
    if not clean_raw:
        return None, ""

    # Regex matching "HEADING TITLE - Requirement sentence..."
    match = re.match(r'^([A-Z0-9\s\_]{3,50})\s*[:\-\–\—]\s*([A-Z][a-z0-9\s\.\,\;].{15,})$', clean_raw)
    if match:
        heading = match.group(1).strip()
        requirement_part = match.group(2).strip()
        return heading, requirement_part

    return None, clean_raw


def is_metadata(text: str) -> Tuple[bool, str]:
    """
    Checks if text is document/academic metadata, author information, or project overview description.

    Args:
        text (str): Input sentence string.

    Returns:
        Tuple[bool, str]: (is_metadata_flag, reason)
    """
    clean = text.strip()
    if not clean:
        return True, "Empty text"

    # Check case-sensitive academic patterns (e.g. BE-6)
    for pat in ACADEMIC_METADATA_PATTERNS_CASE_SENSITIVE:
        if pat.search(clean):
            return True, "Academic Metadata"

    # Check case-insensitive academic patterns
    for pat in ACADEMIC_METADATA_PATTERNS_CASE_INSENSITIVE:
        if pat.search(clean):
            return True, "Academic/Document Metadata"

    # Check project description patterns
    for pat in PROJECT_DESCRIPTION_PATTERNS:
        if pat.search(clean):
            return True, "Project Description Narrative"

    # Check TOC / Document reference patterns
    for pat in TOC_DOCUMENT_PATTERNS:
        if pat.search(clean):
            return True, "Table of Contents or Document Reference"

    return False, "Not Metadata"


def is_narrative_discussion(text: str) -> Tuple[bool, str]:
    """
    Checks if text is a general narrative discourse, problem discussion, or intro sentence.

    Args:
        text (str): Input sentence string.

    Returns:
        Tuple[bool, str]: (is_narrative_flag, reason)
    """
    clean = text.strip()
    if not clean:
        return True, "Empty text"

    for pat in NARRATIVE_DISCOURSE_PATTERNS:
        if pat.search(clean):
            return True, "Narrative Discourse / Overview Discussion"

    return False, "Not Narrative Discourse"


def is_feature_heading(text: str) -> Tuple[bool, str]:
    """
    Detects short standalone feature headings (e.g., "RESERVE SEATS -", "CANCEL RESERVATION-", "LOGIN").
    Uses multi-signal cues (all-caps/title-case, short length, trailing delimiter, absence of requirement verb).

    Args:
        text (str): Input sentence string.

    Returns:
        Tuple[bool, str]: (is_feature_heading_flag, reason)
    """
    clean = text.strip()
    if not clean:
        return True, "Empty text"

    words = clean.split()
    word_count = len(words)

    # Signal 1: Ends with trailing dash or colon (e.g. "RESERVE SEATS -", "CANCEL RESERVATION-")
    ends_with_delimiter = bool(re.search(r'[:\-\–\—]$', clean))
    clean_no_delim = re.sub(r'[:\-\–\—]$', '', clean).strip()

    # Signal 2: Uppercase or Short Title Case
    is_uppercase = clean_no_delim.isupper()
    is_short_title = clean_no_delim.istitle() and word_count <= 4

    # Signal 3: Lacks requirement modal verb cues
    has_modal_verb = bool(MODAL_OBLIGATION_PATTERNS.search(clean))

    if (ends_with_delimiter or is_uppercase or is_short_title) and word_count <= 6 and not has_modal_verb:
        return True, f"Feature Heading ('{clean}')"

    return False, "Not Feature Heading"


def is_incomplete_fragment(text: str) -> Tuple[bool, str]:
    """
    Detects incomplete sentence fragments lacking subject+verb requirement structure
    (e.g., "Reservation against Cancellation.").

    Args:
        text (str): Input sentence string.

    Returns:
        Tuple[bool, str]: (is_fragment_flag, reason)
    """
    clean = text.strip()
    if not clean:
        return True, "Empty text"

    # Check explicitly for modal obligation phrases
    if MODAL_OBLIGATION_PATTERNS.search(clean):
        return False, "Contains Requirement Modal Verb"

    # Parse sentence with spaCy to inspect POS tags and grammatical structure
    if nlp is not None:
        doc = nlp(clean)
        verbs = [token for token in doc if token.pos_ in ("VERB", "AUX")]

        # If sentence contains no verbs or auxiliary verbs, it is a noun phrase fragment
        if not verbs:
            return True, "Incomplete Fragment: Lacks verb/predicate structure"

        # Check if string is a short noun phrase ending with noun/preposition without main predicate
        if len(doc) <= 5 and doc[-1].pos_ in ("NOUN", "PROPN", "ADP"):
            has_nsubj = any(tok.dep_ in ("nsubj", "nsubjpass", "agent") for tok in doc)
            if not has_nsubj:
                return True, "Incomplete Fragment: Noun phrase without active subject-predicate relation"
    else:
        words = clean.split()
        if len(words) <= 4 and not re.search(r'\b(is|are|was|were|be|been|has|have|had|do|does|did|allow|allows|provide|provides|support|supports|enable|enables|display|displays|store|stores|process|processes|send|sends|generate|generates|verify|verifies|validate|validates)\b', clean, re.IGNORECASE):
            return True, "Incomplete Fragment: Lacks verb/predicate structure"

    return False, "Grammatically Complete Clause"


def validate_requirement_candidate(text: str) -> ValidationResult:
    """
    Executes end-to-end rule-based validation on candidate requirement text:
    1. Heading-Description Separation
    2. Metadata Filtering
    3. Narrative & Discourse Filtering
    4. Feature Heading Filtering
    5. Fragment / Incompleteness Filtering

    Args:
        text (str): Input candidate sentence text.

    Returns:
        ValidationResult: Object containing is_valid status, cleaned text, and reason.
    """
    if not isinstance(text, str) or not text.strip():
        return ValidationResult(
            is_valid=False,
            clean_text="",
            rejected_category="Empty",
            reason="Input candidate text is empty."
        )

    # Step 1: Separate glued feature heading from trailing requirement text if present
    heading_prefix, clean_text = separate_heading_from_description(text)

    # Step 2: Check for Metadata
    metadata_flag, meta_reason = is_metadata(clean_text)
    if metadata_flag:
        return ValidationResult(
            is_valid=False,
            clean_text=clean_text,
            rejected_category="Metadata",
            reason=f"REJECT (metadata): {meta_reason}"
        )

    # Step 3: Check for Narrative Discourse & General Discussion
    narrative_flag, narr_reason = is_narrative_discussion(clean_text)
    if narrative_flag:
        return ValidationResult(
            is_valid=False,
            clean_text=clean_text,
            rejected_category="Narrative Discourse",
            reason=f"REJECT (narrative discourse): {narr_reason}"
        )

    # Step 4: Check for Feature Heading
    heading_flag, head_reason = is_feature_heading(clean_text)
    if heading_flag:
        return ValidationResult(
            is_valid=False,
            clean_text=clean_text,
            rejected_category="Feature Heading",
            reason=f"REJECT (feature heading): {head_reason}"
        )

    # Step 5: Check for Incomplete Fragment
    fragment_flag, frag_reason = is_incomplete_fragment(clean_text)
    if fragment_flag:
        return ValidationResult(
            is_valid=False,
            clean_text=clean_text,
            rejected_category="Incomplete Fragment",
            reason=f"REJECT (incomplete fragment): {frag_reason}"
        )

    # Passed all rule-based rejection filters
    return ValidationResult(
        is_valid=True,
        clean_text=clean_text,
        rejected_category=None,
        reason="Passed Rule-Based Requirement Validation"
    )
