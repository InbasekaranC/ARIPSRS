"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 6: spaCy Sentence Splitting & Requirement ID Preservation (sentence_splitter.py)

Segments section text into individual candidate sentences using spaCy NLP.
Extracts and preserves requirement identifiers (e.g. FR-01, REQ-001, NFR-02, 3.2.1)
into structured candidate objects while stripping IDs from the natural-language text.
Separates glued feature title prefixes from trailing requirement text.
"""

import re
from typing import List, Optional, Tuple
from pydantic import BaseModel
from section_detector import ExtractedSection, CleanedBlock
from requirement_validator import separate_heading_from_description

# Singleton spaCy model instance (with serverless regex fallback)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def _get_sentences(text: str) -> List[str]:
    """Splits text into sentences using spaCy if available, otherwise regex sentence boundary splitting."""
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    parts = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9\[\(])|(?=\b(?:REQ|FR|NFR|BR|UC|SR|SYS|FUN|NFUN)-\d+(?:\.\d+)*[:\.\s\-])',
        text
    )
    return [p.strip() for p in parts if p and p.strip()]


class SentenceCandidate(BaseModel):
    """Structured candidate sentence object with ID preservation."""
    id: Optional[str] = None
    text: str
    section_name: str
    page_num: int
    raw_sentence: str


# Regex matching requirement IDs e.g. "FR-01:", "REQ-001 -", "NFR-02.", "3.2.1 "
ID_REGEX = re.compile(
    r'^(?:\[?((?:REQ|FR|NFR|BR|UC|SR|SYS|FUN|NFUN)-\d+(?:\.\d+)*|(?:[0-9]+\.){1,4}[0-9]+)\]?[:\.\s\-]+)',
    re.IGNORECASE
)


def extract_requirement_id(sentence_str: str) -> Tuple[Optional[str], str]:
    """
    Extracts requirement ID prefix from sentence string and returns clean natural-language text.

    Args:
        sentence_str (str): Input raw sentence.

    Returns:
        Tuple[Optional[str], str]: (extracted_id, clean_text)
    """
    clean_raw = sentence_str.strip()
    match = ID_REGEX.match(clean_raw)

    if match:
        req_id = match.group(1).upper()
        clean_text = clean_raw[match.end():].strip()
        # Fallback if clean_text is empty
        if not clean_text:
            clean_text = clean_raw
        return req_id, clean_text

    # Strip simple bullet points or dash markers
    clean_text = re.sub(r'^[\*\-\•\–]\s*', '', clean_raw).strip()
    return None, clean_text if clean_text else clean_raw


def split_sections_into_candidates(sections: List[ExtractedSection]) -> List[SentenceCandidate]:
    """
    Splits text blocks inside sections into candidate sentence objects with preserved IDs.
    Separates glued feature headings from trailing requirement descriptions.

    Args:
        sections (List[ExtractedSection]): List of extracted sections.

    Returns:
        List[SentenceCandidate]: Candidate sentence objects.
    """
    if not sections:
        return []

    candidates: List[SentenceCandidate] = []

    for section in sections:
        # Skip EXCLUDE sections
        if section.classification == "EXCLUDE":
            continue

        for block in section.blocks:
            lines = [l.strip() for l in block.text.splitlines() if l.strip()]
            block_text = " ".join(lines)

            if not block_text:
                continue

            for raw_sent in _get_sentences(block_text):
                if not raw_sent:
                    continue

                # Step 1: Extract requirement ID prefix if present
                req_id, clean_text = extract_requirement_id(raw_sent)

                # Step 2: Separate glued feature heading from trailing requirement text
                _, separated_text = separate_heading_from_description(clean_text)

                # Filter out short fragments (< 10 chars)
                if len(separated_text) >= 10:
                    candidates.append(
                        SentenceCandidate(
                            id=req_id,
                            text=separated_text,
                            section_name=section.section_name,
                            page_num=block.page_num,
                            raw_sentence=raw_sent
                        )
                    )

    return candidates


def split_into_sentences(section_text: str) -> List[str]:
    """
    Backward-compatible string helper for sentence splitting.

    Args:
        section_text (str): Input text string.

    Returns:
        List[str]: List of sentence strings.
    """
    if not isinstance(section_text, str) or not section_text.strip():
        return []

    lines = [l.strip() for l in section_text.splitlines() if l.strip()]
    processed_text = " ".join(lines)
    candidate_sentences: List[str] = []

    for sent_str in _get_sentences(processed_text):
        if not sent_str:
            continue

        _, clean_text = extract_requirement_id(sent_str)
        _, separated_text = separate_heading_from_description(clean_text)
        if len(separated_text) >= 10:
            candidate_sentences.append(separated_text)

    return candidate_sentences
