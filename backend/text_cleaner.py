"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 2: Layout-Aware Text Cleaning Module (text_cleaner.py)

Cleans formatting artifacts from structured PDF page layout data (PyMuPDF).
Detects headers, footers, and page numbers using vertical margin positions
combined with multi-page repetition metrics. Preserves requirement IDs, section numbers,
and line break structures.
"""

import re
from typing import List, Set, Dict, Any
from pydantic import BaseModel
from pdf_extractor import PageLayoutData, TextBlock


class CleanedBlock(BaseModel):
    """Cleaned text block object carrying layout metadata."""
    text: str
    page_num: int
    bbox: tuple[float, float, float, float]
    is_header_or_footer: bool = False


def clean_extracted_layout(pages: List[PageLayoutData]) -> List[CleanedBlock]:
    """
    Cleans extracted page layout blocks using vertical bounding box position heuristics
    and multi-page repetition metrics while preserving line breaks (\n).

    Args:
        pages (List[PageLayoutData]): Page layout objects extracted by PyMuPDF.

    Returns:
        List[CleanedBlock]: Cleaned list of text blocks.
    """
    if not pages:
        return []

    page_count = len(pages)
    block_frequency: Dict[str, int] = {}
    page_number_pattern = re.compile(
        r'^(page\s+\d+(\s+of\s+\d+)?|\d+\s+of\s+\d+|\d+)$',
        re.IGNORECASE
    )

    # Pass 1: Count occurrences of short blocks appearing near vertical margins
    for page in pages:
        p_height = page.height if page.height > 0 else 792.0
        header_threshold = 60.0
        footer_threshold = p_height - 60.0

        for block in page.blocks:
            y0, y1 = block.bbox[1], block.bbox[3]
            single_line_str = " ".join(block.text.strip().split())

            # Count frequency if located in header margin (top 60pt) or footer margin (bottom 60pt)
            if (y0 <= header_threshold or y1 >= footer_threshold) and len(single_line_str) < 80:
                block_frequency[single_line_str] = block_frequency.get(single_line_str, 0) + 1

    cleaned_blocks: List[CleanedBlock] = []

    # Pass 2: Filter blocks
    for page in pages:
        p_height = page.height if page.height > 0 else 792.0
        header_threshold = 60.0
        footer_threshold = p_height - 60.0

        for block in page.blocks:
            y0, y1 = block.bbox[1], block.bbox[3]
            single_line_str = " ".join(block.text.strip().split())

            if not single_line_str:
                continue

            # Strip standalone page numbers at page margins
            if (y0 <= header_threshold or y1 >= footer_threshold) and page_number_pattern.match(single_line_str):
                continue

            # Strip repeated running headers or footers near page margins
            if (y0 <= header_threshold or y1 >= footer_threshold) and len(single_line_str) < 80:
                if page_count > 1 and block_frequency.get(single_line_str, 0) >= 2:
                    continue

            # Clean whitespace line by line to preserve \n
            raw_lines = block.text.splitlines()
            cleaned_lines = [" ".join(l.strip().split()) for l in raw_lines if l.strip()]
            multiline_text = "\n".join(cleaned_lines)

            # Retain legitimate text block
            cleaned_blocks.append(
                CleanedBlock(
                    text=multiline_text,
                    page_num=page.page_num,
                    bbox=block.bbox
                )
            )

    return cleaned_blocks


def clean_extracted_text(raw_text: str) -> str:
    """
    Backward-compatible fallback string cleaner for raw text inputs.

    Args:
        raw_text (str): Raw string text.

    Returns:
        str: Cleaned string text.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""

    lines = raw_text.splitlines()
    stripped_lines: List[str] = []
    page_number_pattern = re.compile(
        r'^(page\s+\d+(\s+of\s+\d+)?|\d+\s+of\s+\d+|\d+)$',
        re.IGNORECASE
    )

    for line in lines:
        cleaned_line = " ".join(line.strip().split())
        if not cleaned_line:
            continue
        if page_number_pattern.match(cleaned_line):
            continue
        stripped_lines.append(cleaned_line)

    if not stripped_lines:
        return ""

    line_frequency = {}
    for line in stripped_lines:
        line_frequency[line] = line_frequency.get(line, 0) + 1

    filtered_lines: List[str] = []
    seen_lines: Set[str] = set()

    for line in stripped_lines:
        if len(line) < 70 and line_frequency[line] >= 3:
            continue
        if line not in seen_lines:
            seen_lines.add(line)
            filtered_lines.append(line)

    return "\n".join(filtered_lines)
