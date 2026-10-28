"""
AI Requirements Intelligence Platform (ARIP)
PDF Extraction Module (pdf_extractor.py) - PyMuPDF Layout-Aware Migration

Handles structured, layout-aware text extraction from PDF documents using PyMuPDF (fitz).
Extracts page dimensions, text blocks, block order, line counts, and bounding box coordinates
to support downstream layout-aware header/footer removal and section detection.
"""

import io
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
import pymupdf  # PyMuPDF


class PDFExtractionError(Exception):
    """Custom exception raised when PDF text extraction fails due to corruption or formatting issues."""
    pass


class TextBlock(BaseModel):
    """Structured text block extracted from a PDF page."""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    block_num: int
    page_num: int
    line_count: int


class PageLayoutData(BaseModel):
    """Structured layout representation of a single PDF page."""
    page_num: int
    width: float
    height: float
    blocks: List[TextBlock]


def extract_layout_from_pdf(file_bytes: bytes) -> List[PageLayoutData]:
    """
    Extracts structured page and block-level layout data from a PDF binary stream using PyMuPDF.

    Args:
        file_bytes (bytes): Raw binary content of the uploaded PDF file.

    Returns:
        List[PageLayoutData]: List of PageLayoutData objects containing bounding boxes and block text.

    Raises:
        PDFExtractionError: If the PDF is empty, corrupted, unreadable, or contains no readable pages.
    """
    if not file_bytes:
        raise PDFExtractionError("PDF file content is empty.")

    pages_layout: List[PageLayoutData] = []

    try:
        # Load PDF document from binary BytesIO stream using PyMuPDF
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        if doc.page_count == 0:
            doc.close()
            raise PDFExtractionError("PDF document contains no readable pages.")

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            rect = page.rect
            page_width = float(rect.width)
            page_height = float(rect.height)

            # Extract text blocks with layout structure (flag 0 includes text blocks with bbox)
            raw_blocks = page.get_text("blocks")
            blocks: List[TextBlock] = []

            for b_idx, b in enumerate(raw_blocks):
                # b format: (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) >= 5 and b[6] == 0:  # block_type 0 = text block
                    x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    block_text = b[4].strip()

                    if block_text:
                        lines = [line.strip() for line in block_text.splitlines() if line.strip()]
                        line_count = len(lines)
                        blocks.append(
                            TextBlock(
                                text="\n".join(lines),
                                bbox=(x0, y0, x1, y1),
                                block_num=b_idx,
                                page_num=page_index + 1,
                                line_count=line_count
                            )
                        )

            pages_layout.append(
                PageLayoutData(
                    page_num=page_index + 1,
                    width=page_width,
                    height=page_height,
                    blocks=blocks
                )
            )

        doc.close()

    except Exception as e:
        if isinstance(e, PDFExtractionError):
            raise e
        raise PDFExtractionError(f"Failed to parse PDF file with PyMuPDF: {str(e)}") from e

    # Check if any pages contain text blocks
    total_blocks = sum(len(p.blocks) for p in pages_layout)
    if total_blocks == 0:
        raise PDFExtractionError("No readable text blocks could be extracted from the PDF file.")

    return pages_layout


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Backward-compatible helper function extracting full raw text string from a PDF.

    Args:
        file_bytes (bytes): Raw binary content of the uploaded PDF file.

    Returns:
        str: Concatenated text content extracted from all readable pages.
    """
    layout_pages = extract_layout_from_pdf(file_bytes)
    page_texts = []
    for p in layout_pages:
        page_text = "\n".join(b.text for b in p.blocks if b.text.strip())
        if page_text.strip():
            page_texts.append(page_text)

    full_text = "\n".join(page_texts)
    if not full_text.strip():
        raise PDFExtractionError("No readable text could be extracted from the PDF file.")

    return full_text
