"""
AI Requirements Intelligence Platform (ARIP)
Central SRS Requirement Extraction Pipeline Orchestrator (requirement_pipeline.py)

Coordinates the multi-stage high-precision extraction pipeline:
1. PyMuPDF (fitz) page/block layout extraction
2. Layout-aware text cleaning & header/footer removal
3. 3-Level Section Decision (INCLUDE / REVIEW / EXCLUDE) & boundary mapping with safe fallback
4. spaCy sentence segmentation with Requirement ID & glued heading separation
5. High-Precision Rule-Based Requirement Validation (Metadata, Feature Heading, Fragment Rejection)
6. Strict ML Requirement Classifier (TF-IDF + Logistic Regression)
7. Conservative duplicate removal (RapidFuzz threshold = 90.0)
8. Original text preservation for ambiguity prediction
9. Unlimited internal storage + 9-item UI response trimming
10. Comprehensive logging, debug mode diagnostics, and user-friendly error handling.
"""

import logging
from typing import List, Dict, Any, Optional
from pdf_extractor import extract_layout_from_pdf, PDFExtractionError, PageLayoutData
from text_cleaner import clean_extracted_layout, CleanedBlock
from section_detector import extract_sections_from_layout, ExtractedSection
from sentence_splitter import split_sections_into_candidates, SentenceCandidate
from requirement_validator import validate_requirement_candidate, ValidationResult
from requirement_classifier import filter_candidates, VerifiedRequirement, predict_sentence_requirement
from duplicate_filter import deduplicate_verified_requirements

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARIP.RequirementPipeline")

# Maximum requirements returned in UI contract payload
MAX_UI_REQUIREMENTS: int = 9


def process_srs(
    file_bytes: bytes,
    filename: Optional[str] = None,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Executes end-to-end high-precision SRS requirement extraction pipeline.

    Args:
        file_bytes (bytes): Binary PDF content.
        filename (Optional[str]): Original PDF filename for logging.
        debug_mode (bool): If True, returns detailed pipeline diagnostic metadata.

    Returns:
        Dict[str, Any]: Dictionary containing 'ui_requirements' (max 9), 'all_requirements' (full list),
                        status message, and optional 'debug_info'.

    Raises:
        PDFExtractionError: User-friendly error message if extraction fails at any stage.
    """
    doc_name = filename if filename else "Uploaded Document"
    logger.info(f"[SRS Extraction Pipeline] Starting extraction for: {doc_name}")

    # Stage 1: Extract page & block layout using PyMuPDF (fitz)
    try:
        pages_layout = extract_layout_from_pdf(file_bytes)
    except PDFExtractionError as e:
        logger.error(f"[Stage 1 Error] {doc_name}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"[Stage 1 Unexpected Error] {doc_name}: {str(e)}")
        raise PDFExtractionError("Failed to parse PDF file. The file may be corrupted or unreadable.") from e

    page_count = len(pages_layout)
    total_raw_blocks = sum(len(p.blocks) for p in pages_layout)

    # Stage 2: Layout-aware text cleaning & header/footer removal
    cleaned_blocks = clean_extracted_layout(pages_layout)
    cleaned_block_count = len(cleaned_blocks)

    if not cleaned_blocks:
        raise PDFExtractionError("No readable text could be extracted from this PDF document.")

    # Stage 3, 4, 5: 3-level section decision & boundary mapping (with safe fallback)
    sections = extract_sections_from_layout(cleaned_blocks)
    section_count = len(sections)

    # Stage 6: spaCy sentence boundary segmentation with Requirement ID preservation & glued heading separation
    candidates = split_sections_into_candidates(sections)
    candidate_count = len(candidates)

    if not candidates:
        raise PDFExtractionError(
            "No requirement sections or candidate statements could be identified in this document. "
            "Please verify that the uploaded file is a valid SRS document."
        )

    # Stage 7: Rule-Based Requirement Validation (Metadata, Feature Heading, Fragment Rejection)
    validated_candidates: List[SentenceCandidate] = []
    rejected_candidates_log: List[Dict[str, str]] = []

    for cand in candidates:
        val_res = validate_requirement_candidate(cand.text)
        if val_res.is_valid:
            # Update candidate text with clean validated text
            cand.text = val_res.clean_text
            validated_candidates.append(cand)
        else:
            rejected_candidates_log.append({
                "candidate": cand.raw_sentence,
                "reason": val_res.reason,
                "category": val_res.rejected_category or "Rejected"
            })
            logger.debug(f"[Rule Validator REJECT] {cand.raw_sentence} -> {val_res.reason}")

    validated_count = len(validated_candidates)

    if not validated_candidates:
        raise PDFExtractionError(
            "No valid software requirement statements were found after noise filtering. "
            "The document contains project metadata, headings, or overview text rather than individual requirements."
        )

    # Stage 8: Strict ML Requirement Classifier (TF-IDF + Logistic Regression)
    verified_reqs = filter_candidates(validated_candidates)
    verified_count = len(verified_reqs)
    ml_rejected_count = validated_count - verified_count

    if not verified_reqs:
        raise PDFExtractionError(
            "No software requirement statements could be recognized in this document. "
            "Please verify that the document contains functional or non-functional requirement statements."
        )

    # Stage 9: Conservative Duplicate Filtering (RapidFuzz threshold 90.0)
    deduplicated_reqs = deduplicate_verified_requirements(verified_reqs)
    dedup_count = verified_count - len(deduplicated_reqs)
    final_count = len(deduplicated_reqs)

    # Stage 10: Preserving original requirement text strings
    all_requirement_strings = [r.text for r in deduplicated_reqs]
    all_requirement_objects = [r.model_dump() for r in deduplicated_reqs]

    # Return all extracted requirements for full analysis
    ui_requirement_strings = all_requirement_strings

    # Stage 11: Structured Logging
    log_msg = (
        f"[SRS Extraction Pipeline Completed] Filename: {doc_name} | "
        f"Pages: {page_count} | Raw blocks: {total_raw_blocks} | Cleaned blocks: {cleaned_block_count} | "
        f"Sections: {section_count} | Candidate sentences: {candidate_count} | "
        f"Rule-Validated: {validated_count} (Rejected noise: {len(rejected_candidates_log)}) | "
        f"ML Verified (YES): {verified_count} | Duplicates removed: {dedup_count} | "
        f"Final extracted requirements: {final_count}"
    )
    logger.info(log_msg)

    # Status message
    status_msg = f"Extracted {final_count} requirement statement{'' if final_count == 1 else 's'} successfully."

    response_payload = {
        "ui_requirements": ui_requirement_strings,
        "all_requirements": all_requirement_strings,
        "requirement_objects": all_requirement_objects,
        "total_extracted_count": final_count,
        "message": status_msg
    }

    # Stage 12: Debug Mode Metadata
    if debug_mode:
        debug_info = {
            "filename": doc_name,
            "page_count": page_count,
            "total_raw_blocks": total_raw_blocks,
            "cleaned_block_count": cleaned_block_count,
            "detected_sections": [
                {
                    "name": s.section_name,
                    "classification": s.classification,
                    "start_page": s.start_page,
                    "block_count": len(s.blocks)
                } for s in sections
            ],
            "candidate_count": candidate_count,
            "rule_validated_count": validated_count,
            "rule_rejected_count": len(rejected_candidates_log),
            "rule_rejected_log": rejected_candidates_log,
            "ml_verified_count": verified_count,
            "duplicates_removed_count": dedup_count,
            "final_count": final_count
        }
        response_payload["debug_info"] = debug_info

    return response_payload
