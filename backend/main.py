"""
AI Requirements Intelligence Platform (ARIP)
Module 1: Requirement Ambiguity Detection
FastAPI Web Application Server (main.py)

This module defines RESTful HTTP API endpoints for analyzing software requirements for ambiguity,
extracting requirements from SRS PDF documents via a multi-stage NLP/ML pipeline, and generating PDF reports.
"""

from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import prediction service from predict.py
from predict import predict_requirement

# Import Conflict Detection module
from conflict_detector import detect_conflicts

# Import PDF extraction error and central requirement pipeline
from pdf_extractor import PDFExtractionError
from requirement_pipeline import process_srs

# Import PDF report generator module
from report.generate_report import generate_pdf_report, GenerateReportRequest

# Application Constants
MAX_REQUIREMENTS: int = 500

# Initialize FastAPI application
app = FastAPI(
    title="AI Requirements Intelligence Platform (ARIP)",
    description="API server for Module 1: Requirement Ambiguity Detection, SRS PDF Extraction, and PDF Report Generation.",
    version="1.4.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure Cross-Origin Resource Sharing (CORS) for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin during development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS (Request & Response Data Transfer Objects)
# -----------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Payload model containing a list of software requirement sentences (max 9)."""
    requirements: List[str] = Field(
        ...,
        description="List of requirement text strings to analyze. Max 9 requirements allowed per request.",
        example=[
            "The system shall allow users to reset passwords within 2 minutes.",
            "The application should be fast and user-friendly."
        ]
    )


class AnalysisResult(BaseModel):
    """Result object for a single analyzed requirement sentence."""
    requirement: str = Field(..., description="Original requirement input text.")
    prediction: str = Field(..., description="Classification result: 'Clear' or 'Ambiguous'.")
    confidence: str = Field(..., description="Model confidence score percentage (e.g. '94%').")
    reason: str = Field(..., description="Explainability reasoning for the prediction.")


class ConflictItemResult(BaseModel):
    """Data model for an itemized requirement conflict pair."""
    requirement_1_id: str = Field(..., description="Stable ID of first requirement (e.g. REQ-001).")
    requirement_1_text: str = Field(..., description="Text of first requirement.")
    requirement_2_id: str = Field(..., description="Stable ID of second requirement (e.g. REQ-014).")
    requirement_2_text: str = Field(..., description="Text of second requirement.")
    conflict_detected: bool = Field(True, description="Conflict status boolean flag.")
    conflict_type: str = Field(..., description="Conflict type label (e.g. Direct Contradiction).")
    severity: str = Field(..., description="Severity level: HIGH, MEDIUM, or LOW.")
    confidence: float = Field(..., description="Model/evidence confidence percentage e.g. 94.6.")
    similarity_score: float = Field(..., description="Cosine similarity score e.g. 0.87.")
    explanation: str = Field(..., description="Human-readable explainability reasoning.")
    evidence: List[str] = Field(..., description="List of evidence points supporting conflict classification.")


class ConflictSummaryResult(BaseModel):
    """Data model summarizing conflict detection overall metrics."""
    total_requirements: int
    total_candidate_pairs: int
    conflicts_found: int
    severity_counts: dict


class AnalyzeResponse(BaseModel):
    """Response wrapper containing results list for all submitted requirements and conflict detection."""
    results: List[AnalysisResult]
    conflicts: List[ConflictItemResult] = Field(default=[], description="Detected requirement conflict pairs.")
    conflict_summary: Optional[ConflictSummaryResult] = Field(default=None, description="Conflict detection summary metrics.")


class DetectConflictsResponse(BaseModel):
    """Response wrapper for dedicated /detect-conflicts endpoint."""
    total_requirements: int
    total_candidate_pairs: int
    conflicts_found: int
    severity_counts: dict
    conflicts: List[ConflictItemResult]


class ExtractPdfResponse(BaseModel):
    """Response model for PDF requirement extraction endpoint."""
    requirements: List[str] = Field(
        ...,
        description="Extracted list of software requirement strings (max 9)."
    )
    message: str = Field(
        ...,
        description="Status message detailing extraction output."
    )


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Health"])
def health_check():
    """
    Health check endpoint to verify backend service status.
    """
    return {
        "status": "ok",
        "platform": "AI Requirements Intelligence Platform (ARIP)",
        "active_module": "Module 1: Requirement Ambiguity Detection",
        "version": "1.4.0"
    }


@app.post(
    "/extract-pdf",
    response_model=ExtractPdfResponse,
    status_code=status.HTTP_200_OK,
    tags=["SRS Document Extraction"],
    summary="Extract requirement statements from uploaded SRS PDF document"
)
async def extract_pdf_requirements(file: UploadFile = File(...), debug: bool = False):
    """
    Accepts an uploaded SRS PDF file and delegates to requirement_pipeline.process_srs()
    for layout-aware, format-independent extraction, PyMuPDF parsing, 3-level section detection,
    spaCy sentence splitting, ML classification, and conservative deduplication.
    Returns up to 9 extracted requirements.
    """
    # Validation 1: Check if file object or filename is missing
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No PDF selected."
        )

    # Validation 2: Check file extension and content-type
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file."
        )

    # Read binary content of uploaded file
    try:
        file_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file."
        )

    # Validation 3: Check for empty file content
    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file."
        )

    # Delegate to central orchestrator module requirement_pipeline.py
    try:
        result = process_srs(file_bytes, filename=file.filename, debug_mode=debug)
    except PDFExtractionError as e:
        detail_msg = str(e) if str(e) else "No requirement statements found."
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail_msg
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file."
        )

    return ExtractPdfResponse(
        requirements=result["ui_requirements"],
        message=result["message"]
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Ambiguity Detection"],
    summary="Analyze software requirements for ambiguity"
)
def analyze_requirements(request: AnalyzeRequest):
    """
    Accepts a list of up to 9 requirement strings, runs ML ambiguity detection
    and explainability analysis, and returns prediction details.
    """
    raw_requirements = request.requirements

    # Validation 1: Check for empty requirements payload
    if not raw_requirements:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Requirements list cannot be empty. Please provide at least one requirement sentence."
        )

    # Filter out empty or whitespace-only requirements
    cleaned_requirements = [req.strip() for req in raw_requirements if req and req.strip()]

    # Validation 2: Check if any non-blank requirements exist
    if not cleaned_requirements:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Submitted requirements contain only blank or whitespace characters."
        )

    # Validation 3: Enforce frontend cap limit of 9 requirements
    if len(cleaned_requirements) > MAX_REQUIREMENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exceeded maximum limit of 9 requirements per analysis request."
        )

    # Execute predictions using predict_requirement from predict.py
    analysis_results: List[AnalysisResult] = []
    for req in cleaned_requirements:
        pred_dict = predict_requirement(req)
        analysis_results.append(
            AnalysisResult(
                requirement=pred_dict["requirement"],
                prediction=pred_dict["prediction"],
                confidence=pred_dict["confidence"],
                reason=pred_dict["reason"]
            )
        )

    # Execute Conflict Detection across submitted requirements
    conflict_payload = detect_conflicts(cleaned_requirements)
    conflict_items = [
        ConflictItemResult(**item) for item in conflict_payload.get("conflicts", [])
    ]
    conflict_summary = ConflictSummaryResult(
        total_requirements=conflict_payload.get("total_requirements", len(cleaned_requirements)),
        total_candidate_pairs=conflict_payload.get("total_candidate_pairs", 0),
        conflicts_found=conflict_payload.get("conflicts_found", 0),
        severity_counts=conflict_payload.get("severity_counts", {"HIGH": 0, "MEDIUM": 0, "LOW": 0})
    )

    return AnalyzeResponse(
        results=analysis_results,
        conflicts=conflict_items,
        conflict_summary=conflict_summary
    )


@app.post(
    "/detect-conflicts",
    response_model=DetectConflictsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Conflict Detection"],
    summary="Detect potential contradictions and conflicts among software requirements"
)
def detect_conflicts_endpoint(request: AnalyzeRequest):
    """
    Accepts a list of requirement statements and executes candidate pair generation,
    TF-IDF similarity matching, polarity/negation analysis, condition evaluation,
    and constraint checking to identify mutually contradictory requirement pairs.
    """
    raw_requirements = request.requirements

    if not raw_requirements:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Requirements list cannot be empty."
        )

    cleaned_requirements = [req.strip() for req in raw_requirements if req and req.strip()]

    if not cleaned_requirements:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Submitted requirements contain only blank characters."
        )

    conflict_payload = detect_conflicts(cleaned_requirements)
    return DetectConflictsResponse(
        total_requirements=conflict_payload.get("total_requirements", len(cleaned_requirements)),
        total_candidate_pairs=conflict_payload.get("total_candidate_pairs", 0),
        conflicts_found=conflict_payload.get("conflicts_found", 0),
        severity_counts=conflict_payload.get("severity_counts", {"HIGH": 0, "MEDIUM": 0, "LOW": 0}),
        conflicts=[ConflictItemResult(**item) for item in conflict_payload.get("conflicts", [])]
    )


@app.post(
    "/generate-report",
    status_code=status.HTTP_200_OK,
    tags=["PDF Report Generation"],
    summary="Generate downloadable SRS Requirement Ambiguity PDF Report"
)
def generate_report_endpoint(request: GenerateReportRequest):
    """
    Accepts current requirement analysis results and metadata, builds a multi-page
    PDF report using ReportLab, and returns it as a downloadable PDF file.
    """
    if not request.results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No analyzed requirements provided for report generation."
        )

    try:
        pdf_bytes = generate_pdf_report(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(e)}"
        )

    filename = "ARIP_Requirement_Ambiguity_Report.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
