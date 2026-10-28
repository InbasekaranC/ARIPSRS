"""
AI Requirements Intelligence Platform (ARIP)
Module 1: Requirement Ambiguity Detection
Report Generation Module (generate_report.py)

This module handles multi-page PDF report generation using ReportLab.
It consumes pre-computed requirement analysis results from Phase 1-3 without re-running
predictions. Each report section is built in a modular, extensible function to allow
seamless addition of future ARIP report sections (Conflict Detection, Quality Score, etc.).
"""

import io
import re
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

# -----------------------------------------------------------------------------
# CONFIGURABLE ACADEMIC & INSTITUTIONAL PLACEHOLDER CONSTANTS
# -----------------------------------------------------------------------------
UNIVERSITY_NAME: str = "Visvesvaraya Technological University"
DEPARTMENT_NAME: str = "Department of Computer Science & Engineering"
STUDENT_NAME: str = "Final Year Engineering Project Team"
GUIDE_NAME: str = "Project Guide / Faculty Advisor"

# -----------------------------------------------------------------------------
# QUALITY SCORE THRESHOLD CONSTANTS
# -----------------------------------------------------------------------------
SCORE_EXCELLENT_THRESHOLD: int = 90
SCORE_GOOD_THRESHOLD: int = 80
SCORE_FAIR_THRESHOLD: int = 70

# -----------------------------------------------------------------------------
# BRAND DESIGN CONSTANTS
# -----------------------------------------------------------------------------
COLOR_PRIMARY_BLUE = colors.HexColor("#2563eb")
COLOR_PRIMARY_DARK = colors.HexColor("#0f172a")
COLOR_TEXT_MUTED = colors.HexColor("#64748b")
COLOR_BG_LIGHT = colors.HexColor("#f8fafc")
COLOR_BORDER = colors.HexColor("#e2e8f0")
COLOR_CLEAR_TEXT = colors.HexColor("#166534")
COLOR_CLEAR_BG = colors.HexColor("#dcfce7")
COLOR_AMBIGUOUS_TEXT = colors.HexColor("#991b1b")
COLOR_AMBIGUOUS_BG = colors.HexColor("#fee2e2")
COLOR_WARNING_TEXT = colors.HexColor("#92400e")


# -----------------------------------------------------------------------------
# DATA MODELS
# -----------------------------------------------------------------------------

class RequirementItemInput(BaseModel):
    """Data model for a single analyzed requirement input item."""
    requirement: str = Field(..., description="Requirement sentence text.")
    prediction: str = Field(..., description="Classification result ('Clear' or 'Ambiguous').")
    confidence: str = Field(..., description="Model confidence score percentage e.g. '95%'.")
    reason: str = Field(..., description="Explainability reasoning string.")


class ConflictItemInput(BaseModel):
    """Data model for a detected requirement conflict pair."""
    requirement_1_id: str
    requirement_1_text: str
    requirement_2_id: str
    requirement_2_text: str
    conflict_detected: bool = True
    conflict_type: str
    severity: str
    confidence: float
    similarity_score: float
    explanation: str
    evidence: List[str] = Field(default=[])


class GenerateReportRequest(BaseModel):
    """Payload data model for PDF report generation requests."""
    results: List[RequirementItemInput] = Field(..., description="List of evaluated requirement objects.")
    conflicts: Optional[List[ConflictItemInput]] = Field(default=[], description="List of detected requirement conflicts.")
    filename: Optional[str] = Field(default=None, description="Uploaded SRS PDF filename if applicable.")
    university_name: Optional[str] = Field(default=UNIVERSITY_NAME)
    department_name: Optional[str] = Field(default=DEPARTMENT_NAME)
    student_name: Optional[str] = Field(default=STUDENT_NAME)
    guide_name: Optional[str] = Field(default=GUIDE_NAME)


# -----------------------------------------------------------------------------
# NUMBERED CANVAS WITH RUNNING HEADER & FOOTER
# -----------------------------------------------------------------------------

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass ReportLab Canvas that computes total page count dynamically,
    drawing running headers, footers, and page numbers ('Page X of Y').
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        # Skip running header/footer on cover page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(COLOR_TEXT_MUTED)

        # Running Header
        self.drawString(54, 802, "AI Requirements Intelligence Platform (ARIP)")
        self.drawRightString(541, 802, "Module 1 — Ambiguity Analysis Report")
        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.5)
        self.line(54, 794, 541, 794)

        # Running Footer
        self.line(54, 45, 541, 45)
        self.setFont("Helvetica", 8)
        self.drawString(54, 32, "Confidential & Proprietary — ARIP Automated Requirements Audit")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 32, page_text)

        self.restoreState()


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def interpret_score(score: int) -> str:
    """
    Categorizes Quality Score into standard assessment bands.

    Args:
        score (int): Quality score percentage (0-100).

    Returns:
        str: Rating interpretation label.
    """
    if score >= SCORE_EXCELLENT_THRESHOLD:
        return "Excellent"
    elif score >= SCORE_GOOD_THRESHOLD:
        return "Good"
    elif score >= SCORE_FAIR_THRESHOLD:
        return "Fair"
    else:
        return "Needs Improvement"


def generate_suggested_improvement(reason: str) -> str:
    """
    Generates a concrete, rewritten-style suggestion sentence from reason text.

    Args:
        reason (str): Explainability reason string from ambiguity model.

    Returns:
        str: Actionable suggestion sentence.
    """
    if not reason:
        return "Rewrite this requirement using specific, quantitative, and measurable criteria."

    # Pattern 1: Extract vague terms e.g. "Contains vague term(s): 'fast', 'user-friendly'"
    vague_match = re.search(r"vague term\(s\):\s*([^.]+)", reason, re.IGNORECASE)
    if vague_match and vague_match.group(1):
        terms = vague_match.group(1).strip()
        return f"Replace {terms} with precise, quantitative performance metrics or exact operational thresholds."

    # Pattern 2: Match quoted terms in single quotes
    quoted_match = re.findall(r"'([^']+)'", reason)
    if quoted_match:
        terms = ", ".join([f"'{t}'" for t in quoted_match])
        return f"Replace {terms} with specific, measurable requirements."

    return "Rewrite this requirement using specific, measurable, and testable criteria."


def generate_general_recommendations(results: List[RequirementItemInput]) -> List[str]:
    """
    Generates a tailored list of professional recommendations based on detected ambiguity reasons.

    Args:
        results (List[RequirementItemInput]): Analyzed requirement items.

    Returns:
        List[str]: Tailored list of general recommendations.
    """
    ambiguous_items = [r for r in results if r.prediction == "Ambiguous"]
    if not ambiguous_items:
        return [
            "Maintain current specification precision standards across future module extensions.",
            "Conduct periodic peer requirement reviews prior to architectural design baseline approval."
        ]

    reasons_text = " ".join([r.reason for r in ambiguous_items]).lower()
    recommendations = []

    if any(k in reasons_text for k in ["fast", "response", "timely", "quick", "latency"]):
        recommendations.append("Specify explicit response time boundaries (e.g., maximum latency under 500ms at 95th percentile) rather than subjective performance descriptors.")

    if any(k in reasons_text for k in ["user-friendly", "modern", "seamless", "intuitive", "easy"]):
        recommendations.append("Define objective usability benchmarks (e.g., completion of core workflow in under 3 clicks or zero specialized training).")

    if any(k in reasons_text for k in ["reasonable", "appropriate", "sufficient", "needed", "regularly"]):
        recommendations.append("Replace discretionary terms ('as needed', 'appropriate', 'reasonable') with quantitative bounds, exact data retention durations, or rule-based conditions.")

    if any(k in reasons_text for k in ["robust", "secure", "high performance", "heavy load"]):
        recommendations.append("Quantify system resilience standards using measurable criteria (e.g., 99.9% uptime, AES-256 encryption, role-based access control).")

    if not recommendations:
        recommendations.append("Replace vague, subjective descriptors with precise, verifiable numerical thresholds.")
        recommendations.append("Ensure all functional requirements specify inputs, expected processing logic, and measurable outputs.")

    recommendations.append("Conduct stakeholder review sessions to resolve ambiguous phrasing before proceeding to system implementation.")
    return recommendations


def generate_conclusion_paragraph(total: int, clear: int, ambiguous: int, score: int) -> str:
    """
    Generates a dynamic conclusion summary paragraph based on actual analysis counts and score.
    """
    if ambiguous == 0:
        return (
            f"The analyzed software specification contains {total} requirements. "
            f"All {clear} requirements are classified as clear and precise, with an overall quality score of {score}/100. "
            "No ambiguities were detected; the requirements specification is verified and ready to proceed to design and implementation."
        )

    return (
        f"The analyzed software specification contains {total} software requirements. "
        f"Among them, {clear} are classified as clear, while {ambiguous} contain ambiguity. "
        f"The overall requirement quality score is {score}/100. "
        "The identified ambiguous requirements should be revised using the recommendations provided before proceeding to design and implementation."
    )


# -----------------------------------------------------------------------------
# MODULAR SECTION BUILDERS
# -----------------------------------------------------------------------------

def _build_cover_page(story: list, styles: dict, data: GenerateReportRequest, now_str: str) -> None:
    """Builds Section 0: Professional Cover Page."""
    story.append(Spacer(1, 40))
    story.append(Paragraph("AI Requirements Intelligence Platform", styles["ReportTitle"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Requirement Ambiguity Analysis Report", styles["ReportSubtitle"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Module 1 — Requirement Ambiguity Detection", styles["ModuleTag"]))
    story.append(Spacer(1, 30))

    story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY_BLUE, spaceAfter=40))

    # Document & Metadata Info Block
    meta_table_data = [
        [Paragraph("<b>Platform:</b>", styles["BodyDark"]), Paragraph("AI Requirements Intelligence Platform (ARIP)", styles["BodyDark"])],
        [Paragraph("<b>Report Type:</b>", styles["BodyDark"]), Paragraph("Requirement Precision & Ambiguity Audit", styles["BodyDark"])],
        [Paragraph("<b>Generated Date & Time:</b>", styles["BodyDark"]), Paragraph(now_str, styles["BodyDark"])],
    ]

    # Append academic / institutional placeholders if configured
    univ = data.university_name or UNIVERSITY_NAME
    dept = data.department_name or DEPARTMENT_NAME
    stud = data.student_name or STUDENT_NAME
    guide = data.guide_name or GUIDE_NAME

    if univ:
        meta_table_data.append([Paragraph("<b>University / Institution:</b>", styles["BodyDark"]), Paragraph(univ, styles["BodyDark"])])
    if dept:
        meta_table_data.append([Paragraph("<b>Department:</b>", styles["BodyDark"]), Paragraph(dept, styles["BodyDark"])])
    if stud:
        meta_table_data.append([Paragraph("<b>Project Team:</b>", styles["BodyDark"]), Paragraph(stud, styles["BodyDark"])])
    if guide:
        meta_table_data.append([Paragraph("<b>Project Guide:</b>", styles["BodyDark"]), Paragraph(guide, styles["BodyDark"])])

    meta_table = Table(meta_table_data, colWidths=[150, 337])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 80))
    story.append(Paragraph("<i>Automated NLP Report generated by ARIP Engine</i>", styles["Caption"]))
    story.append(PageBreak())


def _build_document_info(story: list, styles: dict, data: GenerateReportRequest, now_str: str) -> None:
    """Builds Section 1: Document Information."""
    story.append(Paragraph("1. Document Information", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    source_desc = f"Uploaded PDF File: <b>{data.filename}</b>" if data.filename else "Manual Entry (Typed Input)"
    
    info_table_data = [
        [Paragraph("<b>Source Input Method:</b>", styles["TableTextBold"]), Paragraph(source_desc, styles["TableText"])],
        [Paragraph("<b>Document Format:</b>", styles["TableTextBold"]), Paragraph("PDF SRS Document" if data.filename else "Direct Form Input", styles["TableText"])],
        [Paragraph("<b>Analysis Date & Time:</b>", styles["TableTextBold"]), Paragraph(now_str, styles["TableText"])],
        [Paragraph("<b>Total Extracted Requirements:</b>", styles["TableTextBold"]), Paragraph(str(len(data.results)), styles["TableText"])]
    ]

    info_table = Table(info_table_data, colWidths=[180, 307])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))


def _build_executive_summary(story: list, styles: dict, total: int, clear: int, ambiguous: int, score: int) -> None:
    """Builds Section 2: Executive Summary."""
    story.append(Paragraph("2. Executive Summary", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    summary_data = [
        [Paragraph("<b>Metric Parameter</b>", styles["TableHeader"]), Paragraph("<b>Value</b>", styles["TableHeader"])],
        [Paragraph("Total Requirements Evaluated", styles["TableText"]), Paragraph(str(total), styles["TableText"])],
        [Paragraph("Clear Requirements", styles["TableText"]), Paragraph(f"<font color='{COLOR_CLEAR_TEXT.hexval()}'><b>{clear}</b></font>", styles["TableText"])],
        [Paragraph("Ambiguous Requirements", styles["TableText"]), Paragraph(f"<font color='{COLOR_AMBIGUOUS_TEXT.hexval()}'><b>{ambiguous}</b></font>", styles["TableText"])],
        [Paragraph("Overall Quality Score", styles["TableTextBold"]), Paragraph(f"<b>{score} / 100</b>", styles["TableTextBold"])]
    ]

    summary_table = Table(summary_data, colWidths=[280, 207])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))


def _build_quality_assessment(story: list, styles: dict, score: int, interpretation: str) -> None:
    """Builds Section 3: Quality Assessment."""
    story.append(Paragraph("3. Quality Assessment", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    if interpretation == "Excellent":
        badge_color = COLOR_CLEAR_TEXT.hexval()
        bg_color = COLOR_CLEAR_BG
    elif interpretation == "Good":
        badge_color = COLOR_CLEAR_TEXT.hexval()
        bg_color = COLOR_CLEAR_BG
    elif interpretation == "Fair":
        badge_color = COLOR_WARNING_TEXT.hexval()
        bg_color = colors.HexColor("#fef3c7")
    else:
        badge_color = COLOR_AMBIGUOUS_TEXT.hexval()
        bg_color = COLOR_AMBIGUOUS_BG

    content_text = (
        f"The evaluated specification achieved a Quality Score of <b>{score}%</b>, corresponding to an "
        f"assessment rating of <font color='{badge_color}'><b>{interpretation}</b></font>.<br/>"
        f"Quality Score is computed as the percentage ratio of clear, precise requirement statements relative "
        f"to total submitted requirements. High scores indicate verifiable, testable specifications."
    )

    assess_table = Table([[Paragraph(content_text, styles["BodyDark"])]], colWidths=[487])
    assess_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('PADDING', (0, 0), (-1, -1), 12),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(badge_color)),
    ]))
    story.append(assess_table)
    story.append(Spacer(1, 20))


def _build_detailed_analysis(story: list, styles: dict, data: GenerateReportRequest) -> None:
    """Builds Section 4: Detailed Requirement Analysis (Multi-page Table)."""
    story.append(Paragraph("4. Detailed Requirement Analysis", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    # Table columns: # (25pt), Requirement (200pt), Prediction (60pt), Confidence (55pt), Reason (147pt) = 487pt
    table_headers = [
        Paragraph("<b>#</b>", styles["TableHeader"]),
        Paragraph("<b>Requirement Statement</b>", styles["TableHeader"]),
        Paragraph("<b>Prediction</b>", styles["TableHeader"]),
        Paragraph("<b>Conf.</b>", styles["TableHeader"]),
        Paragraph("<b>Reason / Explainability</b>", styles["TableHeader"])
    ]
    rows = [table_headers]

    for idx, item in enumerate(data.results, 1):
        is_clear = item.prediction == "Clear"
        pred_color = COLOR_CLEAR_TEXT.hexval() if is_clear else COLOR_AMBIGUOUS_TEXT.hexval()
        pred_html = f"<font color='{pred_color}'><b>{item.prediction}</b></font>"

        row = [
            Paragraph(f"#{idx}", styles["TableTextCenter"]),
            Paragraph(item.requirement, styles["TableText"]),
            Paragraph(pred_html, styles["TableTextCenter"]),
            Paragraph(item.confidence, styles["TableTextCenter"]),
            Paragraph(item.reason, styles["TableText"])
        ]
        rows.append(row)

    det_table = Table(rows, colWidths=[25, 200, 60, 55, 147], repeatRows=1)
    det_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_PRIMARY_BLUE),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
    ]))
    story.append(det_table)
    story.append(Spacer(1, 20))


def _build_ambiguous_requirements(story: list, styles: dict, data: GenerateReportRequest) -> None:
    """Builds Section 5: Ambiguous Requirements & Suggested Improvements."""
    story.append(Paragraph("5. Ambiguous Requirements & Suggested Improvements", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    ambiguous_items = [(idx, item) for idx, item in enumerate(data.results, 1) if item.prediction == "Ambiguous"]

    if not ambiguous_items:
        no_amb_text = (
            "<b>✓ Zero Ambiguous Requirements Detected</b><br/>"
            "All evaluated requirement statements contain clear, measurable criteria without vague phrasing."
        )
        no_amb_table = Table([[Paragraph(no_amb_text, styles["BodyDark"])]], colWidths=[487])
        no_amb_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_CLEAR_BG),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_CLEAR_TEXT),
        ]))
        story.append(no_amb_table)
    else:
        for idx, item in ambiguous_items:
            suggestion = generate_suggested_improvement(item.reason)
            card_content = (
                f"<b>Requirement #{idx}:</b> <i>\"{item.requirement}\"</i><br/>"
                f"<b>Flagged Reason:</b> {item.reason}<br/>"
                f"<font color='{COLOR_PRIMARY_BLUE.hexval()}'><b>Suggested Revision:</b> {suggestion}</font>"
            )
            card_table = Table([[Paragraph(card_content, styles["BodyDark"])]], colWidths=[487])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), COLOR_AMBIGUOUS_BG),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, COLOR_AMBIGUOUS_TEXT),
            ]))
            story.append(card_table)
            story.append(Spacer(1, 10))

    story.append(Spacer(1, 10))


def _build_conflict_detection_section(story: list, styles: dict, data: GenerateReportRequest) -> None:
    """Builds Section 6: Requirement Conflict Detection Audit."""
    story.append(Paragraph("6. Requirement Conflict Detection Audit", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    conflicts = data.conflicts or []

    if not conflicts:
        no_conflict_text = (
            "<b>✓ Zero Requirement Conflicts Detected</b><br/>"
            "All candidate requirement pairs were evaluated for direct contradictions, numeric constraint bounds, "
            "temporal rules, and permission consistency. No mutually exclusive or conflicting requirements were found."
        )
        no_conf_table = Table([[Paragraph(no_conflict_text, styles["BodyDark"])]], colWidths=[487])
        no_conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_CLEAR_BG),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BOX', (0, 0), (-1, -1), 1, COLOR_CLEAR_TEXT),
        ]))
        story.append(no_conf_table)
    else:
        for idx, conf in enumerate(conflicts, 1):
            sev_color = COLOR_AMBIGUOUS_TEXT.hexval() if conf.severity == "HIGH" else COLOR_WARNING_TEXT.hexval()
            evidence_str = " | ".join([f"✓ {e}" for e in conf.evidence])
            
            card_content = (
                f"<b>Conflict #{idx}: {conf.requirement_1_id} ↔ {conf.requirement_2_id}</b> &nbsp;&nbsp;"
                f"[<b>Type:</b> {conf.conflict_type} | <font color='{sev_color}'><b>Severity: {conf.severity}</b></font> | <b>Conf:</b> {conf.confidence:.1f}%]<br/>"
                f"<b>Requirement 1 ({conf.requirement_1_id}):</b> <i>\"{conf.requirement_1_text}\"</i><br/>"
                f"<b>Requirement 2 ({conf.requirement_2_id}):</b> <i>\"{conf.requirement_2_text}\"</i><br/>"
                f"<b>Explanation:</b> {conf.explanation}<br/>"
                f"<font color='{COLOR_TEXT_MUTED.hexval()}'><b>Evidence:</b> {evidence_str}</font>"
            )
            card_table = Table([[Paragraph(card_content, styles["BodyDark"])]], colWidths=[487])
            card_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), COLOR_AMBIGUOUS_BG),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(sev_color)),
            ]))
            story.append(card_table)
            story.append(Spacer(1, 10))

    story.append(Spacer(1, 10))


def _build_recommendations(story: list, styles: dict, data: GenerateReportRequest) -> None:
    """Builds Section 7: Tailored Recommendations."""
    story.append(Paragraph("7. Professional Recommendations", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    recs = generate_general_recommendations(data.results)
    for rec in recs:
        bullet_item = f"• {rec}"
        story.append(Paragraph(bullet_item, styles["BulletText"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 16))


def _build_conclusion(story: list, styles: dict, total: int, clear: int, ambiguous: int, score: int) -> None:
    """Builds Section 7: Final Conclusion."""
    story.append(Paragraph("7. Final Conclusion", styles["SectionHeading"]))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY_BLUE, spaceAfter=14))

    paragraph_text = generate_conclusion_paragraph(total, clear, ambiguous, score)
    story.append(Paragraph(paragraph_text, styles["BodyDark"]))
    story.append(Spacer(1, 20))


# -----------------------------------------------------------------------------
# MAIN PUBLIC REPORT GENERATOR API
# -----------------------------------------------------------------------------

def generate_pdf_report(data: GenerateReportRequest) -> bytes:
    """
    Generates a complete, multi-page PDF analysis report in bytes.

    Args:
        data (GenerateReportRequest): Report dataset containing requirement results and metadata.

    Returns:
        bytes: Binary PDF document content.
    """
    buffer = io.BytesIO()

    # Document setup with A4 portrait size and 54pt (0.75 in) margins
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    # Styles Setup
    base_styles = getSampleStyleSheet()
    styles = {
        "ReportTitle": ParagraphStyle(
            "ReportTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=COLOR_PRIMARY_DARK,
            alignment=0
        ),
        "ReportSubtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=COLOR_PRIMARY_BLUE,
            alignment=0
        ),
        "ModuleTag": ParagraphStyle(
            "ModuleTag",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=COLOR_TEXT_MUTED,
            alignment=0
        ),
        "SectionHeading": ParagraphStyle(
            "SectionHeading",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=COLOR_PRIMARY_DARK,
            spaceAfter=4
        ),
        "BodyDark": ParagraphStyle(
            "BodyDark",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=COLOR_PRIMARY_DARK
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base_styles["Italic"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=COLOR_TEXT_MUTED
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.white
        ),
        "TableText": ParagraphStyle(
            "TableText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=COLOR_PRIMARY_DARK
        ),
        "TableTextBold": ParagraphStyle(
            "TableTextBold",
            parent=base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=COLOR_PRIMARY_DARK
        ),
        "TableTextCenter": ParagraphStyle(
            "TableTextCenter",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=COLOR_PRIMARY_DARK,
            alignment=1
        ),
        "BulletText": ParagraphStyle(
            "BulletText",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=COLOR_PRIMARY_DARK
        ),
    }

    # Calculations
    now_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total_count = len(data.results)
    clear_count = sum(1 for r in data.results if r.prediction == "Clear")
    ambiguous_count = total_count - clear_count
    quality_score = round((clear_count / total_count) * 100) if total_count > 0 else 0
    score_interpretation = interpret_score(quality_score)

    story = []

    # Compose PDF Section Flowables
    _build_cover_page(story, styles, data, now_str)
    _build_document_info(story, styles, data, now_str)
    _build_executive_summary(story, styles, total_count, clear_count, ambiguous_count, quality_score)
    _build_quality_assessment(story, styles, quality_score, score_interpretation)
    _build_detailed_analysis(story, styles, data)
    _build_ambiguous_requirements(story, styles, data)
    _build_conflict_detection_section(story, styles, data)
    _build_recommendations(story, styles, data)
    _build_conclusion(story, styles, total_count, clear_count, ambiguous_count, quality_score)

    # Build PDF with NumberedCanvas (Page X of Y headers/footers)
    doc.build(story, canvasmaker=NumberedCanvas)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
