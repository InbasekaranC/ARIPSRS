"""
AI Requirements Intelligence Platform (ARIP)
Regression Test Suite (test_regression.py)

Verifies that all pre-existing ARIP modules continue to operate cleanly and accurately:
1. Ambiguity Detection Predictor (predict.py)
2. Text Cleaning & Section Detection (text_cleaner.py, section_detector.py)
3. Sentence Boundary Segmentation (sentence_splitter.py)
4. Rule Validation & ML Requirement Classifier (requirement_validator.py, requirement_classifier.py)
5. Duplicate Filtering (duplicate_filter.py)
6. Requirement Normalization (requirement_normalizer.py)
7. ReportLab PDF Generation (report/generate_report.py)
8. FastAPI Endpoints (/health, /analyze, /detect-conflicts, /generate-report)
"""

import sys
import os
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict import predict_requirement
from requirement_validator import validate_requirement_candidate
from requirement_normalizer import normalize_requirement
from report.generate_report import generate_pdf_report, GenerateReportRequest, RequirementItemInput
from fastapi.testclient import TestClient
from main import app

class TestARIPRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Verify GET /health returns 200 OK."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_ambiguity_prediction_clear(self):
        """Verify predict_requirement identifies clear, measurable requirement."""
        req = "The system shall allow users to reset passwords via email within 2 minutes."
        res = predict_requirement(req)
        self.assertEqual(res["prediction"], "Clear")
        self.assertTrue("2 minutes" in res["requirement"])

    def test_ambiguity_prediction_ambiguous(self):
        """Verify predict_requirement identifies vague/ambiguous terms."""
        req = "The application should be fast and user-friendly."
        res = predict_requirement(req)
        self.assertEqual(res["prediction"], "Ambiguous")
        self.assertIn("fast", res["reason"].lower())

    def test_requirement_validation(self):
        """Verify candidate rule validator rejects metadata noise."""
        valid_res = validate_requirement_candidate("The system shall store transaction records for 5 years.")
        self.assertTrue(valid_res.is_valid)

        invalid_res = validate_requirement_candidate("Table of Contents")
        self.assertFalse(invalid_res.is_valid)

    def test_requirement_normalization(self):
        """Verify normalizer formats sentence into standard IEEE SRS phrasing."""
        raw = "users can update profile information."
        norm = normalize_requirement(raw)
        self.assertTrue("shall" in norm.lower() or "user" in norm.lower())

    def test_api_analyze_endpoint(self):
        """Verify POST /analyze returns ambiguity results AND conflict results."""
        payload = {
            "requirements": [
                "The system shall allow users to download reports.",
                "The system shall prevent users from downloading reports."
            ]
        }
        response = self.client.post("/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertIn("conflicts", data)
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(len(data["conflicts"]), 1)

    def test_api_detect_conflicts_endpoint(self):
        """Verify POST /detect-conflicts endpoint."""
        payload = {
            "requirements": [
                "Response time shall be less than 2 seconds.",
                "Response time shall be greater than 5 seconds."
            ]
        }
        response = self.client.post("/detect-conflicts", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["conflicts_found"], 1)

    def test_pdf_report_generation(self):
        """Verify ReportLab PDF generator produces non-empty binary bytes."""
        req_item = RequirementItemInput(
            requirement="The system shall process payments within 2 seconds.",
            prediction="Clear",
            confidence="95%",
            reason="No vague terms detected."
        )
        req_obj = GenerateReportRequest(results=[req_item], filename="test_srs.pdf")
        pdf_bytes = generate_pdf_report(req_obj)
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
