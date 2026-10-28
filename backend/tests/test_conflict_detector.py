"""
AI Requirements Intelligence Platform (ARIP)
Automated Test Suite for Conflict Detection Engine (test_conflict_detector.py)

Tests all 7 required test cases specified in the ARIP Conflict Detection prompt:
- Test 1: Direct Contradiction
- Test 2: Condition Difference (No Conflict)
- Test 3: Compatible Numeric Constraint (No Conflict)
- Test 4: Conflicting Numeric Constraint
- Test 5: Unrelated Requirements (No Conflict)
- Test 6: Authorization / Permission Conflict
- Test 7: Similar But Non-Conflicting Requirements (No Conflict)
"""

import sys
import os
import unittest

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from conflict_detector import detect_conflicts, TYPE_DIRECT_CONTRADICTION, TYPE_NUMERIC_CONFLICT, TYPE_PERMISSION_CONFLICT


class TestConflictDetector(unittest.TestCase):

    def test_case_1_direct_contradiction(self):
        """TEST 1 — DIRECT CONTRADICTION: Allow vs Prevent download reports."""
        reqs = [
            "The system shall allow users to download reports.",
            "The system shall prevent users from downloading reports."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 1)
        conflict = res["conflicts"][0]
        self.assertEqual(conflict["requirement_1_id"], "REQ-001")
        self.assertEqual(conflict["requirement_2_id"], "REQ-002")
        self.assertEqual(conflict["conflict_type"], TYPE_DIRECT_CONTRADICTION)
        self.assertEqual(conflict["severity"], "HIGH")
        self.assertTrue(conflict["confidence"] > 70.0)
        self.assertIn("permits", conflict["explanation"])

    def test_case_2_condition_difference(self):
        """TEST 2 — CONDITION DIFFERENCE: Cancel before vs after shipment (NO CONFLICT)."""
        reqs = [
            "Customers can cancel orders before shipment.",
            "Customers cannot cancel orders after shipment."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 0, "Condition difference should prevent false positive conflict.")

    def test_case_3_compatible_numeric_constraint(self):
        """TEST 3 — COMPATIBLE NUMERIC CONSTRAINT: At least 8 vs 12 chars (NO CONFLICT)."""
        reqs = [
            "The password shall contain at least 8 characters.",
            "The password shall contain at least 12 characters."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 0, "Stronger compatible constraint should not trigger conflict.")

    def test_case_4_conflicting_numeric_constraint(self):
        """TEST 4 — CONFLICTING NUMERIC CONSTRAINT: Less than 2s vs Greater than 5s."""
        reqs = [
            "Response time shall be less than 2 seconds.",
            "Response time shall be greater than 5 seconds."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 1)
        conflict = res["conflicts"][0]
        self.assertEqual(conflict["conflict_type"], TYPE_NUMERIC_CONFLICT)
        self.assertEqual(conflict["severity"], "HIGH")

    def test_case_5_unrelated_requirements(self):
        """TEST 5 — UNRELATED REQUIREMENTS: Encrypt passwords vs Sales reports (NO CONFLICT)."""
        reqs = [
            "The system shall encrypt passwords using AES-256.",
            "The system shall generate monthly sales reports."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 0)

    def test_case_6_authorization_conflict(self):
        """TEST 6 — AUTHORIZATION: Only admins delete vs All users allowed delete."""
        reqs = [
            "Only administrators shall delete user accounts.",
            "All registered users shall be allowed to delete user accounts."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 1)
        conflict = res["conflicts"][0]
        self.assertEqual(conflict["conflict_type"], TYPE_PERMISSION_CONFLICT)
        self.assertEqual(conflict["severity"], "HIGH")

    def test_case_7_similar_but_not_conflicting(self):
        """TEST 7 — SIMILAR BUT NOT CONFLICTING: Email vs OTP recovery (NO CONFLICT)."""
        reqs = [
            "The system shall support password recovery through email.",
            "The system shall support password recovery through OTP."
        ]
        res = detect_conflicts(reqs)
        self.assertEqual(res["conflicts_found"], 0, "Complementary recovery options should not be flagged as conflict.")


if __name__ == "__main__":
    unittest.main()
