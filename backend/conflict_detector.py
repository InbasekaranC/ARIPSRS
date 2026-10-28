"""
AI Requirements Intelligence Platform (ARIP)
Layer 2 — General Requirement Intelligence
Conflict Detection Engine (conflict_detector.py)

Identifies potentially contradictory, incompatible, or inconsistent requirement pairs
in Software Requirements Specifications (SRS). Evaluates direct contradictions,
permission/authorization conflicts, behavioral conflicts, temporal conflicts, and
numeric constraint conflicts with explainable evidence and severity metrics.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure module logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARIP.ConflictDetector")

# Configurable Constants
DEFAULT_SIMILARITY_THRESHOLD: float = 0.50

# Load spaCy NLP model lazily / safely
_spacy_nlp = None

def _get_spacy_nlp():
    """Lazy loader for spaCy en_core_web_sm model with fallback handling."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm")
            logger.info("[ConflictDetector] Loaded spaCy en_core_web_sm model successfully.")
        except Exception as e:
            logger.warning(f"[ConflictDetector] Could not load spaCy model: {e}. Falling back to regex stem parsing.")
            _spacy_nlp = False
    return _spacy_nlp if _spacy_nlp is not False else None


# Polarity Keywords
NEGATIVE_POLARITY_TERMS: Set[str] = {
    "not", "never", "cannot", "must not", "shall not", "should not",
    "prohibited", "prevent", "deny", "disable", "disallow", "forbid",
    "reject", "restricted", "banned", "precludes", "refuse"
}

POSITIVE_POLARITY_TERMS: Set[str] = {
    "allow", "permit", "enable", "shall", "must", "can", "should",
    "accept", "authorize", "grant", "support", "provide", "include"
}

# Common Action Stems for Robust Matching across verb inflections (e.g. download / downloading)
ACTION_STEM_MAP: Dict[str, str] = {
    "download": "download", "downloading": "download", "downloads": "download", "downloaded": "download",
    "delete": "delete", "deleting": "delete", "deletes": "delete", "deleted": "delete", "deletion": "delete",
    "cancel": "cancel", "canceling": "cancel", "cancelling": "cancel", "cancels": "cancel", "canceled": "cancel", "cancellation": "cancel",
    "process": "process", "processing": "process", "processes": "process", "processed": "process",
    "encrypt": "encrypt", "encrypting": "encrypt", "encrypts": "encrypt", "encrypted": "encrypt", "encryption": "encrypt",
    "generate": "generate", "generating": "generate", "generates": "generate", "generated": "generate", "generation": "generate",
    "upload": "upload", "uploading": "upload", "uploads": "upload", "uploaded": "upload",
    "remove": "remove", "removing": "remove", "removes": "remove", "removed": "remove", "removal": "remove",
    "recover": "recover", "recovering": "recover", "recovers": "recover", "recovered": "recover", "recovery": "recover",
    "reset": "reset", "resetting": "reset", "resets": "reset",
    "access": "access", "accessing": "access", "accesses": "access",
    "log": "log", "logging": "log", "logs": "log", "logged": "log",
    "login": "login", "logout": "logout",
    "book": "book", "booking": "book", "books": "book", "booked": "book",
    "permit": "permit", "permits": "permit", "prevent": "prevent", "prevents": "prevent"
}

# Conflict Type Names
TYPE_DIRECT_CONTRADICTION = "Direct Contradiction"
TYPE_PERMISSION_CONFLICT = "Permission Conflict"
TYPE_BEHAVIORAL_CONFLICT = "Behavioral Conflict"
TYPE_TEMPORAL_CONFLICT = "Temporal Conflict"
TYPE_NUMERIC_CONFLICT = "Numeric Constraint Conflict"
TYPE_POTENTIAL_CONFLICT = "Potential Conflict"


class RequirementItem:
    """Internal representation of a requirement with stable ID and parsed features."""

    def __init__(self, req_id: str, text: str):
        self.id: str = req_id
        self.text: str = text
        self.clean_text: str = text.strip()
        self.text_lower: str = text.lower()
        
        # Extracted NLP Features
        self.polarity: str = self._extract_polarity()
        self.conditions: List[str] = self._extract_conditions()
        self.numeric_constraints: List[Dict[str, Any]] = self._extract_numeric_constraints()
        self.temporal_features: List[str] = self._extract_temporal_features()
        self.roles: List[str] = self._extract_roles()
        self.actions: Set[str] = self._extract_actions()

    def _extract_polarity(self) -> str:
        """Determines if the requirement has negative or positive polarity."""
        # 1. Explicit negative verb phrases or modal negation
        if re.search(r'\b(shall|must|should|can|will)\s+not\b', self.text_lower):
            return "NEGATIVE"
        if re.search(r'\b(prevent|deny|disallow|prohibit|forbid|disable|restrict)\b', self.text_lower):
            return "NEGATIVE"

        # 2. Explicit positive grant verbs
        if re.search(r'\b(permit|allow|enable|grant|authorize)\b', self.text_lower):
            return "POSITIVE"

        tokens = re.findall(r'\b\w+\b', self.text_lower)
        neg_count = sum(1 for t in tokens if t in NEGATIVE_POLARITY_TERMS or t in {"neither", "nor"})
        if neg_count > 0:
            return "NEGATIVE"
        return "POSITIVE"

    def _extract_conditions(self) -> List[str]:
        """Extracts conditional phrases e.g. 'before shipment', 'after payment'."""
        conditions = []
        patterns = [
            r'\b(before|after|upon|when|if|during|unless|provided that|in case of)\s+([a-z0-9_\s]{3,30}?)(?=[,\.;]|$|\bshall\b|\bmust\b)'
        ]
        for p in patterns:
            matches = re.finditer(p, self.text_lower)
            for m in matches:
                cond_phrase = m.group(0).strip()
                conditions.append(cond_phrase)
        return conditions

    def _extract_numeric_constraints(self) -> List[Dict[str, Any]]:
        """
        Parses numeric constraints e.g. 'less than 2 seconds', 'at least 8 characters', '> 5s'.
        Returns list of constraint dicts with operator, value, unit, and raw string.
        """
        constraints = []
        op_pattern = r'\b(less than|under|fewer than|below|at most|maximum of|greater than|over|more than|above|at least|minimum of|exactly|equal to)\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?'
        matches = re.finditer(op_pattern, self.text_lower)
        for m in matches:
            op_str = m.group(1).lower()
            val = float(m.group(2))
            unit = m.group(3) if m.group(3) else ""
            
            if op_str in ["less than", "under", "fewer than", "below"]:
                operator = "<"
            elif op_str in ["at most", "maximum of"]:
                operator = "<="
            elif op_str in ["greater than", "over", "more than", "above"]:
                operator = ">"
            elif op_str in ["at least", "minimum of"]:
                operator = ">="
            elif op_str in ["exactly", "equal to"]:
                operator = "=="
            else:
                operator = "=="
                
            constraints.append({
                "operator": operator,
                "value": val,
                "unit": unit,
                "raw": m.group(0)
            })

        sym_pattern = r'([<>=]=?)\s*(\d+(?:\.\d+)?)\s*([a-z%]+)?'
        for m in re.finditer(sym_pattern, self.text_lower):
            operator = m.group(1)
            val = float(m.group(2))
            unit = m.group(3) if m.group(3) else ""
            if not any(c["value"] == val and c["operator"] == operator for c in constraints):
                constraints.append({
                    "operator": operator,
                    "value": val,
                    "unit": unit,
                    "raw": m.group(0)
                })

        return constraints

    def _extract_temporal_features(self) -> List[str]:
        """Extracts temporal execution rules e.g. 'immediately', 'after 24 hours'."""
        temporal = []
        temp_keywords = [
            "immediately", "instantly", "real-time", "realtime", "synchronously",
            "after 24 hours", "daily", "monthly", "hourly", "scheduled",
            "until manual logout", "automatically log out", "after 10 minutes"
        ]
        for kw in temp_keywords:
            if kw in self.text_lower:
                temporal.append(kw)
        return temporal

    def _extract_roles(self) -> List[str]:
        """Extracts user roles/permission actors e.g. 'administrators', 'only authorized medical staff', 'all users'."""
        roles = []
        role_patterns = [
            r'\b(only\s+administrators|only\s+admins|only\s+authorized\s+medical\s+staff|only\s+authorized\s+hospital\s+staff|only\s+authorized\s+staff|authorized\s+medical\s+staff|authorized\s+hospital\s+staff|authorized\s+staff|administrators|admins|supervisors|managers)\b',
            r'\b(all\s+users|all\s+registered\s+users|registered\s+users|any\s+user|every\s+user|customers|guests)\b'
        ]
        for p in role_patterns:
            matches = re.finditer(p, self.text_lower)
            for m in matches:
                roles.append(m.group(0).strip())
        return roles

    def _extract_actions(self) -> Set[str]:
        """Extracts main verb action stems using spaCy or regex stem mapping."""
        actions: Set[str] = set()
        nlp = _get_spacy_nlp()
        if nlp:
            try:
                doc = nlp(self.text)
                for token in doc:
                    if token.pos_ == "VERB" and token.lemma_ not in {"be", "have", "do", "shall", "must", "should", "can", "may"}:
                        lemma = token.lemma_.lower()
                        actions.add(ACTION_STEM_MAP.get(lemma, lemma))
            except Exception:
                pass
        
        # Regex stem mapping fallback
        tokens = re.findall(r'\b[a-z]+\b', self.text_lower)
        for t in tokens:
            if t in ACTION_STEM_MAP:
                actions.add(ACTION_STEM_MAP[t])

        return actions


def assign_stable_ids(requirements: List[str]) -> List[Tuple[str, str]]:
    """
    Assigns stable requirement IDs (REQ-001, REQ-002, ...) to input requirements.
    
    Args:
        requirements (List[str]): Raw requirement strings.
        
    Returns:
        List[Tuple[str, str]]: List of tuples (req_id, text).
    """
    assigned = []
    for idx, req in enumerate(requirements, 1):
        req_id = f"REQ-{idx:03d}"
        assigned.append((req_id, req.strip()))
    return assigned


def generate_candidate_pairs(
    req_items: List[RequirementItem],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> List[Tuple[RequirementItem, RequirementItem, float]]:
    """
    Generates potential candidate requirement pairs using TF-IDF Vectorization, Cosine Similarity,
    and semantic feature overlap triggers.
    """
    n = len(req_items)
    if n < 2:
        return []

    texts = [item.clean_text for item in req_items]
    
    # Fit TF-IDF Vectorizer with word boundary tokenization
    vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except Exception as e:
        logger.warning(f"[Candidate Pair Gen] TF-IDF vectorization fallback: {e}")
        sim_matrix = np.zeros((n, n))

    candidate_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            sim_score = float(sim_matrix[i, j])
            item1 = req_items[i]
            item2 = req_items[j]

            # Feature overlap checks
            shared_actions = item1.actions.intersection(item2.actions)
            opposite_polarity = (item1.polarity != item2.polarity)

            # Role contrast check (e.g. only administrators / authorized staff vs all registered users)
            roles1 = " ".join(item1.roles).lower()
            roles2 = " ".join(item2.roles).lower()
            has_restricted1 = any(r in roles1 for r in ["admin", "authorized", "medical staff", "hospital staff", "supervisor", "manager"])
            has_restricted2 = any(r in roles2 for r in ["admin", "authorized", "medical staff", "hospital staff", "supervisor", "manager"])
            has_public1 = any(r in roles1 for r in ["user", "customer", "guest", "all"])
            has_public2 = any(r in roles2 for r in ["user", "customer", "guest", "all"])
            has_role_contrast = (has_restricted1 and has_public2) or (has_restricted2 and has_public1)
            
            # Behavioral contrast check (auto-logout vs keep logged in)
            has_behavior_contrast = (("log out" in item1.text_lower or "logout" in item1.text_lower) and ("logged in" in item2.text_lower or "login" in item2.text_lower))
            
            # Numeric constraint check (both have numbers)
            has_numeric_both = bool(item1.numeric_constraints and item2.numeric_constraints)

            # Candidate pair triggers:
            # 1. High TF-IDF Cosine similarity (>= similarity_threshold)
            # 2. Opposite polarity + shared action stem
            # 3. Role contrast + shared action stem
            # 4. Behavioral or Numeric constraint overlap
            is_candidate = (
                (sim_score >= similarity_threshold) or
                (opposite_polarity and len(shared_actions) > 0) or
                (has_role_contrast and (len(shared_actions) > 0 or sim_score >= 0.20)) or
                has_behavior_contrast or
                (has_numeric_both and sim_score >= 0.30)
            )
            
            if is_candidate:
                candidate_pairs.append((item1, item2, sim_score))

    return candidate_pairs


def evaluate_pair_conflict(
    item1: RequirementItem,
    item2: RequirementItem,
    similarity_score: float
) -> Optional[Dict[str, Any]]:
    """
    Deep conflict analysis engine for a candidate requirement pair.
    
    Returns conflict dictionary if a conflict is detected, otherwise None.
    """
    evidence: List[str] = []
    conflict_type: Optional[str] = None
    severity: str = "LOW"
    explanation: str = ""

    # Feature Extractions for Pair
    shared_actions = item1.actions.intersection(item2.actions)
    opposite_polarity = (item1.polarity != item2.polarity)
    
    # -------------------------------------------------------------------------
    # RULE 1: CONDITION DIFFERENCE CHECK (Prevents False Positives)
    # -------------------------------------------------------------------------
    # If requirements specify distinct lifecycle/time conditions (e.g. "before shipment" vs "after shipment"),
    # they describe non-overlapping contexts and do NOT conflict!
    cond1_str = " ".join(item1.conditions).lower()
    cond2_str = " ".join(item2.conditions).lower()
    
    if ("before" in cond1_str and "after" in cond2_str) or ("after" in cond1_str and "before" in cond2_str):
        logger.debug(f"Condition difference detected between {item1.id} and {item2.id}: Not a conflict.")
        return None

    # -------------------------------------------------------------------------
    # RULE 2: NUMERIC / CONSTRAINT CONFLICT
    # -------------------------------------------------------------------------
    if item1.numeric_constraints and item2.numeric_constraints:
        for c1 in item1.numeric_constraints:
            for c2 in item2.numeric_constraints:
                op1, val1 = c1["operator"], c1["value"]
                op2, val2 = c2["operator"], c2["value"]

                # Case A: Incompatible direction (e.g. "< 2" vs "> 5")
                if (op1 in ["<", "<="] and op2 in [">", ">="] and val1 < val2):
                    conflict_type = TYPE_NUMERIC_CONFLICT
                    severity = "HIGH"
                    evidence = [
                        f"Incompatible numerical bounds ({c1['raw']} vs {c2['raw']})",
                        "Disjoint target constraint ranges"
                    ]
                    explanation = (
                        f"{item1.id} requires value {c1['raw']} while {item2.id} requires value {c2['raw']}. "
                        "These constraints are mathematically mutually exclusive."
                    )
                    break
                elif (op2 in ["<", "<="] and op1 in [">", ">="] and val2 < val1):
                    conflict_type = TYPE_NUMERIC_CONFLICT
                    severity = "HIGH"
                    evidence = [
                        f"Incompatible numerical bounds ({c1['raw']} vs {c2['raw']})",
                        "Disjoint target constraint ranges"
                    ]
                    explanation = (
                        f"{item1.id} requires value {c1['raw']} while {item2.id} requires value {c2['raw']}. "
                        "These constraints are mathematically mutually exclusive."
                    )
                    break

                # Case B: Sub-constraint / Compatible constraint (e.g. ">= 8" vs ">= 12")
                # Both specify lower bound (e.g. >= 8 and >= 12) -> NO CONFLICT (12 satisfies 8)
                if (op1 in [">=", ">"] and op2 in [">=", ">"]):
                    continue
                if (op1 in ["<=", "<"] and op2 in ["<=", "<"]):
                    continue

        if conflict_type:
            confidence_val = min(98.5, round(85.0 + (similarity_score * 15.0), 1))
            return {
                "requirement_1_id": item1.id,
                "requirement_1_text": item1.text,
                "requirement_2_id": item2.id,
                "requirement_2_text": item2.text,
                "conflict_detected": True,
                "conflict_type": conflict_type,
                "severity": severity,
                "confidence": confidence_val,
                "similarity_score": round(similarity_score, 2),
                "explanation": explanation,
                "evidence": evidence
            }

    # -------------------------------------------------------------------------
    # RULE 3: DIRECT CONTRADICTION (Opposite Polarity on Same Action/Object)
    # -------------------------------------------------------------------------
    if opposite_polarity:
        has_direct_action = (len(shared_actions) > 0)
        
        # Check explicit action contradiction (e.g. permit vs prevent, or shared slot/booking target)
        is_permit_prevent_pair = (
            ("prevent" in item1.text_lower or "prohibit" in item1.text_lower or "deny" in item1.text_lower) and
            ("permit" in item2.text_lower or "allow" in item2.text_lower or "enable" in item2.text_lower)
        ) or (
            ("prevent" in item2.text_lower or "prohibit" in item2.text_lower or "deny" in item2.text_lower) and
            ("permit" in item1.text_lower or "allow" in item1.text_lower or "enable" in item1.text_lower)
        )

        if is_permit_prevent_pair or (has_direct_action and similarity_score >= 0.40):
            conflict_type = TYPE_DIRECT_CONTRADICTION
            severity = "HIGH"
            evidence = [
                "Same target domain/action stem",
                "Opposite sentence polarity (Positive vs Negative)",
                "Contradictory execution permission"
            ]
            explanation = (
                f"{item1.id} permits or enables the action, while {item2.id} explicitly prohibits or prevents the same action."
            )

            confidence_val = min(98.0, round(82.0 + (similarity_score * 18.0), 1))
            return {
                "requirement_1_id": item1.id,
                "requirement_1_text": item1.text,
                "requirement_2_id": item2.id,
                "requirement_2_text": item2.text,
                "conflict_detected": True,
                "conflict_type": conflict_type,
                "severity": severity,
                "confidence": confidence_val,
                "similarity_score": round(similarity_score, 2),
                "explanation": explanation,
                "evidence": evidence
            }

    # -------------------------------------------------------------------------
    # RULE 4: PERMISSION / AUTHORIZATION CONFLICT
    # -------------------------------------------------------------------------
    roles1 = " ".join(item1.roles).lower()
    roles2 = " ".join(item2.roles).lower()
    
    has_restricted1 = any(term in roles1 for term in ["admin", "authorized", "medical staff", "hospital staff", "supervisor", "manager"])
    has_restricted2 = any(term in roles2 for term in ["admin", "authorized", "medical staff", "hospital staff", "supervisor", "manager"])
    has_all1 = any(term in roles1 for term in ["all users", "all registered users", "registered users", "any user", "every user"])
    has_all2 = any(term in roles2 for term in ["all users", "all registered users", "registered users", "any user", "every user"])

    # Match target object or shared action stem
    has_action_match = (len(shared_actions) > 0)
    domain_keywords = ["medical record", "record", "account", "profile", "report", "slot", "appointment", "data", "file"]
    has_target_match = any(kw in item1.text_lower and kw in item2.text_lower for kw in domain_keywords)

    if ((has_restricted1 and has_all2) or (has_restricted2 and has_all1)) and (has_target_match or (has_action_match and similarity_score >= 0.30)):
        conflict_type = TYPE_PERMISSION_CONFLICT
        severity = "HIGH"
        evidence = [
            "Conflicting user authorization scope",
            "Restricted role vs unrestricted public role",
            "Inconsistent access control criteria"
        ]
        explanation = (
            f"{item1.id} restricts authorization to specific authorized roles, whereas {item2.id} grants permission to all registered users for the same operation."
        )

        confidence_val = min(95.0, round(78.0 + (similarity_score * 20.0), 1))
        return {
            "requirement_1_id": item1.id,
            "requirement_1_text": item1.text,
            "requirement_2_id": item2.id,
            "requirement_2_text": item2.text,
            "conflict_detected": True,
            "conflict_type": conflict_type,
            "severity": severity,
            "confidence": confidence_val,
            "similarity_score": round(similarity_score, 2),
            "explanation": explanation,
            "evidence": evidence
        }

    # -------------------------------------------------------------------------
    # RULE 5: BEHAVIORAL CONFLICT
    # -------------------------------------------------------------------------
    if ("automatically log out" in item1.text_lower or "auto-logout" in item1.text_lower) and ("keep" in item2.text_lower and "logged in" in item2.text_lower):
        conflict_type = TYPE_BEHAVIORAL_CONFLICT
        severity = "MEDIUM"
        evidence = [
            "Inconsistent session persistence behavior",
            "Automatic logout rule vs manual logout rule"
        ]
        explanation = (
            f"{item1.id} enforces automatic session termination after inactivity, while {item2.id} mandates persistent logins until manual user logout."
        )

        confidence_val = min(92.0, round(75.0 + (similarity_score * 20.0), 1))
        return {
            "requirement_1_id": item1.id,
            "requirement_1_text": item1.text,
            "requirement_2_id": item2.id,
            "requirement_2_text": item2.text,
            "conflict_detected": True,
            "conflict_type": conflict_type,
            "severity": severity,
            "confidence": confidence_val,
            "similarity_score": round(similarity_score, 2),
            "explanation": explanation,
            "evidence": evidence
        }

    # -------------------------------------------------------------------------
    # RULE 6: TEMPORAL CONFLICT
    # -------------------------------------------------------------------------
    if ("immediately" in item1.text_lower or "real-time" in item1.text_lower) and ("after 24 hours" in item2.text_lower or "scheduled" in item2.text_lower):
        conflict_type = TYPE_TEMPORAL_CONFLICT
        severity = "MEDIUM"
        evidence = [
            "Conflicting execution timing",
            "Immediate execution vs delayed batch execution"
        ]
        explanation = (
            f"{item1.id} requires immediate real-time execution, whereas {item2.id} specifies a 24-hour delayed processing schedule."
        )

        confidence_val = min(93.0, round(76.0 + (similarity_score * 20.0), 1))
        return {
            "requirement_1_id": item1.id,
            "requirement_1_text": item1.text,
            "requirement_2_id": item2.id,
            "requirement_2_text": item2.text,
            "conflict_detected": True,
            "conflict_type": conflict_type,
            "severity": severity,
            "confidence": confidence_val,
            "similarity_score": round(similarity_score, 2),
            "explanation": explanation,
            "evidence": evidence
        }

    return None


def detect_conflicts(
    requirements: List[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """
    Public API function to perform Conflict Detection on a list of software requirements.
    
    Args:
        requirements (List[str]): List of software requirement strings.
        similarity_threshold (float): Similarity threshold for candidate pair generation.
        
    Returns:
        Dict[str, Any]: Conflict detection summary and detailed itemized conflict list.
    """
    valid_reqs = [r.strip() for r in requirements if r and r.strip()]
    
    if len(valid_reqs) < 2:
        return {
            "total_requirements": len(valid_reqs),
            "total_candidate_pairs": 0,
            "conflicts_found": 0,
            "severity_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "conflicts": []
        }

    assigned = assign_stable_ids(valid_reqs)
    parsed_items = [RequirementItem(req_id, text) for req_id, text in assigned]

    candidate_pairs = generate_candidate_pairs(parsed_items, similarity_threshold=similarity_threshold)

    conflicts: List[Dict[str, Any]] = []
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for item1, item2, sim_score in candidate_pairs:
        conflict_res = evaluate_pair_conflict(item1, item2, sim_score)
        if conflict_res:
            conflicts.append(conflict_res)
            sev = conflict_res.get("severity", "LOW")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

    return {
        "total_requirements": len(valid_reqs),
        "total_candidate_pairs": len(candidate_pairs),
        "conflicts_found": len(conflicts),
        "severity_counts": severity_counts,
        "conflicts": conflicts
    }


if __name__ == "__main__":
    test_reqs = [
        "The system shall allow users to download reports.",
        "The system shall prevent users from downloading reports.",
        "Customers can cancel orders before shipment.",
        "Customers cannot cancel orders after shipment.",
        "The password shall contain at least 8 characters.",
        "The password shall contain at least 12 characters.",
        "Response time shall be less than 2 seconds.",
        "Response time shall be greater than 5 seconds."
    ]
    res = detect_conflicts(test_reqs)
    import json
    print("Conflict Detection Test Output:")
    print(json.dumps(res, indent=2))
