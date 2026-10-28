"""
AI Requirements Intelligence Platform (ARIP)
Pipeline Stage 8: Requirement Normalization Module (requirement_normalizer.py)

Normalizes candidate requirement statements into consistent software requirement phrasing
(e.g., converting "Users can view balance." to "The system shall allow users to view balance.")
without altering meaning or inventing missing parameters.
"""

import re
from typing import List


def normalize_requirement(sentence: str) -> str:
    """
    Normalizes a single candidate requirement sentence into standard SRS phrasing.

    Args:
        sentence (str): Input requirement string.

    Returns:
        str: Normalized requirement string.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return ""

    text = sentence.strip()

    # If sentence already contains IEEE modal verbs (shall, should, must, may), preserve it
    if re.search(r'\b(shall|should|must|may)\b', text, re.IGNORECASE):
        # Ensure capitalization and ending punctuation
        text = text[0].upper() + text[1:]
        if not text.endswith(('.', '!', '?')):
            text += "."
        return text

    # Declarative pattern rewrites:
    # Pattern 1: "Users can <verb>..." -> "The system shall allow users to <verb>..."
    match_can = re.match(r'^(?:the\s+)?([a-z0-9_\s]+)\s+(?:can|is\s+able\s+to|shall\s+be\s+able\s+to)\s+(.+)$', text, re.IGNORECASE)
    if match_can:
        actor = match_can.group(1).strip().lower()
        action = match_can.group(2).strip()
        norm = f"The system shall allow {actor} to {action}"
        if not norm.endswith(('.', '!', '?')):
            norm += "."
        return norm

    # Pattern 2: "Administrators manage..." -> "The system shall allow administrators to manage..."
    match_actor = re.match(r'^(administrators?|users?|customers?|nurses?|doctors?|teachers?|students?|managers?)\s+([a-z]+)\s+(.+)$', text, re.IGNORECASE)
    if match_actor:
        actor = match_actor.group(1).strip().lower()
        verb = match_actor.group(2).strip().lower()
        rest = match_actor.group(3).strip()
        norm = f"The system shall allow {actor} to {verb} {rest}"
        if not norm.endswith(('.', '!', '?')):
            norm += "."
        return norm

    # Fallback default: Prefix declarative statement with "The system shall "
    text_lower = text[0].lower() + text[1:]
    norm = f"The system shall {text_lower}"
    if not norm.endswith(('.', '!', '?')):
        norm += "."

    return norm


def normalize_requirements_list(requirements: List[str]) -> List[str]:
    """
    Normalizes a list of requirement statements.

    Args:
        requirements (List[str]): List of raw requirement statements.

    Returns:
        List[str]: List of normalized requirement statements.
    """
    if not requirements:
        return []

    normalized_list = []
    for req in requirements:
        norm = normalize_requirement(req)
        if norm:
            normalized_list.append(norm)

    return normalized_list
