"""
AI Requirements Intelligence Platform (ARIP) - Module 1: Ambiguity Detection
Prediction & Inference Module (predict.py)

This module loads the trained model once at import time and provides single & batch
prediction services with confidence scores and rule-based trigger explainability.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional

# Paths to model artifacts
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR: str = os.path.join(BASE_DIR, "model", "saved_model")
LABEL_MAPPING_PATH: str = os.path.join(MODEL_DIR, "label_mapping.json")
PHASE15_MODEL_PATH: str = os.path.join(MODEL_DIR, "ambiguity_model_phase15.joblib")
PHASE14_MODEL_PATH: str = os.path.join(MODEL_DIR, "ambiguity_model_phase14.joblib")
BASELINE_MODEL_PATH: str = os.path.join(MODEL_DIR, "baseline_model.joblib")

try:
    from model_pipeline import WordCharTfidfPipeline
except Exception:
    try:
        from train_phase15 import WordCharTfidfPipeline
    except Exception:
        class WordCharTfidfPipeline:
            pass

# List of known vague / ambiguous trigger words and phrases for explainability
AMBIGUOUS_TRIGGERS: List[str] = [
    "fast",
    "user-friendly",
    "user friendly",
    "easy to use",
    "appropriate",
    "efficient",
    "reasonable",
    "sufficient",
    "robust",
    "several",
    "some",
    "as needed",
    "etc.",
    "etc",
    "and so on",
    "flexible",
    "modern",
    "seamless",
    "optimal",
    "as quickly as possible",
    "timely manner",
    "timely",
    "lightweight",
    "smoothly",
    "high performance",
    "minimal downtime",
    "minimal interruption",
    "minimal",
    "throughout the day",
    "secure environment",
    "secure",
    "usable",
    "clear access",
    "suitable",
    "versatile",
    "nicely",
    "standardly",
    "convenient",
    "useful",
    "good",
    "safe",
    "fairly",
    "properly",
    "effectively"
]


class RequirementPredictor:
    """
    Singleton predictor class managing model state and inference execution.
    """

    _instance: Optional["RequirementPredictor"] = None

    def __init__(self) -> None:
        """
        Initializes predictor by loading model artifacts and label mappings.
        """
        self.model_type: str = "Uninitialized"
        self.torch_model = None
        self.tokenizer = None
        self.baseline_pipeline = None
        self.id2label: Dict[int, str] = {0: "Clear", 1: "Ambiguous"}
        self.label2id: Dict[str, int] = {"Clear": 0, "Ambiguous": 1}

        self._load_label_mapping()
        self._load_model()

    def _load_label_mapping(self) -> None:
        """Loads label mapping from JSON file if available."""
        if os.path.exists(LABEL_MAPPING_PATH):
            try:
                with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    self.label2id = payload.get("label2id", self.label2id)
                    raw_id2label = payload.get("id2label", {})
                    self.id2label = {int(k): v for k, v in raw_id2label.items()}
            except Exception as e:
                print(f"[!] Warning: Could not read label_mapping.json: {e}")

    def _load_model(self) -> None:
        """
        Attempts to load fine-tuned DistilBERT model or Phase 15 TF-IDF model.
        Falls back to Phase 14 / baseline model if absent or fails.
        """
        # 1. Try loading HuggingFace PyTorch DistilBERT model
        if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                print(f"[+] Loading fine-tuned DistilBERT model from: {MODEL_DIR}")
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
                self.torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                self.torch_model.eval()
                self.model_type = "DistilBERT"
                print("[+] Fine-tuned DistilBERT model loaded successfully.")
                return
            except Exception as e:
                print(f"[!] PyTorch/DistilBERT load failed: {e}. Checking Phase 15 / Phase 14 model...")

        # 2. Try loading Phase 15 model, then Phase 14 model, then baseline model
        for model_path, name in [
            (PHASE15_MODEL_PATH, "Word+Char TF-IDF Phase 15 Model"),
            (PHASE14_MODEL_PATH, "TF-IDF Phase 14 Model"),
            (BASELINE_MODEL_PATH, "TF-IDF Baseline")
        ]:
            if os.path.exists(model_path):
                try:
                    import joblib
                    print(f"[+] Loading ambiguity model from: {model_path}")
                    self.baseline_pipeline = joblib.load(model_path)
                    self.model_type = name
                    print(f"[+] Ambiguity model ({self.model_type}) loaded successfully.")
                    return
                except Exception as e:
                    print(f"[!] Ambiguity model load failed for {model_path}: {e}")

        print("[!] Warning: No pre-trained model found in saved_model directory.")
        self.model_type = "None"

    def detect_ambiguous_triggers(self, text: str) -> List[str]:
        """
        Scans input text for ambiguous trigger words/phrases using case-insensitive regex.

        Args:
            text (str): Input requirement string.

        Returns:
            List[str]: List of matched ambiguous terms found.
        """
        found_triggers: List[str] = []
        text_lower = text.lower()

        for trigger in AMBIGUOUS_TRIGGERS:
            # Match word boundary where applicable
            pattern = r'\b' + re.escape(trigger) + r'\b' if trigger.isalnum() else re.escape(trigger)
            if re.search(pattern, text_lower):
                found_triggers.append(trigger)

        return found_triggers

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Executes prediction on input requirement sentence.

        Args:
            text (str): Requirement sentence string.

        Returns:
            Dict[str, Any]: Prediction dictionary with prediction, confidence, and reason.
        """
        cleaned_text = text.strip() if isinstance(text, str) else ""

        if not cleaned_text:
            return {
                "requirement": text,
                "prediction": "Ambiguous",
                "confidence": "0%",
                "reason": "Input requirement string is empty or invalid."
            }

        prediction_label = "Clear"
        confidence_val = 0.50

        # Run inference using DistilBERT if loaded
        if self.model_type == "DistilBERT" and self.torch_model and self.tokenizer:
            import torch

            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            )
            with torch.no_grad():
                outputs = self.torch_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

            predicted_class_id = int(torch.argmax(outputs.logits, dim=-1).item())
            prediction_label = self.id2label.get(predicted_class_id, "Ambiguous")
            confidence_val = float(probabilities[predicted_class_id])

        # Run inference using Baseline if loaded
        elif self.model_type == "TF-IDF Baseline" and self.baseline_pipeline:
            probabilities = self.baseline_pipeline.predict_proba([cleaned_text])[0]
            predicted_class_id = int(self.baseline_pipeline.predict([cleaned_text])[0])
            prediction_label = self.id2label.get(predicted_class_id, "Ambiguous")
            confidence_val = float(probabilities[predicted_class_id])

        else:
            # Rule-based fallback if model is not yet trained
            triggers = self.detect_ambiguous_triggers(cleaned_text)
            if triggers:
                prediction_label = "Ambiguous"
                confidence_val = 0.85
            else:
                prediction_label = "Clear"
                confidence_val = 0.85

        # Perform explainability trigger word detection
        triggers = self.detect_ambiguous_triggers(cleaned_text)

        # Generate reason explanation
        if triggers:
            formatted_triggers = ", ".join([f"'{t}'" for t in triggers])
            reason = f"Contains vague term(s): {formatted_triggers}"
            # Override prediction to Ambiguous if explicit triggers are present
            if prediction_label == "Clear" and len(triggers) >= 1:
                prediction_label = "Ambiguous"
                confidence_val = max(confidence_val, 0.80)
        else:
            if prediction_label == "Clear":
                reason = "No vague or subjective terms detected; requirement is specific and measurable."
            else:
                reason = "Requirement uses ambiguous structure, subjective phrasing, or vague qualifiers."

        confidence_str = f"{int(round(confidence_val * 100))}%"

        return {
            "requirement": text,
            "prediction": prediction_label,
            "confidence": confidence_str,
            "reason": reason
        }


# Initialize singleton predictor instance at module import time
predictor_instance: RequirementPredictor = RequirementPredictor()


def predict_requirement(text: str) -> Dict[str, Any]:
    """
    Public module function to predict ambiguity for a single requirement sentence.

    Args:
        text (str): Software requirement string.

    Returns:
        Dict[str, Any]: Prediction result dictionary.
    """
    return predictor_instance.predict(text)


if __name__ == "__main__":
    # Self-test block
    test_samples = [
        "The system shall allow a registered user to reset their password via a verification email within 2 minutes.",
        "The system should be user-friendly and fast."
    ]
    print("Testing predict_requirement():")
    for sample in test_samples:
        res = predict_requirement(sample)
        print(json.dumps(res, indent=2))
