"""
Predictor Module for Prompt Quality Assessment.
Loads the trained Random Forest model artifact and generates quality scores,
categorical ratings, suggestions, and improved prompts.
"""
import os
import sys
import numpy as np
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.feature_extraction import get_feature_vector, extract_features
from utils.suggestions import generate_suggestions, generate_improved_prompt

MODEL_FILE = os.path.join(PROJECT_ROOT, "model", "prompt_quality_model.pkl")

def get_quality_category(score: float) -> str:
    """
    Categorizes the overall quality score:
      0–39   → Poor
      40–59  → Needs Improvement
      60–79  → Good
      80–100 → Excellent
    """
    if score < 40:
        return "Poor"
    elif score < 60:
        return "Needs Improvement"
    elif score < 80:
        return "Good"
    else:
        return "Excellent"

class PromptQualityPredictor:
    def __init__(self, model_path: str = MODEL_FILE):
        self.model_path = model_path
        self.model = None
        self.metrics = {}
        self.load_model()

    def load_model(self):
        """Loads trained Random Forest model from disk."""
        if os.path.exists(self.model_path):
            try:
                payload = joblib.load(self.model_path)
                if isinstance(payload, dict) and "model" in payload:
                    self.model = payload["model"]
                    self.metrics = payload.get("metrics", {})
                else:
                    self.model = payload
            except Exception as e:
                print(f"Warning: Could not load model from {self.model_path}: {e}")
                self.model = None

    def predict(self, prompt: str, domain: str = "General") -> dict:
        """
        Evaluates a prompt in the context of an enterprise domain (HR, Finance, Marketing, General)
        and returns scores, category, domain, suggestions, and improved prompt.
        """
        clean = (prompt or "").strip()
        valid_domains = ["HR", "Finance", "Marketing", "General"]
        domain = domain if domain in valid_domains else "General"

        if not clean:
            return {
                "error": "Prompt cannot be empty.",
                "domain": domain,
                "clarity": 0.0,
                "specificity": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "overall_score": 0,
                "category": "Poor",
                "suggestions": ["Please enter a prompt to assess."],
                "improved_prompt": ""
            }

        # 1. Feature extraction with domain alignment
        feats = extract_features(clean, domain=domain)
        vec = np.array([get_feature_vector(clean, domain=domain)])

        # 2. Model prediction
        if self.model is None:
            self.load_model()

        if self.model is not None:
            preds = self.model.predict(vec)[0]
            clarity = float(np.clip(preds[0], 0, 100))
            specificity = float(np.clip(preds[1], 0, 100))
            completeness = float(np.clip(preds[2], 0, 100))
            relevance = float(np.clip(preds[3], 0, 100))
        else:
            # Fallback heuristic calculation if model file is missing
            clarity = min(100.0, feats["word_count"] * 4.0 + feats["instruction_words"] * 10.0)
            specificity = min(100.0, feats["specificity_keywords"] * 15.0 + feats["example_indicators"] * 15.0)
            completeness = min(100.0, feats["sentence_count"] * 20.0 + feats["context_indicators"] * 15.0)
            relevance = min(100.0, 40.0 + feats["domain_relevance"] * 12.0 + feats["instruction_words"] * 8.0)

        # 3. Overall Score formula:
        # Overall Score = 0.30 * Clarity + 0.30 * Specificity + 0.20 * Completeness + 0.20 * Relevance
        overall_score = (
            0.30 * clarity +
            0.30 * specificity +
            0.20 * completeness +
            0.20 * relevance
        )
        overall_score = int(round(float(np.clip(overall_score, 0.0, 100.0))))

        # 4. Determine category
        category = get_quality_category(overall_score)

        # 5. Rule-based domain-specific suggestions
        dimensions = {
            "clarity": clarity,
            "specificity": specificity,
            "completeness": completeness,
            "relevance": relevance
        }
        suggestions = generate_suggestions(clean, domain=domain, dimensions=dimensions)

        # 6. Deterministic domain-specific improved prompt
        improved_prompt = generate_improved_prompt(clean, domain=domain, overall_score=overall_score)

        return {
            "prompt": clean,
            "domain": domain,
            "clarity": round(clarity, 1),
            "specificity": round(specificity, 1),
            "completeness": round(completeness, 1),
            "relevance": round(relevance, 1),
            "overall_score": overall_score,
            "category": category,
            "features": feats,
            "suggestions": suggestions,
            "improved_prompt": improved_prompt
        }
