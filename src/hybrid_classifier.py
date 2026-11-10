"""
Hybrid Classification Engine for Dialogue Classifier 2.0.

Combines rule-based pattern matching and machine learning probability estimation
with transparent conflict resolution, provenance tracking, and clear explanations.
"""

from typing import Dict, Any, Optional

from src.rule_classifier import EnhancedRuleClassifier
from src.ml_classifier import DialogueMLClassifier


class HybridClassifier:
    """
    Intelligent Hybrid Classifier.
    Evaluates both rule-based heuristics and machine learning models, resolving conflicts
    via transparent provenance rules:
    - High-confidence structural rules (e.g. strict greetings, apologies, thanks, polite requests)
      take precedence over ambiguous ML predictions.
    - Strong ML confidence overrides default declarative rule fallbacks.
    - Consensus agreement between both systems reinforces classification confidence.
    """

    def __init__(
        self,
        rule_classifier: Optional[EnhancedRuleClassifier] = None,
        ml_classifier: Optional[DialogueMLClassifier] = None,
        ml_confidence_threshold: float = 0.65
    ):
        self.rule_classifier = rule_classifier or EnhancedRuleClassifier()
        self.ml_classifier = ml_classifier or DialogueMLClassifier()
        self.ml_confidence_threshold = ml_confidence_threshold
        self.name = "Hybrid Rule-ML Classifier (v2.0)"

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text by combining rule heuristics with ML predictions.
        """
        if not text or text.isspace():
            return {
                "category": "EMPTY INPUT",
                "confidence": None,
                "confidence_score": 0.0,
                "probabilities": None,
                "explanation": "No text provided to classify.",
                "matched_keywords": [],
                "matched_patterns": [],
                "mode": "hybrid",
                "subsystem_used": "none",
                "is_rule_based": True
            }

        rule_result = self.rule_classifier.classify(text)

        # Fallback if ML is not trained
        if not self.ml_classifier.is_trained:
            rule_result["mode"] = "hybrid"
            rule_result["subsystem_used"] = "rule_engine"
            rule_result["explanation"] = (
                f"ML model is not trained yet. Defaulted gracefully to Enhanced Rule Engine: "
                f"{rule_result['explanation']}"
            )
            return rule_result

        ml_result = self.ml_classifier.predict(text)

        rule_cat = rule_result["category"]
        rule_score = rule_result.get("confidence_score", 0.70)
        ml_cat = ml_result["category"]
        ml_conf = ml_result.get("confidence", 0.0) or 0.0

        # Case 1: Full Consensus between ML and Rule Engine
        if rule_cat == ml_cat:
            explanation = (
                f"Consensus reached: Both the Enhanced Rule Engine and the {ml_result['model_name']} "
                f"agreed on '{rule_cat}' (ML probability: {round(ml_conf * 100, 1)}%)."
            )
            return {
                "category": rule_cat,
                "confidence": ml_conf,
                "confidence_score": ml_conf,
                "probabilities": ml_result.get("probabilities"),
                "explanation": explanation,
                "matched_keywords": list(set(rule_result.get("matched_keywords", []) + ml_result.get("matched_keywords", []))),
                "matched_patterns": rule_result.get("matched_patterns", []),
                "mode": "hybrid",
                "subsystem_used": "hybrid_consensus",
                "is_rule_based": False,
                "sentence_breakdown": rule_result.get("sentence_breakdown", [])
            }

        # Case 2: Explicit high-priority rule patterns (GREETING, FAREWELL, THANKS, APOLOGY, REQUEST)
        # where specific linguistic phrases exist and rule score >= 0.90
        high_precision_rule_categories = {"GREETING", "FAREWELL", "THANKS", "APOLOGY", "REQUEST"}
        if rule_cat in high_precision_rule_categories and rule_score >= 0.90 and ml_conf < 0.88:
            explanation = (
                f"Rule Engine prioritized: Detected explicit speech act pattern for '{rule_cat}' "
                f"({rule_result['explanation']}). ML predicted '{ml_cat}' with {round(ml_conf * 100, 1)}%."
            )
            return {
                "category": rule_cat,
                "confidence": None,  # Transparent: rule match, not fake probability
                "confidence_score": rule_score,
                "probabilities": ml_result.get("probabilities"),
                "explanation": explanation,
                "matched_keywords": rule_result.get("matched_keywords", []),
                "matched_patterns": rule_result.get("matched_patterns", []),
                "mode": "hybrid",
                "subsystem_used": "rule_engine",
                "is_rule_based": True,
                "sentence_breakdown": rule_result.get("sentence_breakdown", [])
            }

        # Case 3: Rule engine returned default STATEMENT fallback, but ML has strong conviction
        if rule_cat == "STATEMENT" and ml_conf >= self.ml_confidence_threshold:
            explanation = (
                f"ML Model prioritized: {ml_result['model_name']} classified dialogue as '{ml_cat}' "
                f"with {round(ml_conf * 100, 1)}% probability over the rule engine's default declarative fallback."
            )
            return {
                "category": ml_cat,
                "confidence": ml_conf,
                "confidence_score": ml_conf,
                "probabilities": ml_result.get("probabilities"),
                "explanation": explanation,
                "matched_keywords": ml_result.get("matched_keywords", []),
                "matched_patterns": ml_result.get("matched_patterns", []),
                "mode": "hybrid",
                "subsystem_used": "ml_model",
                "is_rule_based": False,
                "sentence_breakdown": rule_result.get("sentence_breakdown", [])
            }

        # Case 4: ML has very high confidence (>= 0.85)
        if ml_conf >= 0.85:
            explanation = (
                f"ML Model prioritized: {ml_result['model_name']} predicted '{ml_cat}' with high confidence "
                f"({round(ml_conf * 100, 1)}%), superseding rule '{rule_cat}'."
            )
            return {
                "category": ml_cat,
                "confidence": ml_conf,
                "confidence_score": ml_conf,
                "probabilities": ml_result.get("probabilities"),
                "explanation": explanation,
                "matched_keywords": ml_result.get("matched_keywords", []),
                "matched_patterns": ml_result.get("matched_patterns", []),
                "mode": "hybrid",
                "subsystem_used": "ml_model",
                "is_rule_based": False,
                "sentence_breakdown": rule_result.get("sentence_breakdown", [])
            }

        # Case 5: Default to Rule Engine with explanation
        explanation = (
            f"Rule Engine selected: Rule heuristics identified '{rule_cat}' while ML prediction for "
            f"'{ml_cat}' remained below decisive confidence threshold ({round(ml_conf * 100, 1)}%)."
        )
        return {
            "category": rule_cat,
            "confidence": None,
            "confidence_score": rule_score,
            "probabilities": ml_result.get("probabilities"),
            "explanation": explanation,
            "matched_keywords": rule_result.get("matched_keywords", []),
            "matched_patterns": rule_result.get("matched_patterns", []),
            "mode": "hybrid",
            "subsystem_used": "rule_engine",
            "is_rule_based": True,
            "sentence_breakdown": rule_result.get("sentence_breakdown", [])
        }
