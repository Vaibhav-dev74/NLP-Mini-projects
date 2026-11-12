"""
Tests for Machine Learning and Hybrid Classification Engines.
"""

import os
import pytest
from src.ml_classifier import DialogueMLClassifier
from src.hybrid_classifier import HybridClassifier
from src.rule_classifier import EnhancedRuleClassifier


@pytest.fixture(scope="module")
def trained_ml():
    """Train ML model once for testing."""
    classifier = DialogueMLClassifier(models_dir="models")
    classifier.train("data/dialogue_dataset.csv")
    return classifier


def test_ml_training_and_evaluation(trained_ml):
    """Test model training produces valid models and non-zero metrics."""
    assert trained_ml.is_trained is True
    assert trained_ml.model is not None
    assert trained_ml.vectorizer is not None

    metadata = trained_ml.metadata
    assert "models_comparison" in metadata
    assert "Logistic Regression" in metadata["models_comparison"]
    assert "Multinomial Naive Bayes" in metadata["models_comparison"]
    assert metadata["selected_accuracy"] > 0.60
    assert metadata["selected_f1_score"] > 0.60


def test_ml_prediction_probabilities(trained_ml):
    """Test predictions return real probabilities summing to 1.0."""
    result = trained_ml.predict("Could you please send me the final report?")
    assert result["category"] in DialogueMLClassifier.SUPPORTED_CATEGORIES
    assert result["is_rule_based"] is False
    assert result["confidence"] is not None
    assert 0.0 <= result["confidence"] <= 1.0

    # Ensure probabilities dictionary exists and sums to ~1.0
    probs = result["probabilities"]
    assert probs is not None
    assert len(probs) == len(DialogueMLClassifier.SUPPORTED_CATEGORIES)
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.05


def test_hybrid_classification(trained_ml):
    """Test HybridClassifier integrates rules and ML with provenance explanation."""
    rule_clf = EnhancedRuleClassifier()
    hybrid = HybridClassifier(rule_classifier=rule_clf, ml_classifier=trained_ml)

    # Greeting test
    res = hybrid.classify("Hello there, good morning!")
    assert res["category"] == "GREETING"
    assert res["mode"] == "hybrid"
    assert res["subsystem_used"] in ["hybrid_consensus", "rule_engine", "ml_model"]
    assert len(res["explanation"]) > 0

    # Empty test
    res_empty = hybrid.classify("")
    assert res_empty["category"] == "EMPTY INPUT"


def test_unknown_categories_and_invalid_dataset(tmp_path):
    """Test validation catches unknown categories and missing columns."""
    clf = DialogueMLClassifier("models")

    # 1. Unknown category label
    invalid_csv = tmp_path / "invalid_cat.csv"
    invalid_csv.write_text("text,category\nHello,UNKNOWN_LABEL\nGood morning,GREETING\n")
    with pytest.raises(ValueError, match="unsupported category labels"):
        clf.load_dataset(str(invalid_csv))

    # 2. Missing required columns
    missing_cols_csv = tmp_path / "missing_cols.csv"
    missing_cols_csv.write_text("sentence,label\nHello,GREETING\n")
    with pytest.raises(ValueError, match="missing mandatory column"):
        clf.load_dataset(str(missing_cols_csv))

