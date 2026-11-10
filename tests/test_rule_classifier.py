"""
Tests for Phase 1: Rule-Based Classifiers (Legacy and Enhanced).
Verifies:
1. Legacy original classifier accuracy on historic samples.
2. Enhanced classifier coverage of all 10 categories.
3. Multi-sentence handling and priority conflict resolution.
4. Word boundary precision (no false positives for substrings like 'hi' in 'history').
"""

import pytest
from src.rule_classifier import OriginalRuleClassifier, EnhancedRuleClassifier, legacy_classify_dialogue


def test_legacy_original_classifier_compatibility():
    """Verify legacy logic produces identical behavior to original dialogue_classifier.py."""
    orig = OriginalRuleClassifier()

    # Empty
    assert legacy_classify_dialogue("") == "EMPTY INPUT"
    assert orig.classify("")["category"] == "EMPTY INPUT"

    # Punctuation priority in legacy
    assert legacy_classify_dialogue("Where are you?") == "QUESTION"
    assert orig.classify("Where are you?")["category"] == "QUESTION"

    # In legacy, 'Goodbye!' was EXCLAMATION/EMPHASIS because of '!'
    assert legacy_classify_dialogue("Goodbye!") == "EXCLAMATION/EMPHASIS"
    assert orig.classify("Goodbye!")["category"] == "EXCLAMATION/EMPHASIS"

    # Greeting
    assert legacy_classify_dialogue("Hello there") == "GREETING"
    assert orig.classify("Hello there")["category"] == "GREETING"

    # Request prefix
    assert legacy_classify_dialogue("please help me") == "REQUEST"
    assert orig.classify("please help me")["category"] == "REQUEST"

    # Statement default
    assert legacy_classify_dialogue("I am reading a book.") == "STATEMENT"
    assert orig.classify("I am reading a book.")["category"] == "STATEMENT"


def test_enhanced_classifier_10_categories():
    """Verify EnhancedRuleClassifier accurately identifies all 10 target categories."""
    clf = EnhancedRuleClassifier()

    test_cases = [
        ("Hello, how are you today?", "GREETING"),
        ("Goodbye, see you tomorrow!", "FAREWELL"),
        ("What time does the conference start?", "QUESTION"),
        ("Could you please review the attached document?", "REQUEST"),
        ("Wow, that was an absolutely incredible presentation!", "EXCLAMATION"),
        ("The server has been running steadily since Monday.", "STATEMENT"),
        ("This software keeps crashing and it is completely unacceptable.", "COMPLAINT"),
        ("You should try clearing your browser cache.", "SUGGESTION"),
        ("I am truly sorry for the unexpected delay.", "APOLOGY"),
        ("Thank you so much for your assistance!", "THANKS"),
    ]

    for text, expected_category in test_cases:
        res = clf.classify(text)
        assert res["category"] == expected_category, (
            f"Failed on '{text}'. Expected {expected_category}, got {res['category']}. "
            f"Explanation: {res['explanation']}"
        )
        assert res["is_rule_based"] is True
        assert res["confidence"] is None  # Never fake probabilities for rules!
        assert len(res["explanation"]) > 0


def test_word_boundary_precision():
    """Test that words like 'hi' or 'cya' don't falsely match substrings."""
    clf = EnhancedRuleClassifier()

    # 'hi' inside 'history' or 'this' should NOT trigger GREETING
    res = clf.classify("The history of science is fascinating.")
    assert res["category"] == "STATEMENT"

    # 'bye' inside 'byelorussian' or 'by' should NOT trigger FAREWELL
    res = clf.classify("Standing by the door.")
    assert res["category"] == "STATEMENT"


def test_multi_sentence_handling():
    """Test multi-sentence dialogues resolve to primary communicative intent."""
    clf = EnhancedRuleClassifier()

    # Greeting + Request -> Primary intent is REQUEST
    text = "Good morning team. Could you please send me the financial report?"
    res = clf.classify(text)
    assert res["category"] == "REQUEST"
    assert len(res["sentence_breakdown"]) == 2

    # Greeting + Question -> Primary intent is QUESTION
    text = "Hello! Where is the nearest metro station?"
    res = clf.classify(text)
    assert res["category"] == "QUESTION"
    assert len(res["sentence_breakdown"]) == 2
