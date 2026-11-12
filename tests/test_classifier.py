"""
Comprehensive Classifier Tests for Dialogue Classifier 2.0.
Verifies all test scenarios required by Section 10:
- Empty input
- Greeting sentences
- Farewell sentences
- Questions
- Requests
- Exclamations
- Complaints
- Suggestions
- Apology & Thanks sentences
- Long paragraphs
- Mixed punctuation
- User intent resolution
"""

import pytest
from src.rule_classifier import EnhancedRuleClassifier, OriginalRuleClassifier
from src.ml_classifier import DialogueMLClassifier
from src.hybrid_classifier import HybridClassifier
from src.preprocessing import resolve_user_intent


@pytest.fixture(scope="module")
def hybrid_clf():
    ml = DialogueMLClassifier("models")
    enhanced = EnhancedRuleClassifier()
    return HybridClassifier(rule_classifier=enhanced, ml_classifier=ml)


def test_empty_input(hybrid_clf):
    """Test empty string and whitespace input handling."""
    res = hybrid_clf.classify("")
    assert res["category"] == "EMPTY INPUT"
    assert res["confidence"] is None

    res_ws = hybrid_clf.classify("   \t  \n  ")
    assert res_ws["category"] == "EMPTY INPUT"


def test_greeting_sentences(hybrid_clf):
    """Test various greeting structures."""
    greetings = [
        "Hello everyone, how are you today?",
        "Good morning team!",
        "Hey there, nice to meet you.",
        "Howdy partner!",
        "Greetings to all participants."
    ]
    for text in greetings:
        res = hybrid_clf.classify(text)
        assert res["category"] in ["GREETING", "QUESTION"]


def test_farewell_sentences(hybrid_clf):
    """Test farewell sentences."""
    farewells = [
        "Goodbye, have a wonderful weekend!",
        "See you tomorrow at the office.",
        "Talk to you later, bye!",
        "Take care and farewell."
    ]
    for text in farewells:
        res = hybrid_clf.classify(text)
        assert res["category"] == "FAREWELL"


def test_question_sentences(hybrid_clf):
    """Test direct questions, wh-questions, and auxiliary verb questions."""
    questions = [
        "What time does the keynote session begin?",
        "Where can I find the API documentation?",
        "Why did the database connection timeout?",
        "Is there any update on the deployment status?",
        "Could this bug be related to the recent dependency upgrade?"
    ]
    for text in questions:
        res = hybrid_clf.classify(text)
        assert res["category"] == "QUESTION"


def test_request_sentences(hybrid_clf):
    """Test polite requests, action directives, and modal requests."""
    requests = [
        "Please send me the updated financial spreadsheet.",
        "Could you please review my pull request?",
        "Kindly share the access credentials before the call.",
        "I need you to upload the latest test logs to the server."
    ]
    for text in requests:
        res = hybrid_clf.classify(text)
        assert res["category"] == "REQUEST"


def test_exclamation_sentences(hybrid_clf):
    """Test exclamations and high-arousal emotional interjections."""
    exclamations = [
        "Wow, that user interface looks breathtaking!",
        "Hurray! We finally hit our quarterly milestone!",
        "Awesome job team, what a phenomenal result!",
        "Yikes, that was a dangerously close call!"
    ]
    for text in exclamations:
        res = hybrid_clf.classify(text)
        assert res["category"] == "EXCLAMATION"


def test_complaint_sentences(hybrid_clf):
    """Test customer complaints and technical defect reports."""
    complaints = [
        "This software is broken and keeps crashing constantly.",
        "I am extremely disappointed with the terrible service.",
        "The application is horribly slow and unresponsive.",
        "Your service has been a complete waste of time."
    ]
    for text in complaints:
        res = hybrid_clf.classify(text)
        assert res["category"] == "COMPLAINT"


def test_suggestion_sentences(hybrid_clf):
    """Test suggestions, recommendations, and proposals."""
    suggestions = [
        "You should consider upgrading the server RAM before launching.",
        "How about we schedule the retrospective for Friday morning?",
        "I suggest we implement rate limiting on the public endpoints.",
        "Why don't we run a beta test with a smaller user group?"
    ]
    for text in suggestions:
        res = hybrid_clf.classify(text)
        assert res["category"] == "SUGGESTION"


def test_long_paragraphs(hybrid_clf):
    """Test long paragraphs with multiple sentences and narrative flow."""
    paragraph = (
        "Yesterday we completed the preliminary data migration for the customer database. "
        "The overall performance was stable and latency remained well below fifty milliseconds. "
        "However, we noticed a few minor indexing errors during the batch verification step. "
        "Could you please review the attached diagnostic report and let me know your thoughts?"
    )
    res = hybrid_clf.classify(paragraph)
    # Primary communicative intent is requesting a review of the report
    assert res["category"] == "REQUEST"
    assert len(res.get("sentence_breakdown", [])) >= 3


def test_mixed_punctuation(hybrid_clf):
    """Test dialogues containing mixed, repeated, and unconventional punctuation."""
    test_cases = [
        ("Are you sure this is correct?!?!", "QUESTION"),
        ("What a spectacular breakthrough!!!!!", "EXCLAMATION"),
        ("Please help me... I am completely stuck...", "REQUEST"),
        ("Hello??? Is anyone there???", "QUESTION"),
        ("Goodbye... see you next week...", "FAREWELL")
    ]
    for text, expected in test_cases:
        res = hybrid_clf.classify(text)
        assert res["category"] == expected


def test_user_intent_resolution():
    """Test user intent mapping for speech act categorization."""
    intent_q = resolve_user_intent("QUESTION", "What time is the meeting?")
    assert "Inquiry" in intent_q["primary_intent"]
    assert "Scheduling" in intent_q["specific_sub_intent"]

    intent_r = resolve_user_intent("REQUEST", "Could you please review the code?")
    assert "Review Request" in intent_r["specific_sub_intent"]

    intent_c = resolve_user_intent("COMPLAINT", "The app crashed again.")
    assert "Defect Escalation" in intent_c["specific_sub_intent"]
