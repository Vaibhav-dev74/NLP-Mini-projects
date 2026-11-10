"""
Tests for Sentiment Analysis and Emotion Detection Modules.
"""

import pytest
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_detector import EmotionDetector


def test_sentiment_positive_negative_neutral():
    analyzer = SentimentAnalyzer()

    # Positive
    res_pos = analyzer.analyze("I am extremely pleased and delighted with this wonderful service!")
    assert res_pos["sentiment"] == "POSITIVE"
    assert res_pos["polarity_score"] > 0.3
    assert len(res_pos["positive_words"]) > 0

    # Negative with negation
    res_neg = analyzer.analyze("This product is not good at all, it completely broke down.")
    assert res_neg["sentiment"] == "NEGATIVE"
    assert res_neg["polarity_score"] < -0.15

    # Neutral
    res_neu = analyzer.analyze("The meeting is scheduled for tomorrow at two o'clock.")
    assert res_neu["sentiment"] == "NEUTRAL"
    assert abs(res_neu["polarity_score"]) <= 0.15


def test_emotion_detection():
    detector = EmotionDetector()

    # Happy
    res_happy = detector.detect("Yay, this is fantastic news! I am so excited and happy!")
    assert res_happy["emotion"] == "HAPPY"
    assert "happy" in res_happy["detected_markers"].get("HAPPY", []) or "excited" in res_happy["detected_markers"].get("HAPPY", [])

    # Angry
    res_angry = detector.detect("I am furious and outraged by this terrible service!")
    assert res_angry["emotion"] == "ANGRY"

    # Neutral
    res_neutral = detector.detect("The document contains five chapters.")
    assert res_neutral["emotion"] == "NEUTRAL"
    assert res_neutral["disclaimer"] is not None
