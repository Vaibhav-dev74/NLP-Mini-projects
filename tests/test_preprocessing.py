"""
Unit tests for the Text Preprocessing and Statistics Module.
"""

import pytest
from src.preprocessing import TextPreprocessor


def test_empty_input():
    preprocessor = TextPreprocessor()
    stats = preprocessor.get_text_statistics("")
    assert stats["word_count"] == 0
    assert stats["character_count"] == 0
    assert stats["sentence_count"] == 0
    assert preprocessor.clean_text("") == ""
    assert preprocessor.tokenize("") == []


def test_whitespace_and_unicode_normalization():
    preprocessor = TextPreprocessor()
    raw = "  Hello \t\n world!   ‘This’ is a “quote”.  "
    cleaned = preprocessor.clean_text(raw, preserve_key_punctuation=True, lowercase=True)
    assert "  " not in cleaned
    assert "‘" not in cleaned
    assert "“" not in cleaned
    assert "hello" in cleaned


def test_key_punctuation_preservation():
    preprocessor = TextPreprocessor()
    text = "Can you help me? Yes, absolutely!"
    
    # Preserve key punctuation
    preserved = preprocessor.clean_text(text, preserve_key_punctuation=True)
    assert "?" in preserved
    assert "!" in preserved

    # ML format
    ml_cleaned = preprocessor.clean_for_ml(text)
    assert "__question__" in ml_cleaned
    assert "__exclamation__" in ml_cleaned


def test_sentence_splitting():
    preprocessor = TextPreprocessor()
    text = "Good morning! How are you doing today? I hope all is well."
    sentences = preprocessor.split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Good morning!"
    assert sentences[1] == "How are you doing today?"
    assert sentences[2] == "I hope all is well."


def test_text_statistics():
    preprocessor = TextPreprocessor()
    text = "Hello world! How are you doing today? Great news."
    stats = preprocessor.get_text_statistics(text)

    assert stats["word_count"] == 9
    assert stats["sentence_count"] == 3
    assert stats["question_marks"] == 1
    assert stats["exclamation_marks"] == 1
    assert stats["character_count"] == len(text)
    assert stats["avg_word_length"] > 0
    assert stats["lexical_diversity"] > 0
