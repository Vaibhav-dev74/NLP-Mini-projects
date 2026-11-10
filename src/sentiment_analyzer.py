"""
Sentiment Analysis Module for Dialogue Classifier 2.0.

Provides rule and lexicon-based sentiment analysis with negation handling,
valence intensification, polarity scoring (-1.0 to +1.0), and categorical classification
into POSITIVE, NEGATIVE, or NEUTRAL.
"""

import re
from typing import Dict, Any, List, Set, Tuple

# Core sentiment lexicon
POSITIVE_WORDS: Dict[str, float] = {
    "great": 0.8, "excellent": 0.9, "amazing": 0.9, "awesome": 0.85, "good": 0.6,
    "fantastic": 0.9, "wonderful": 0.85, "happy": 0.75, "love": 0.9, "best": 0.85,
    "brilliant": 0.85, "superb": 0.9, "glad": 0.65, "pleased": 0.7, "delighted": 0.85,
    "perfect": 0.95, "thank": 0.6, "thanks": 0.6, "appreciate": 0.7, "grateful": 0.8,
    "helpful": 0.65, "nice": 0.55, "friendly": 0.6, "fast": 0.5, "smooth": 0.6,
    "clean": 0.5, "solid": 0.55, "cool": 0.6, "yay": 0.75, "hurray": 0.8, "bravo": 0.85,
    "outstanding": 0.9, "remarkable": 0.8, "terrific": 0.85, "kind": 0.6, "kindly": 0.5
}

NEGATIVE_WORDS: Dict[str, float] = {
    "bad": -0.6, "terrible": -0.9, "horrible": -0.9, "awful": -0.85, "worst": -0.95,
    "poor": -0.6, "broken": -0.75, "hate": -0.85, "disappointed": -0.8, "disappointing": -0.75,
    "unacceptable": -0.9, "frustrated": -0.75, "annoying": -0.7, "annoyed": -0.7,
    "useless": -0.8, "waste": -0.75, "slow": -0.5, "crash": -0.7, "crashed": -0.7,
    "error": -0.5, "failed": -0.7, "failure": -0.75, "bug": -0.4, "defect": -0.65,
    "damage": -0.7, "damaged": -0.7, "defective": -0.75, "wrong": -0.5, "painful": -0.7,
    "ugly": -0.6, "dirty": -0.55, "confusing": -0.5, "difficult": -0.45, "delay": -0.5,
    "delayed": -0.5, "loss": -0.6, "lost": -0.55, "angry": -0.8, "upset": -0.7, "sorry": -0.3
}

INTENSIFIERS: Dict[str, float] = {
    "very": 1.4, "extremely": 1.7, "incredibly": 1.6, "super": 1.5,
    "deeply": 1.5, "truly": 1.4, "really": 1.3, "so": 1.3,
    "absolutely": 1.6, "completely": 1.5, "highly": 1.4, "exceptionally": 1.6
}

DIMINISHERS: Dict[str, float] = {
    "slightly": 0.6, "barely": 0.5, "hardly": 0.4, "somewhat": 0.7, "a bit": 0.7, "little": 0.7
}

NEGATIONS: Set[str] = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't",
    "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't"
}


class SentimentAnalyzer:
    """
    Intelligent lexicon-based sentiment analyzer with contextual negation detection,
    valence modification (intensifiers/diminishers), and confidence estimation.
    """

    def __init__(self):
        self.name = "Lexicon-Contextual Sentiment Analyzer"

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the polarity and intensity of user dialogue.
        Returns:
            category: POSITIVE, NEGATIVE, or NEUTRAL
            polarity_score: float in range [-1.0, 1.0]
            confidence: estimation of confidence in verdict
            intensity: "Weak", "Moderate", "Strong"
            positive_words: matched positive tokens
            negative_words: matched negative tokens
            explanation: detailed human explanation
        """
        if not text or text.isspace():
            return {
                "sentiment": "NEUTRAL",
                "polarity_score": 0.0,
                "confidence": 0.5,
                "intensity": "Neutral",
                "positive_words": [],
                "negative_words": [],
                "explanation": "No text provided for sentiment analysis."
            }

        tokens = re.findall(r"\b[\w'-]+\b", text.lower())
        if not tokens:
            return {
                "sentiment": "NEUTRAL",
                "polarity_score": 0.0,
                "confidence": 0.5,
                "intensity": "Neutral",
                "positive_words": [],
                "negative_words": [],
                "explanation": "No alphanumeric tokens found."
            }

        scores: List[float] = []
        positive_words: List[str] = []
        negative_words: List[str] = []

        window_size = 3
        for i, word in enumerate(tokens):
            val = None
            if word in POSITIVE_WORDS:
                val = POSITIVE_WORDS[word]
            elif word in NEGATIVE_WORDS:
                val = NEGATIVE_WORDS[word]

            if val is not None:
                # Look backwards for negations and intensifiers within window
                lookback = tokens[max(0, i - window_size):i]
                negated = any(nw in NEGATIONS for nw in lookback)
                
                multiplier = 1.0
                for lb_word in lookback:
                    if lb_word in INTENSIFIERS:
                        multiplier *= INTENSIFIERS[lb_word]
                    elif lb_word in DIMINISHERS:
                        multiplier *= DIMINISHERS[lb_word]

                effective_score = val * multiplier
                if negated:
                    # Invert polarity with damping
                    effective_score = -0.75 * effective_score

                scores.append(effective_score)
                if effective_score > 0:
                    positive_words.append(word)
                elif effective_score < 0:
                    negative_words.append(word)

        if not scores:
            polarity_score = 0.0
            sentiment = "NEUTRAL"
            confidence = 0.65
            intensity = "Neutral"
            explanation = "Objective or neutral tone with no significant affective or opinion markers."
        else:
            raw_sum = sum(scores)
            # Normalize bounded in [-1.0, 1.0]
            polarity_score = max(-1.0, min(1.0, raw_sum / (len(scores) + 0.5)))
            polarity_score = round(polarity_score, 3)

            if polarity_score > 0.15:
                sentiment = "POSITIVE"
                intensity = "Strong" if polarity_score >= 0.55 else ("Moderate" if polarity_score >= 0.3 else "Mild")
                confidence = round(min(0.98, 0.60 + abs(polarity_score) * 0.4), 2)
                explanation = f"Detected positive tone driven by words: {', '.join(set(positive_words))}."
            elif polarity_score < -0.15:
                sentiment = "NEGATIVE"
                intensity = "Strong" if polarity_score <= -0.55 else ("Moderate" if polarity_score <= -0.3 else "Mild")
                confidence = round(min(0.98, 0.60 + abs(polarity_score) * 0.4), 2)
                explanation = f"Detected negative tone driven by words: {', '.join(set(negative_words))}."
            else:
                sentiment = "NEUTRAL"
                intensity = "Mild"
                confidence = 0.70
                explanation = "Balanced or mild sentiment with balanced or minimal polarity."

        return {
            "sentiment": sentiment,
            "polarity_score": polarity_score,
            "confidence": confidence,
            "intensity": intensity,
            "positive_words": list(set(positive_words)),
            "negative_words": list(set(negative_words)),
            "explanation": explanation
        }
