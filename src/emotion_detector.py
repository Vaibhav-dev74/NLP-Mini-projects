"""
Emotion Detection Module for Dialogue Classifier 2.0.

Detects basic affective states: HAPPY, SAD, ANGRY, FEAR, SURPRISE, NEUTRAL.
Clearly distinguishes rule-based affective matching from trained deep neural models.
"""

import re
from typing import Dict, Any, List

# Lexical mapping for affective classes
EMOTION_LEXICON: Dict[str, List[str]] = {
    "HAPPY": [
        "happy", "joy", "delight", "glad", "pleased", "excited", "love", "awesome",
        "great", "wonderful", "celebrate", "congrats", "congratulations", "fantastic",
        "superb", "brilliant", "cheer", "cheers", "yay", "hurray", "hooray", "smiling", "fun"
    ],
    "SAD": [
        "sad", "unhappy", "depressed", "sorrow", "grief", "miserable", "crying",
        "tears", "heartbroken", "down", "gloomy", "disappointed", "lonely", "pity",
        "regret", "mourn", "hurt", "hopeless", "despair"
    ],
    "ANGRY": [
        "angry", "mad", "furious", "outraged", "irritated", "annoyed", "pissed",
        "rage", "hate", "frustrated", "hostile", "bitter", "disgusted", "offensive",
        "unacceptable", "terrible", "worst", "screwed", "ridiculous"
    ],
    "FEAR": [
        "afraid", "scared", "fear", "terrified", "panic", "worried", "anxious",
        "nervous", "dread", "frightened", "horrified", "threat", "danger", "insecure", "alarm"
    ],
    "SURPRISE": [
        "surprise", "surprised", "shocked", "amazed", "astonished", "stunned",
        "unbelievable", "wow", "omg", "whoa", "unexpected", "incredible", "speechless", "yikes"
    ]
}


class EmotionDetector:
    """
    Keyword and affective marker detector for dialogue text.
    Classifies into 6 distinct categories: HAPPY, SAD, ANGRY, FEAR, SURPRISE, NEUTRAL.
    """

    def __init__(self):
        self.name = "Affective Lexicon Emotion Detector"
        self.disclaimer = (
            "Rule-Based Affective Lexicon Matcher. Provides fast, interpretable keyword-based "
            "emotion scoring without heavy deep-learning dependencies."
        )

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detects primary emotion and returns distributions across all 6 classes.
        """
        if not text or text.isspace():
            return {
                "emotion": "NEUTRAL",
                "confidence": 0.5,
                "scores": {e: 0.0 for e in ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "NEUTRAL"]},
                "detected_markers": {},
                "explanation": "No text provided for emotion detection.",
                "disclaimer": self.disclaimer,
                "model_type": "Rule-Based Lexicon"
            }

        tokens = set(re.findall(r"\b[\w'-]+\b", text.lower()))
        scores = {e: 0.0 for e in ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "NEUTRAL"]}
        detected_markers = {}

        for emotion, keywords in EMOTION_LEXICON.items():
            matched = [k for k in keywords if k in tokens]
            if matched:
                scores[emotion] = len(matched) * 1.5
                detected_markers[emotion] = matched

        # Punctuation heuristics
        if "!" in text:
            # Exclamations boost SURPRISE, HAPPY, or ANGRY depending on context
            if scores["HAPPY"] > 0:
                scores["HAPPY"] += 0.5
            if scores["ANGRY"] > 0:
                scores["ANGRY"] += 0.5
            if scores["SURPRISE"] > 0:
                scores["SURPRISE"] += 0.8
            elif not detected_markers:
                scores["SURPRISE"] += 0.5

        if "?" in text and not detected_markers:
            scores["SURPRISE"] += 0.3

        total_score = sum(scores.values())

        if total_score == 0:
            scores["NEUTRAL"] = 1.0
            dominant_emotion = "NEUTRAL"
            confidence = 0.70
            explanation = "No strong emotional markers detected; baseline neutral affect."
        else:
            # Normalize scores to pseudo-probabilities
            norm_scores = {e: round(s / total_score, 3) for e, s in scores.items()}
            # If highest score is weak, boost neutral
            sorted_emotions = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
            dominant_emotion = sorted_emotions[0][0]
            confidence = min(0.95, round(0.55 + sorted_emotions[0][1] * 0.4, 2))

            markers = detected_markers.get(dominant_emotion, [])
            if markers:
                explanation = f"Detected emotion '{dominant_emotion}' prompted by cues: {', '.join(markers)}."
            else:
                explanation = f"Emotion '{dominant_emotion}' indicated by structural and punctuation cues."

        # Ensure normalized representation
        s_sum = sum(scores.values()) or 1.0
        normalized_distribution = {e: round(scores[e] / s_sum, 3) for e in scores}

        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "scores": normalized_distribution,
            "detected_markers": detected_markers,
            "explanation": explanation,
            "disclaimer": self.disclaimer,
            "model_type": "Rule-Based Lexicon"
        }
