"""
Text Preprocessing and Linguistic Statistics Module.
Provides modular text cleaning, tokenization, sentence segmentation,
punctuation preservation, stopword filtering, and comprehensive text statistics.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set

# Standard English dialogue stopwords (pronouns and modal verbs can be retained for dialogue intent)
STANDARD_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "no", "nor", "not", "only", "own", "same", "too", "very", "s", "t", "can",
    "will", "don", "should", "now"
}

# Dialogue-safe stopwords (preserves question words and conversational pronouns)
DIALOGUE_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "so", "then", "just", "very",
    "too", "to", "from", "in", "out", "on", "off", "over", "under", "of",
    "at", "by", "for", "with", "about", "against", "between", "into"
}

# Standard Intent and Pragmatic Speech Act Taxonomy
INTENT_TAXONOMY: Dict[str, Dict[str, str]] = {
    "GREETING": {
        "primary_intent": "Initiate Conversation / Social Salutation",
        "description": "User opens a conversational channel and establishes social rapport."
    },
    "FAREWELL": {
        "primary_intent": "Conversation Termination / Parting Well-Wishes",
        "description": "User signals closure of conversational dialogue and offers well-wishes."
    },
    "QUESTION": {
        "primary_intent": "Information Seeking / Clarification Inquiry",
        "description": "User seeks information, explanations, facts, or confirmation."
    },
    "REQUEST": {
        "primary_intent": "Action Directive / Assistance Seeking",
        "description": "User requests specific actions, assistance, resources, or services."
    },
    "EXCLAMATION": {
        "primary_intent": "Emotional Reaction / Discourse Emphasis",
        "description": "User expresses high-arousal emotional response or dramatic emphasis."
    },
    "STATEMENT": {
        "primary_intent": "Information Sharing / Fact Assertion",
        "description": "User provides declarative observations, facts, status, or context."
    },
    "COMPLAINT": {
        "primary_intent": "Dissatisfaction Escalation / Issue Reporting",
        "description": "User reports grievances, bugs, poor service, or failed expectations."
    },
    "SUGGESTION": {
        "primary_intent": "Advisory Proposal / Recommendation",
        "description": "User proposes ideas, optimizations, advice, or alternative courses of action."
    },
    "APOLOGY": {
        "primary_intent": "Social Repair / Regret Expression",
        "description": "User acknowledges an error, inconvenience, or delayed response with regret."
    },
    "THANKS": {
        "primary_intent": "Gratitude Expression / Social Closure",
        "description": "User expresses appreciation, indebtedness, or acknowledgment of helpfulness."
    },
    "EMPTY INPUT": {
        "primary_intent": "Null Input / No Communicative Intent",
        "description": "No text provided to discern intent."
    }
}


def resolve_user_intent(category: str, text: str = "") -> Dict[str, str]:
    """
    Maps the predicted dialogue category and text cues to specific user intent,
    communicative purpose, and domain action type.
    """
    cat_key = category.upper() if category else "STATEMENT"
    base = INTENT_TAXONOMY.get(cat_key, {
        "primary_intent": f"General {cat_key} Intent",
        "description": f"User engaged in {cat_key.lower()} dialogue act."
    })

    sub_intent = base["primary_intent"]
    t_lower = text.lower() if text else ""

    # Fine-grained contextual sub-intent detection
    if cat_key == "QUESTION":
        if any(w in t_lower for w in ["what time", "when", "schedule"]):
            sub_intent = "Temporal / Scheduling Inquiry"
        elif any(w in t_lower for w in ["where", "location", "path", "find"]):
            sub_intent = "Resource / Location Inquiry"
        elif any(w in t_lower for w in ["why", "cause", "reason"]):
            sub_intent = "Causality / Diagnostic Inquiry"
        elif any(w in t_lower for w in ["how", "steps", "procedure"]):
            sub_intent = "Procedural / Implementation Inquiry"
        elif any(w in t_lower for w in ["who", "responsible", "owner"]):
            sub_intent = "Ownership / Accountability Inquiry"
    elif cat_key == "REQUEST":
        if any(w in t_lower for w in ["review", "pr", "pull request", "inspect"]):
            sub_intent = "Code / Document Review Request"
        elif any(w in t_lower for w in ["help", "assist", "support"]):
            sub_intent = "Troubleshooting / Assistance Request"
        elif any(w in t_lower for w in ["send", "share", "upload", "provide"]):
            sub_intent = "Asset / Data Transfer Request"
        elif any(w in t_lower for w in ["restart", "deploy", "run", "execute"]):
            sub_intent = "System / DevOps Action Request"
    elif cat_key == "COMPLAINT":
        if any(w in t_lower for w in ["crash", "freeze", "bug", "broken", "glitch"]):
            sub_intent = "Technical Stability / Defect Escalation"
        elif any(w in t_lower for w in ["delay", "late", "waiting", "hours"]):
            sub_intent = "SLA Breach / Delivery Delay Complaint"
        elif any(w in t_lower for w in ["service", "support", "agent", "nobody"]):
            sub_intent = "Customer Service Quality Complaint"
        elif any(w in t_lower for w in ["charge", "billed", "invoice", "money", "payment"]):
            sub_intent = "Billing & Financial Dispute"

    return {
        "primary_intent": base["primary_intent"],
        "specific_sub_intent": sub_intent,
        "description": base["description"]
    }



class TextPreprocessor:
    """
    Modular text preprocessor for dialogue classification, ML vectorization,
    and linguistic statistical analysis.
    """

    def __init__(self, custom_stopwords: Optional[Set[str]] = None):
        self.stopwords = custom_stopwords if custom_stopwords is not None else DIALOGUE_STOPWORDS

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces, tabs, and newlines into single spaces."""
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def unicode_normalize(text: str) -> str:
        """Standardize unicode characters (e.g. smart quotes, em-dashes)."""
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        # Normalize curved quotes and apostrophes to standard ascii
        text = text.replace('‘', "'").replace('’', "'").replace('“', '"').replace('”', '"')
        text = text.replace('—', ' - ').replace('–', ' - ')
        return text

    def clean_text(
        self,
        text: str,
        preserve_key_punctuation: bool = True,
        lowercase: bool = True
    ) -> str:
        """
        Cleans text while giving the option to preserve communicative punctuation ('?', '!').
        """
        if not text:
            return ""

        text = self.unicode_normalize(text)

        if lowercase:
            text = text.lower()

        if preserve_key_punctuation:
            # Remove symbols except alphanumeric, whitespace, and '?' '!' '.' ',' "'"
            text = re.sub(r"[^\w\s\?!.,']", ' ', text)
            # Ensure spacing around terminal punctuation for clean tokenization if attached
            text = re.sub(r'([?!]+)', r' \1 ', text)
        else:
            # Strip all punctuation and special characters
            text = re.sub(r'[^\w\s]', ' ', text)

        return self.normalize_whitespace(text)

    def clean_for_ml(self, text: str) -> str:
        """
        Preprocessing pipeline specifically formatted for TF-IDF vectorization.
        Retains key communicative punctuation tokens ('?' and '!') as distinct tokens.
        """
        if not text:
            return ""
        cleaned = self.clean_text(text, preserve_key_punctuation=True, lowercase=True)
        # Convert standalone '?' and '!' to explicit feature tokens: __question__ __exclamation__
        tokens = cleaned.split()
        transformed = []
        for t in tokens:
            if '?' in t:
                transformed.append("__question__")
            elif '!' in t:
                transformed.append("__exclamation__")
            else:
                transformed.append(t)
        return " ".join(transformed)

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Splits dialogue into sentences using punctuation boundaries.
        Handles abbreviations and keeps ending punctuation attached.
        """
        if not text:
            return []
        raw_parts = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in raw_parts if s.strip()]
        return sentences if sentences else [text.strip()]

    def tokenize(self, text: str, remove_stopwords: bool = False) -> List[str]:
        """
        Splits text into word tokens, with optional stopword filtering.
        """
        if not text:
            return []
        cleaned = self.clean_text(text, preserve_key_punctuation=False, lowercase=True)
        tokens = [w for w in cleaned.split() if w]
        if remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords]
        return tokens

    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Extracts comprehensive linguistic and structural statistics from text.
        """
        raw = text.strip() if text else ""
        if not raw:
            return {
                "character_count": 0,
                "character_count_no_spaces": 0,
                "word_count": 0,
                "unique_word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "question_marks": 0,
                "exclamation_marks": 0,
                "lexical_diversity": 0.0,
                "reading_time_seconds": 0.0
            }

        sentences = self.split_sentences(raw)
        words = self.tokenize(raw, remove_stopwords=False)
        unique_words = set(words)

        char_count = len(raw)
        char_count_no_spaces = len(re.sub(r'\s+', '', raw))
        word_count = len(words)
        unique_word_count = len(unique_words)
        sentence_count = len(sentences)

        avg_word_length = round(sum(len(w) for w in words) / word_count, 2) if word_count > 0 else 0.0
        lexical_diversity = round((unique_word_count / word_count) * 100, 2) if word_count > 0 else 0.0

        question_marks = raw.count('?')
        exclamation_marks = raw.count('!')

        # Average reading speed is ~200-250 words per minute (~3.5-4 words/sec)
        reading_time_seconds = round(word_count / 3.5, 2) if word_count > 0 else 0.0

        return {
            "character_count": char_count,
            "character_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "unique_word_count": unique_word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "question_marks": question_marks,
            "exclamation_marks": exclamation_marks,
            "lexical_diversity": lexical_diversity,
            "reading_time_seconds": reading_time_seconds
        }
