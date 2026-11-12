"""
Rule-Based Classification Engine for Dialogue Classifier 2.0.

Provides two classification engines:
1. OriginalRuleClassifier:
   - 100% faithful preservation of the original v1.0 logic from dialogue_classifier.py.
   - Preserves legacy punctuation-first priority and keyword sets.
2. EnhancedRuleClassifier:
   - Upgraded rule engine covering 10 distinct dialogue categories:
     GREETING, FAREWELL, QUESTION, REQUEST, EXCLAMATION,
     STATEMENT, COMPLAINT, SUGGESTION, APOLOGY, THANKS.
   - Implements robust regex word-boundary matching, multi-sentence intent analysis,
     intelligent conflict resolution, and matched pattern/keyword extraction.
"""

import re
from typing import Dict, List, Any, Optional, Tuple


# =====================================================================
# 1. ORIGINAL RULE CLASSIFIER (Legacy v1.0 Preservation)
# =====================================================================

def legacy_clean_text(text: str) -> str:
    """Original cleaning logic from dialogue_classifier.py (v1.0)."""
    text = text.lower().strip()
    for char in ('.', ',', ':', ';'):
        text = text.replace(char, '')
    return text


def legacy_classify_dialogue(dialogue_text: str) -> str:
    """Original classification function from dialogue_classifier.py (v1.0)."""
    if not dialogue_text or dialogue_text.isspace():
        return "EMPTY INPUT"

    cleaned = legacy_clean_text(dialogue_text)

    # 1. Punctuation Feature Check (Highest Priority in original)
    if dialogue_text.strip().endswith('?'):
        return "QUESTION"
    if dialogue_text.strip().endswith('!'):
        return "EXCLAMATION/EMPHASIS"

    # 2. Enhanced Keyword/Structure Check
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "greetings"]
    farewell_keywords = ["bye", "goodbye", "see you", "cya", "farewell", "later", "take care"]
    request_phrases = ["can you", "could you", "would you", "please", "i need", "i would like to"]
    implicit_question_keywords = ["who", "what", "where", "when", "why", "how", "tell me about"]

    for word in greeting_keywords:
        if f" {word} " in f" {cleaned} ":
            return "GREETING"

    for word in farewell_keywords:
        if f" {word} " in f" {cleaned} ":
            return "FAREWELL"

    for word in request_phrases:
        if cleaned.startswith(word):
            return "REQUEST"

    for word in implicit_question_keywords:
        if f" {word} " in f" {cleaned} ":
            return "QUESTION"

    return "STATEMENT"


class OriginalRuleClassifier:
    """
    Wrapper for the original v1.0 rule-based classification algorithm.
    Produces structured metadata while retaining 100% fidelity to the original logic.
    """

    def __init__(self):
        self.name = "Original Rule-Based Classifier (v1.0)"

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify text using original rules with explanation and metadata."""
        if not text or text.isspace():
            return {
                "category": "EMPTY INPUT",
                "confidence": None,
                "confidence_score": 0.0,
                "explanation": "No text provided to classify.",
                "matched_keywords": [],
                "matched_patterns": [],
                "mode": "rule_based_original",
                "is_rule_based": True,
                "probabilities": None
            }

        raw = text.strip()
        cleaned = legacy_clean_text(raw)
        matched_keywords: List[str] = []
        matched_patterns: List[str] = []
        explanation = ""

        # Check ending punctuation
        if raw.endswith('?'):
            category = "QUESTION"
            matched_patterns.append("ending_question_mark '?'")
            explanation = "Matched terminal question mark ('?'). Original priority: punctuation over keywords."
        elif raw.endswith('!'):
            category = "EXCLAMATION/EMPHASIS"
            matched_patterns.append("ending_exclamation_mark '!'")
            explanation = "Matched terminal exclamation mark ('!'). Original priority: punctuation over keywords."
        else:
            greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "greetings"]
            farewell_keywords = ["bye", "goodbye", "see you", "cya", "farewell", "later", "take care"]
            request_phrases = ["can you", "could you", "would you", "please", "i need", "i would like to"]
            implicit_question_keywords = ["who", "what", "where", "when", "why", "how", "tell me about"]

            category = "STATEMENT"
            explanation = "No rule matched; defaulted to STATEMENT as per v1.0 logic."

            # Check greetings
            for word in greeting_keywords:
                if f" {word} " in f" {cleaned} ":
                    category = "GREETING"
                    matched_keywords.append(word)
                    explanation = f"Matched greeting keyword '{word}'."
                    break

            # Check farewells
            if category == "STATEMENT":
                for word in farewell_keywords:
                    if f" {word} " in f" {cleaned} ":
                        category = "FAREWELL"
                        matched_keywords.append(word)
                        explanation = f"Matched farewell keyword '{word}'."
                        break

            # Check requests
            if category == "STATEMENT":
                for word in request_phrases:
                    if cleaned.startswith(word):
                        category = "REQUEST"
                        matched_keywords.append(word)
                        matched_patterns.append(f"starts_with('{word}')")
                        explanation = f"Input starts with request phrase '{word}'."
                        break

            # Check implicit questions
            if category == "STATEMENT":
                for word in implicit_question_keywords:
                    if f" {word} " in f" {cleaned} ":
                        category = "QUESTION"
                        matched_keywords.append(word)
                        explanation = f"Matched implicit question keyword '{word}'."
                        break

        return {
            "category": category,
            "confidence": None,
            "confidence_score": 1.0 if category != "STATEMENT" else 0.5,
            "explanation": explanation,
            "matched_keywords": matched_keywords,
            "matched_patterns": matched_patterns,
            "mode": "rule_based_original",
            "is_rule_based": True,
            "probabilities": None
        }


# =====================================================================
# 2. ENHANCED RULE CLASSIFIER (v2.0 10-Class Engine)
# =====================================================================

class EnhancedRuleClassifier:
    """
    Upgraded rule-based classifier for 10 dialogue categories.
    Features:
    - Regex word boundaries (\\b) avoiding substring false positives.
    - Multi-sentence dialogue parsing.
    - Priority-based intent conflict resolution.
    - Pattern matching (polite requests, apologies, thanks, complaints, suggestions).
    """

    CATEGORIES = [
        "GREETING",
        "FAREWELL",
        "QUESTION",
        "REQUEST",
        "EXCLAMATION",
        "STATEMENT",
        "COMPLAINT",
        "SUGGESTION",
        "APOLOGY",
        "THANKS"
    ]

    def __init__(self):
        self.name = "Enhanced Rule-Based Classifier (v2.0)"
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for high-performance and strict boundary matching."""
        # 1. GREETING
        self.greeting_patterns = [
            (re.compile(r'\b(hello|hey|hi|howdy|hiya|greetings|hola)\b', re.IGNORECASE), "greeting_salutation"),
            (re.compile(r'\bgood\s+(morning|afternoon|evening|day)\b', re.IGNORECASE), "time_greeting"),
            (re.compile(r'\b(what\'?s\s+up|sup|how\s+are\s+you|how\'?s\s+it\s+going|how\s+do\s+you\s+do)\b', re.IGNORECASE), "inquiry_greeting")
        ]

        # 2. FAREWELL
        self.farewell_patterns = [
            (re.compile(r'\b(goodbye|bye|bye-bye|cya|farewell|adieu)\b', re.IGNORECASE), "farewell_salutation"),
            (re.compile(r'\b(see\s+you(\s+later|\s+soon|\s+around)?|catch\s+you\s+later|talk\s+to\s+you\s+later)\b', re.IGNORECASE), "see_you_phrase"),
            (re.compile(r'\b(take\s+care|have\s+a\s+(good|great|nice)\s+(day|night|weekend|evening))\b', re.IGNORECASE), "well_wishes_farewell"),
            (re.compile(r'\b(good\s*night)\b', re.IGNORECASE), "goodnight_phrase")
        ]

        # 3. THANKS
        self.thanks_patterns = [
            (re.compile(r'\b(thank\s+you|thanks|thx|ty|much\s+obliged|grateful)\b', re.IGNORECASE), "thanks_expression"),
            (re.compile(r'\b(i\s+appreciate\s+(it|your\s+help|your\s+time|this))\b', re.IGNORECASE), "appreciation_phrase"),
            (re.compile(r'\bmany\s+thanks\b', re.IGNORECASE), "many_thanks_phrase")
        ]

        # 4. APOLOGY
        self.apology_patterns = [
            (re.compile(r'\b(sorry|my\s+bad|apologies|pardon\s+me|excuse\s+me|forgive\s+me)\b', re.IGNORECASE), "apology_keyword"),
            (re.compile(r'\b(i\s+apologize|i\'?m\s+sorry|i\s+am\s+sorry|so\s+sorry|deeply\s+sorry)\b', re.IGNORECASE), "formal_apology")
        ]

        # 5. REQUEST (Takes priority over general questions when polite directives/asks are present)
        self.request_patterns = [
            (re.compile(r'^\s*(please|kindly)\b', re.IGNORECASE), "starts_with_polite_marker"),
            (re.compile(r'\b(please|kindly)\s+(help|send|provide|give|tell|assist|do|share|call|check|update)\b', re.IGNORECASE), "polite_directive"),
            (re.compile(r'^\s*(can|could|would|will)\s+you\s+(please\s+)?([a-z]+)', re.IGNORECASE), "modal_request_prefix"),
            (re.compile(r'\b(i\s+need\s+you\s+to|i\s+would\s+like\s+you\s+to|i\'?d\s+like\s+to\s+request|could\s+you\s+possibly)\b', re.IGNORECASE), "request_phrase")
        ]

        # 6. SUGGESTION
        self.suggestion_patterns = [
            (re.compile(r'\b(you\s+should|you\s+could|you\s+might\s+want\s+to|we\s+should|we\s+could)\b', re.IGNORECASE), "modal_suggestion"),
            (re.compile(r'\b(i\s+suggest|i\s+recommend|my\s+suggestion\s+is|my\s+advice\s+is)\b', re.IGNORECASE), "direct_suggestion"),
            (re.compile(r'\b(how\s+about|what\s+if|why\s+don\'?t\s+(we|you))\b', re.IGNORECASE), "inquiry_suggestion")
        ]

        # 7. COMPLAINT
        self.complaint_patterns = [
            (re.compile(r'\b(terrible|terribly|horrible|horribly|awful|awfully|worst|disaster|unacceptable|poor|poorly|useless|broken|defective|unresponsive|unreliable|glitchy|buggy)\b', re.IGNORECASE), "strong_negative_adjective"),
            (re.compile(r'\b(not\s+working|doesn\'?t\s+work|does\s+not\s+work|failed\s+to|crashed|keeps\s+crashing|stopped\s+working|keeps\s+freezing|freezes|too\s+slow|horribly\s+slow)\b', re.IGNORECASE), "malfunction_phrase"),
            (re.compile(r'\b(i\s+am\s+(very\s+)?(disappointed|upset|frustrated|angry|dissatisfied))\b', re.IGNORECASE), "discontent_expression"),
            (re.compile(r'\b(bad\s+service|waste\s+of\s+money|waste\s+of\s+time|ridiculous\s+delay|still\s+haven\'?t\s+received)\b', re.IGNORECASE), "complaint_phrase")
        ]

        # 8. QUESTION
        self.question_wh_pattern = re.compile(
            r'\b(who|what|where|when|why|how|which|whose|whom)\b', re.IGNORECASE
        )
        self.question_aux_start = re.compile(
            r'^\s*(is|are|am|was|were|do|does|did|can|could|should|would|will|shall|may|might|has|have|had)\s+[a-z]+',
            re.IGNORECASE
        )

        # 9. EXCLAMATION
        self.exclamation_interjections = re.compile(
            r'\b(wow|omg|hurray|hooray|yay|ouch|oops|whoa|bravo|alas|yikes|awesome|incredible|unbelievable)\b',
            re.IGNORECASE
        )
        self.exclamative_phrase = re.compile(
            r'\b(what\s+a(n)?\s+[a-z]+|how\s+[a-z]+)\b', re.IGNORECASE
        )

    def split_sentences(self, text: str) -> List[str]:
        """Split input into distinct sentences while preserving punctuation."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def classify_single_sentence(self, sentence: str) -> Tuple[str, float, str, List[str], List[str]]:
        """
        Classifies a single sentence with priority-based rules.
        Returns: (category, confidence_score, explanation, matched_keywords, matched_patterns)
        """
        raw = sentence.strip()
        has_question_mark = '?' in raw
        has_exclamation_mark = '!' in raw

        matched_keywords: List[str] = []
        matched_patterns: List[str] = []

        # --- Priority 1: THANKS & APOLOGY ---
        for pat, tag in self.thanks_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "THANKS", 0.98, f"Detected expression of gratitude: '{m.group(0)}'.", matched_keywords, matched_patterns

        for pat, tag in self.apology_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "APOLOGY", 0.97, f"Detected apology/regret expression: '{m.group(0)}'.", matched_keywords, matched_patterns

        # --- Priority 2: GREETING & FAREWELL (Even if ending in !) ---
        for pat, tag in self.greeting_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "GREETING", 0.96, f"Matched greeting expression: '{m.group(0)}'.", matched_keywords, matched_patterns

        for pat, tag in self.farewell_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "FAREWELL", 0.96, f"Matched farewell expression: '{m.group(0)}'.", matched_keywords, matched_patterns

        # --- Priority 3: REQUEST ---
        for pat, tag in self.request_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "REQUEST", 0.95, f"Matched polite request/action directive pattern: '{m.group(0)}'.", matched_keywords, matched_patterns

        # --- Priority 4: SUGGESTION ---
        for pat, tag in self.suggestion_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "SUGGESTION", 0.92, f"Matched suggestion/recommendation phrase: '{m.group(0)}'.", matched_keywords, matched_patterns

        # --- Priority 5: COMPLAINT ---
        for pat, tag in self.complaint_patterns:
            m = pat.search(raw)
            if m:
                matched_keywords.append(m.group(0))
                matched_patterns.append(tag)
                return "COMPLAINT", 0.92, f"Matched customer dissatisfaction or defect phrase: '{m.group(0)}'.", matched_keywords, matched_patterns

        # --- Priority 6: EXCLAMATION (When containing explicit ! and no ?) ---
        m_interj = self.exclamation_interjections.search(raw)
        m_exclam_phrase = self.exclamative_phrase.search(raw)

        if has_exclamation_mark and not has_question_mark:
            matched_patterns.append("terminal_exclamation_mark")
            if m_interj:
                matched_keywords.append(m_interj.group(0))
                matched_patterns.append("interjection")
                return "EXCLAMATION", 0.96, f"Contains high-intensity emotional interjection: '{m_interj.group(0)}'.", matched_keywords, matched_patterns
            if m_exclam_phrase:
                matched_keywords.append(m_exclam_phrase.group(0))
                matched_patterns.append("exclamative_phrase")
                return "EXCLAMATION", 0.94, f"Contains exclamative phrase construct: '{m_exclam_phrase.group(0)}'.", matched_keywords, matched_patterns
            return "EXCLAMATION", 0.88, "Ends with exclamation mark ('!') denoting emphasis.", matched_keywords, matched_patterns

        if m_interj and not has_question_mark:
            matched_keywords.append(m_interj.group(0))
            matched_patterns.append("interjection")
            return "EXCLAMATION", 0.92, f"Contains emotional interjection: '{m_interj.group(0)}'.", matched_keywords, matched_patterns

        # --- Priority 7: QUESTION ---
        if has_question_mark:
            matched_patterns.append("terminal_question_mark")
            wh_match = self.question_wh_pattern.search(raw)
            if wh_match:
                matched_keywords.append(wh_match.group(0))
                return "QUESTION", 0.98, f"Contains '?' and interrogative word '{wh_match.group(0)}'.", matched_keywords, matched_patterns
            return "QUESTION", 0.95, "Contains interrogative terminal punctuation '?'.", matched_keywords, matched_patterns

        if self.question_aux_start.search(raw) or (self.question_wh_pattern.search(raw) and len(raw.split()) <= 12):
            m = self.question_wh_pattern.search(raw) or self.question_aux_start.search(raw)
            kw = m.group(0) if m else "syntactic inversion"
            matched_keywords.append(kw)
            matched_patterns.append("inquiry_syntax")
            return "QUESTION", 0.88, f"Inverted auxiliary or wh-word syntax indicates question: '{kw}'.", matched_keywords, matched_patterns

        # --- Priority 8: STATEMENT (Default) ---
        return "STATEMENT", 0.70, "Declarative or informative sentence with no dominant speech act flags.", [], ["default_declarative"]

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies multi-sentence or single-sentence dialogue.
        Resolves multi-sentence dialogues by determining the primary speech act intent.
        """
        if not text or text.isspace():
            return {
                "category": "EMPTY INPUT",
                "confidence": None,
                "confidence_score": 0.0,
                "explanation": "No text provided to classify.",
                "matched_keywords": [],
                "matched_patterns": [],
                "mode": "rule_based_enhanced",
                "is_rule_based": True,
                "probabilities": None,
                "sentence_breakdown": []
            }

        sentences = self.split_sentences(text)
        results = [self.classify_single_sentence(s) for s in sentences]

        sentence_breakdown = []
        for s, (cat, conf, expl, kws, pats) in zip(sentences, results):
            sentence_breakdown.append({
                "sentence": s,
                "category": cat,
                "confidence": conf,
                "explanation": expl,
                "keywords": kws
            })

        # Single sentence case
        if len(sentences) == 1:
            cat, conf, expl, kws, pats = results[0]
            return {
                "category": cat,
                "confidence": None,
                "confidence_score": conf,
                "explanation": expl,
                "matched_keywords": list(set(kws)),
                "matched_patterns": pats,
                "mode": "rule_based_enhanced",
                "is_rule_based": True,
                "probabilities": None,
                "sentence_breakdown": sentence_breakdown
            }

        # Multi-sentence resolution:
        # Determine primary intent with importance ranking:
        # REQUEST > COMPLAINT > SUGGESTION > QUESTION > APOLOGY > THANKS > GREETING > FAREWELL > EXCLAMATION > STATEMENT
        priority_order = [
            "REQUEST", "COMPLAINT", "SUGGESTION", "QUESTION",
            "APOLOGY", "THANKS", "GREETING", "FAREWELL",
            "EXCLAMATION", "STATEMENT"
        ]

        # Gather all detected categories
        detected_categories = [r[0] for r in results]

        # Pick highest priority detected category
        chosen_category = "STATEMENT"
        chosen_idx = 0
        for p in priority_order:
            if p in detected_categories:
                chosen_category = p
                chosen_idx = detected_categories.index(p)
                break

        chosen_result = results[chosen_idx]
        all_kws = []
        all_pats = []
        for r in results:
            all_kws.extend(r[3])
            all_pats.extend(r[4])

        summary_expl = (
            f"Multi-sentence dialogue. Primary communicative intent identified as {chosen_category} "
            f"(from sentence: \"{sentences[chosen_idx]}\"). {chosen_result[2]}"
        )

        return {
            "category": chosen_category,
            "confidence": None,
            "confidence_score": chosen_result[1],
            "explanation": summary_expl,
            "matched_keywords": list(set(all_kws)),
            "matched_patterns": list(set(all_pats)),
            "mode": "rule_based_enhanced",
            "is_rule_based": True,
            "probabilities": None,
            "sentence_breakdown": sentence_breakdown
        }
