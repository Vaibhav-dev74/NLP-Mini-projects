"""
Batch Text Processor for Dialogue Classifier 2.0.
Handles CSV upload validation, column inspection, bulk classification,
sentiment/emotion enrichment, and exportable CSV generation.
"""

import os
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from config import Config
from src.rule_classifier import EnhancedRuleClassifier, OriginalRuleClassifier
from src.ml_classifier import DialogueMLClassifier
from src.hybrid_classifier import HybridClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_detector import EmotionDetector
from src.database import Database


class BatchProcessor:
    """Processes CSV files for bulk dialogue classification and analytics enrichment."""

    def __init__(
        self,
        db: Optional[Database] = None,
        ml_classifier: Optional[DialogueMLClassifier] = None
    ):
        self.db = db or Database()
        self.ml_classifier = ml_classifier or DialogueMLClassifier()
        self.enhanced_rule = EnhancedRuleClassifier()
        self.original_rule = OriginalRuleClassifier()
        self.hybrid = HybridClassifier(
            rule_classifier=self.enhanced_rule,
            ml_classifier=self.ml_classifier
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_detector = EmotionDetector()

    def inspect_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Validates and previews a CSV file, detecting potential text columns.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("CSV file does not exist.")

        # Try reading with utf-8 or latin1 fallback
        try:
            df = pd.read_csv(file_path, nrows=5, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, nrows=5, encoding="latin1")

        columns = list(df.columns)
        if not columns:
            raise ValueError("The uploaded CSV file is empty or has no columns.")

        # Detect likely text column (text, dialogue, sentence, utterance, query, message, etc.)
        suggested_column = None
        candidates = ["text", "dialogue", "sentence", "utterance", "message", "query", "input", "content"]
        for c in columns:
            if c.lower() in candidates:
                suggested_column = c
                break

        if not suggested_column:
            # Fallback to first string/object column
            for c in columns:
                if df[c].dtype == object:
                    suggested_column = c
                    break
        if not suggested_column:
            suggested_column = columns[0]

        return {
            "columns": columns,
            "suggested_column": suggested_column,
            "preview_rows": df.head(5).to_dict(orient="records"),
            "file_size_bytes": os.path.getsize(file_path)
        }

    def process_file(
        self,
        input_csv_path: str,
        text_column: str,
        mode: str = "hybrid",
        save_to_db: bool = True,
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """
        Processes up to max_rows of a CSV file, adding classification and sentiment columns.
        """
        start_time = time.time()

        try:
            df = pd.read_csv(input_csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv_path, encoding="latin1")

        if text_column not in df.columns:
            raise ValueError(f"Selected column '{text_column}' not found in CSV. Available: {list(df.columns)}")

        # Limit row processing for performance
        if len(df) > max_rows:
            df = df.iloc[:max_rows].copy()

        # Select classification engine
        if mode == "rule_based_original":
            engine = self.original_rule
        elif mode == "rule_based_enhanced":
            engine = self.enhanced_rule
        elif mode == "machine_learning":
            engine = self.ml_classifier
        else:
            engine = self.hybrid

        categories = []
        confidences = []
        sentiments = []
        emotions = []
        explanations = []

        category_counts: Dict[str, int] = {}

        for _, row in df.iterrows():
            raw_text = str(row.get(text_column, "") or "")

            if mode in ["rule_based_original", "rule_based_enhanced", "hybrid"]:
                cls_res = engine.classify(raw_text)
            else:
                cls_res = engine.predict(raw_text)

            sent_res = self.sentiment_analyzer.analyze(raw_text)
            emo_res = self.emotion_detector.detect(raw_text)

            cat = cls_res.get("category", "STATEMENT")
            conf = cls_res.get("confidence") or cls_res.get("confidence_score")
            sent = sent_res.get("sentiment", "NEUTRAL")
            emo = emo_res.get("emotion", "NEUTRAL")
            expl = cls_res.get("explanation", "")

            categories.append(cat)
            confidences.append(conf if conf is not None else "")
            sentiments.append(sent)
            emotions.append(emo)
            explanations.append(expl)

            category_counts[cat] = category_counts.get(cat, 0) + 1

            if save_to_db and raw_text.strip():
                self.db.insert_classification(
                    input_text=raw_text[:500],
                    predicted_category=cat,
                    confidence_score=conf if isinstance(conf, (int, float)) else None,
                    sentiment=sent,
                    sentiment_polarity=sent_res.get("polarity_score"),
                    emotion=emo,
                    classification_mode=mode,
                    explanation=expl,
                    keywords=cls_res.get("matched_keywords", [])
                )

        # Append new columns to dataframe
        df["predicted_category"] = categories
        df["confidence"] = confidences
        df["sentiment"] = sentiments
        df["emotion"] = emotions
        df["explanation"] = explanations

        # Save processed CSV
        out_filename = f"processed_{uuid.uuid4().hex[:8]}.csv"
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        out_path = os.path.join(Config.UPLOAD_FOLDER, out_filename)
        df.to_csv(out_path, index=False, encoding="utf-8")

        duration_sec = round(time.time() - start_time, 2)

        return {
            "total_rows": len(df),
            "processed_file_name": out_filename,
            "output_file_path": out_path,
            "category_distribution": category_counts,
            "time_taken_seconds": duration_sec,
            "preview": df.head(10).to_dict(orient="records"),
            "columns": list(df.columns)
        }
