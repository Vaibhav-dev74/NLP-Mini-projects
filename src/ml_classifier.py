"""
Machine Learning Classification Engine for Dialogue Classifier 2.0.

Implements TF-IDF vectorization, dual-model training (Logistic Regression vs.
Multinomial Naive Bayes), model comparison, evaluation metric tracking,
Joblib serialization, and real probability-based predictions.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

from src.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class DialogueMLClassifier:
    """
    Scikit-Learn ML classifier for Dialogue Category detection.
    Trains Logistic Regression and Multinomial Naive Bayes, selects the champion model,
    saves artifacts with joblib, and yields real probability distributions.
    """

    SUPPORTED_CATEGORIES = [
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

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "classifier_model.pkl")
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

        self.preprocessor = TextPreprocessor()
        self.model: Optional[Any] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.metadata: Dict[str, Any] = {}
        self.is_trained: bool = False

        # Attempt to auto-load if models already exist on disk
        self.load_model()

    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Loads and strictly validates the training dataset.
        Verifies presence of 'text' and 'category' columns and valid categories.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training dataset file not found at: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate required columns
        for col in ["text", "category"]:
            if col not in df.columns:
                raise ValueError(f"Dataset missing mandatory column: '{col}'. Present: {list(df.columns)}")

        # Drop null or whitespace-only rows
        df = df.dropna(subset=["text", "category"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip().str.upper()
        df = df[df["text"] != ""]

        # Validate categories
        invalid_categories = set(df["category"]) - set(self.SUPPORTED_CATEGORIES)
        if invalid_categories:
            raise ValueError(
                f"Dataset contains unsupported category labels: {invalid_categories}. "
                f"Supported: {self.SUPPORTED_CATEGORIES}"
            )

        if len(df) < 20:
            raise ValueError(f"Dataset too small ({len(df)} samples). Minimum 20 required for training/testing.")

        return df

    def train(
        self,
        csv_path: str = "data/dialogue_dataset.csv",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Executes complete training pipeline:
        1. Preprocessing and feature engineering
        2. Train/Test split without data leakage
        3. TF-IDF fitting on training split only
        4. Logistic Regression training and evaluation
        5. Multinomial Naive Bayes training and evaluation
        6. Best model selection based on weighted F1 score
        7. Model & metadata persistence using joblib
        """
        os.makedirs(self.models_dir, exist_ok=True)
        df = self.load_dataset(csv_path)

        # Preprocess text
        cleaned_texts = [self.preprocessor.clean_for_ml(t) for t in df["text"]]
        # Convert to standard Python lists / numpy object array to ensure compatibility with pandas 3.0+ arrow backends
        labels = list(df["category"].tolist())

        # Split with stratification to preserve class distributions
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            cleaned_texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        # Fit TF-IDF only on training data (avoids data leakage)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            strip_accents="unicode"
        )
        X_train_vec = vectorizer.fit_transform(X_train_raw)
        X_test_vec = vectorizer.transform(X_test_raw)

        # 1. Model A: Logistic Regression
        lr_model = LogisticRegression(
            C=2.0,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        )
        lr_model.fit(X_train_vec, y_train)
        lr_preds = lr_model.predict(X_test_vec)

        lr_acc = float(accuracy_score(y_test, lr_preds))
        lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_preds, average="weighted", zero_division=0)
        lr_report = classification_report(y_test, lr_preds, output_dict=True, zero_division=0)

        # 2. Model B: Multinomial Naive Bayes
        nb_model = MultinomialNB(alpha=0.3)
        nb_model.fit(X_train_vec, y_train)
        nb_preds = nb_model.predict(X_test_vec)

        nb_acc = float(accuracy_score(y_test, nb_preds))
        nb_p, nb_r, nb_f1, _ = precision_recall_fscore_support(y_test, nb_preds, average="weighted", zero_division=0)
        nb_report = classification_report(y_test, nb_preds, output_dict=True, zero_division=0)

        # Model comparison and selection
        if lr_f1 >= nb_f1:
            best_model_name = "Logistic Regression"
            best_model = lr_model
            best_acc = lr_acc
            best_f1 = float(lr_f1)
        else:
            best_model_name = "Multinomial Naive Bayes"
            best_model = nb_model
            best_acc = nb_acc
            best_f1 = float(nb_f1)

        # Save artifacts
        joblib.dump(best_model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)

        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "dataset_file": csv_path,
            "total_samples": len(df),
            "training_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "selected_model": best_model_name,
            "selected_accuracy": round(best_acc, 4),
            "selected_f1_score": round(best_f1, 4),
            "classes": sorted(list(set(labels))),
            "models_comparison": {
                "Logistic Regression": {
                    "accuracy": round(lr_acc, 4),
                    "precision_weighted": round(float(lr_p), 4),
                    "recall_weighted": round(float(lr_r), 4),
                    "f1_weighted": round(float(lr_f1), 4),
                    "classification_report": lr_report
                },
                "Multinomial Naive Bayes": {
                    "accuracy": round(nb_acc, 4),
                    "precision_weighted": round(float(nb_p), 4),
                    "recall_weighted": round(float(nb_r), 4),
                    "f1_weighted": round(float(nb_f1), 4),
                    "classification_report": nb_report
                }
            },
            "experimental_notice": (
                "Trained on baseline curated dialogue dataset. Demonstrates functional ML pipeline, "
                "TF-IDF vectorization, dual-model comparison, and true probabilistic scoring."
            )
        }

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Update instance state
        self.model = best_model
        self.vectorizer = vectorizer
        self.metadata = metadata
        self.is_trained = True

        return metadata

    def load_model(self) -> bool:
        """Loads serialized model and vectorizer from disk if present."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                self.is_trained = True
                return True
            except Exception as e:
                logger.warning(f"Could not load serialized models: {e}")
                self.is_trained = False
                return False
        self.is_trained = False
        return False

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies input text using the trained ML model.
        Returns true probabilities (never synthetic or hardcoded values).
        """
        if not text or text.isspace():
            return {
                "category": "EMPTY INPUT",
                "confidence": None,
                "confidence_score": 0.0,
                "probabilities": None,
                "explanation": "No text provided to classify.",
                "matched_keywords": [],
                "matched_patterns": [],
                "mode": "machine_learning",
                "is_rule_based": False,
                "model_name": self.metadata.get("selected_model", "Untrained")
            }

        if not self.is_trained or self.model is None or self.vectorizer is None:
            return {
                "category": "UNTRAINED_MODEL",
                "confidence": None,
                "confidence_score": 0.0,
                "probabilities": None,
                "explanation": "Machine learning model has not been trained yet. Please train the model from dataset.",
                "matched_keywords": [],
                "matched_patterns": [],
                "mode": "machine_learning",
                "is_rule_based": False,
                "model_name": "None"
            }

        # Preprocess input
        ml_input = self.preprocessor.clean_for_ml(text)
        vec_input = self.vectorizer.transform([ml_input])

        # Get true probabilities
        probs = self.model.predict_proba(vec_input)[0]
        classes = self.model.classes_

        prob_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probs)}
        # Sort classes descending by probability
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        top_category = list(sorted_probs.keys())[0]
        top_confidence = sorted_probs[top_category]

        second_category = list(sorted_probs.keys())[1] if len(sorted_probs) > 1 else None
        second_confidence = sorted_probs[second_category] if second_category else 0.0

        model_name = self.metadata.get("selected_model", "Scikit-Learn Classifier")

        explanation = (
            f"Classified by {model_name} with {round(top_confidence * 100, 1)}% confidence. "
        )
        if second_category:
            explanation += f"Second most probable category: {second_category} ({round(second_confidence * 100, 1)}%)."

        # Extract top unigram/bigram features from input that influenced the prediction
        tokens = self.preprocessor.tokenize(text, remove_stopwords=False)
        matched_tokens = [t for t in tokens if t in self.vectorizer.vocabulary_]

        return {
            "category": top_category,
            "confidence": top_confidence,
            "confidence_score": top_confidence,
            "probabilities": sorted_probs,
            "explanation": explanation,
            "matched_keywords": matched_tokens[:5],
            "matched_patterns": [f"tfidf_vectorizer_top_class_{top_category}"],
            "mode": "machine_learning",
            "is_rule_based": False,
            "model_name": model_name
        }
