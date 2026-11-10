"""
Configuration settings for Dialogue Classifier 2.0.
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dialogue-classifier-2.0-secure-key-2026")
    
    # Database
    INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
    DATABASE_PATH = os.path.join(INSTANCE_DIR, "classifier.db")
    
    # Models and Data
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DATASET_PATH = os.path.join(DATA_DIR, "dialogue_dataset.csv")
    UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploads")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    ALLOWED_EXTENSIONS = {"csv"}
    
    # Supported categories
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
