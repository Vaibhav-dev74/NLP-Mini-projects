"""
Unit tests for Database, Analytics Engine, and Batch Processor.
"""

import os
import pytest
from src.database import Database
from src.analytics import AnalyticsEngine
from src.batch_processor import BatchProcessor
from src.ml_classifier import DialogueMLClassifier


@pytest.fixture
def temp_db(tmp_path):
    db_file = os.path.join(tmp_path, "test_classifier.db")
    db = Database(db_path=db_file)
    return db


def test_database_crud(temp_db):
    """Test inserting, querying, and deleting classification records."""
    rec_id = temp_db.insert_classification(
        input_text="Good morning everyone!",
        predicted_category="GREETING",
        confidence_score=0.95,
        sentiment="POSITIVE",
        sentiment_polarity=0.6,
        emotion="HAPPY",
        classification_mode="rule_based_enhanced",
        explanation="Matched greeting pattern.",
        keywords=["morning"]
    )
    assert rec_id > 0

    record = temp_db.get_record_by_id(rec_id)
    assert record is not None
    assert record["predicted_category"] == "GREETING"
    assert record["sentiment"] == "POSITIVE"

    history, total = temp_db.get_history(limit=10, search="morning")
    assert total == 1
    assert len(history) == 1

    deleted = temp_db.delete_record(rec_id)
    assert deleted is True

    record_after = temp_db.get_record_by_id(rec_id)
    assert record_after is None


def test_analytics_metrics(temp_db):
    """Test analytics calculations with and without data."""
    analytics = AnalyticsEngine(db=temp_db)

    # Empty state
    empty_res = analytics.get_summary_metrics()
    assert empty_res["total_analyzed"] == 0
    assert empty_res["is_empty"] is True

    # Insert sample records
    temp_db.insert_classification(
        input_text="Hello",
        predicted_category="GREETING",
        confidence_score=0.9,
        sentiment="POSITIVE",
        sentiment_polarity=0.5,
        emotion="HAPPY",
        classification_mode="hybrid",
        explanation="test",
        keywords=[]
    )
    temp_db.insert_classification(
        input_text="Help me",
        predicted_category="REQUEST",
        confidence_score=0.85,
        sentiment="NEUTRAL",
        sentiment_polarity=0.0,
        emotion="NEUTRAL",
        classification_mode="hybrid",
        explanation="test",
        keywords=[]
    )

    metrics = analytics.get_summary_metrics()
    assert metrics["total_analyzed"] == 2
    assert metrics["is_empty"] is False
    assert "GREETING" in metrics["category_distribution"]
    assert "REQUEST" in metrics["category_distribution"]
