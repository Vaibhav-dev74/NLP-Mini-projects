"""
Unit tests for SQLite Database operations in Dialogue Classifier 2.0.
Verifies:
- Schema initialization and indexing
- Record insertion with parameterized SQL
- Historical queries and filtering (by category, sentiment, mode, search term)
- Record retrieval by ID
- Record deletion
- Clearing complete history
"""

import os
import pytest
from src.database import Database


@pytest.fixture
def test_db(tmp_path):
    db_file = os.path.join(tmp_path, "test_database_ops.db")
    return Database(db_path=db_file)


def test_database_initialization_and_schema(test_db):
    """Test tables and indexes exist upon initialization."""
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='classifications';")
        table = cursor.fetchone()
        assert table is not None

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_classifications_created_at';")
        idx = cursor.fetchone()
        assert idx is not None


def test_database_crud_operations(test_db):
    """Test insert, read, filter, delete operations."""
    # 1. Insert multiple records
    id1 = test_db.insert_classification(
        input_text="Good morning everyone!",
        predicted_category="GREETING",
        confidence_score=0.96,
        sentiment="POSITIVE",
        sentiment_polarity=0.7,
        emotion="HAPPY",
        classification_mode="hybrid",
        explanation="Greeting matched.",
        keywords=["morning", "everyone"]
    )
    assert id1 > 0

    id2 = test_db.insert_classification(
        input_text="This app is terribly slow and unresponsive.",
        predicted_category="COMPLAINT",
        confidence_score=0.92,
        sentiment="NEGATIVE",
        sentiment_polarity=-0.8,
        emotion="ANGRY",
        classification_mode="rule_based_enhanced",
        explanation="Complaint keywords detected.",
        keywords=["slow", "terribly"]
    )
    assert id2 > id1

    # 2. Get record by ID
    rec = test_db.get_record_by_id(id1)
    assert rec is not None
    assert rec["predicted_category"] == "GREETING"
    assert rec["sentiment"] == "POSITIVE"

    # 3. Filter by category
    complaints, count_c = test_db.get_history(category="COMPLAINT")
    assert count_c == 1
    assert complaints[0]["predicted_category"] == "COMPLAINT"

    # 4. Filter by sentiment
    positives, count_p = test_db.get_history(sentiment="POSITIVE")
    assert count_p == 1
    assert positives[0]["id"] == id1

    # 5. Search keyword
    searched, count_s = test_db.get_history(search="unresponsive")
    assert count_s == 1
    assert searched[0]["id"] == id2

    # 6. Delete single record
    deleted = test_db.delete_record(id1)
    assert deleted is True
    assert test_db.get_record_by_id(id1) is None

    # 7. Clear all remaining records
    cleared_count = test_db.clear_history()
    assert cleared_count == 1
    all_recs, total = test_db.get_history()
    assert total == 0
    assert len(all_recs) == 0
