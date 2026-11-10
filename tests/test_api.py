"""
API Integration Tests for Dialogue Classifier 2.0.
Verifies all REST endpoints: health, classify, history, analytics, export, and batch.
"""

import io
import json
import pytest
from app import app, db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health(client):
    """Test health endpoint returns 200 and valid JSON."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "supported_categories" in data


def test_api_classify_empty(client):
    """Test empty text payload returns 400 error."""
    res = client.post("/api/classify", json={"text": ""})
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_classify_modes(client):
    """Test classification across various modes."""
    # 1. Hybrid Mode
    res_hybrid = client.post("/api/classify", json={
        "text": "Hello there! Could you please share the access credentials?",
        "mode": "hybrid"
    })
    assert res_hybrid.status_code == 200
    data_h = res_hybrid.get_json()
    assert data_h["category"] in ["REQUEST", "GREETING"]
    assert "sentiment" in data_h
    assert "emotion" in data_h
    assert "statistics" in data_h

    # 2. Enhanced Rule Mode
    res_rule = client.post("/api/classify", json={
        "text": "This software is broken and completely unusable.",
        "mode": "rule_based_enhanced"
    })
    assert res_rule.status_code == 200
    data_r = res_rule.get_json()
    assert data_r["category"] == "COMPLAINT"
    assert data_r["is_rule_based"] is True

    # 3. Original Legacy Mode
    res_orig = client.post("/api/classify", json={
        "text": "Where is the documentation?",
        "mode": "rule_based_original"
    })
    assert res_orig.status_code == 200
    data_o = res_orig.get_json()
    assert data_o["category"] == "QUESTION"


def test_api_history_and_analytics(client):
    """Test retrieving history, deleting a record, and getting analytics."""
    # Create a record first
    res = client.post("/api/classify", json={
        "text": "Thank you for the fantastic presentation!",
        "mode": "hybrid"
    })
    rec_id = res.get_json().get("record_id")
    assert rec_id is not None

    # Get history
    hist_res = client.get("/api/history?page=1&limit=10")
    assert hist_res.status_code == 200
    hist_data = hist_res.get_json()
    assert hist_data["total"] > 0

    # Get analytics
    analytics_res = client.get("/api/analytics")
    assert analytics_res.status_code == 200
    a_data = analytics_res.get_json()
    assert a_data["total_analyzed"] > 0

    # Delete record
    del_res = client.delete(f"/api/history/{rec_id}")
    assert del_res.status_code == 200


def test_api_exports(client):
    """Test CSV and Report export endpoints."""
    csv_res = client.get("/api/export/csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.content_type

    report_res = client.get("/api/export/report")
    assert report_res.status_code == 200
    assert "text/plain" in report_res.content_type


def test_api_batch_inspect_and_process(client):
    """Test batch CSV file inspection and classification."""
    csv_content = "sentence\nHello world!\nCould you help me?\nThis is great news!\n"
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "test_batch.csv")
    }

    inspect_res = client.post("/api/batch/inspect", data=data, content_type="multipart/form-data")
    assert inspect_res.status_code == 200
    i_data = inspect_res.get_json()
    assert "temp_filename" in i_data
    assert "sentence" in i_data["columns"]

    # Process batch
    batch_res = client.post("/api/classify/batch", data={
        "temp_filename": i_data["temp_filename"],
        "text_column": "sentence",
        "mode": "hybrid"
    })
    assert batch_res.status_code == 200
    b_data = batch_res.get_json()
    assert b_data["total_rows"] == 3
    assert "processed_file_name" in b_data
