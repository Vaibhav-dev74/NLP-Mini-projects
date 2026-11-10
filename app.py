"""
Flask Application and REST API Gateway for Dialogue Classifier 2.0.
"""

import os
import io
import csv
from datetime import datetime, timezone
from typing import Dict, Any

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory,
    abort
)
from werkzeug.utils import secure_filename

from config import Config
from src.preprocessing import TextPreprocessor
from src.rule_classifier import EnhancedRuleClassifier, OriginalRuleClassifier
from src.ml_classifier import DialogueMLClassifier
from src.hybrid_classifier import HybridClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.emotion_detector import EmotionDetector
from src.database import Database
from src.analytics import AnalyticsEngine
from src.batch_processor import BatchProcessor

app = Flask(__name__)
app.config.from_object(Config)

# Ensure required directories exist
os.makedirs(Config.INSTANCE_DIR, exist_ok=True)
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)

# Initialize singletons
db = Database(Config.DATABASE_PATH)
preprocessor = TextPreprocessor()
original_rule_clf = OriginalRuleClassifier()
enhanced_rule_clf = EnhancedRuleClassifier()
ml_clf = DialogueMLClassifier(Config.MODELS_DIR)

# Auto-train ML model if not already trained and dataset is present
if not ml_clf.is_trained and os.path.exists(Config.DATASET_PATH):
    try:
        ml_clf.train(Config.DATASET_PATH)
    except Exception as e:
        app.logger.warning(f"Initial ML training skipped: {e}")

hybrid_clf = HybridClassifier(
    rule_classifier=enhanced_rule_clf,
    ml_classifier=ml_clf
)
sentiment_analyzer = SentimentAnalyzer()
emotion_detector = EmotionDetector()
analytics_engine = AnalyticsEngine(db)
batch_processor = BatchProcessor(db=db, ml_classifier=ml_clf)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


# =====================================================================
# HTML TEMPLATE ROUTES
# =====================================================================

@app.route("/")
def dashboard_view():
    metrics = analytics_engine.get_summary_metrics()
    recent_history, _ = db.get_history(limit=5)
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        metrics=metrics,
        recent_history=recent_history,
        ml_trained=ml_clf.is_trained
    )


@app.route("/classifier")
def classifier_view():
    return render_template(
        "classifier.html",
        active_page="classifier",
        categories=Config.CATEGORIES,
        ml_trained=ml_clf.is_trained
    )


@app.route("/batch")
def batch_view():
    return render_template(
        "batch.html",
        active_page="batch",
        ml_trained=ml_clf.is_trained
    )


@app.route("/history")
def history_view():
    return render_template(
        "history.html",
        active_page="history",
        categories=Config.CATEGORIES
    )


@app.route("/analytics")
def analytics_view():
    metrics = analytics_engine.get_summary_metrics()
    return render_template(
        "analytics.html",
        active_page="analytics",
        metrics=metrics
    )


@app.route("/about")
def about_view():
    ml_metadata = ml_clf.metadata if ml_clf.is_trained else {}
    return render_template(
        "about.html",
        active_page="about",
        ml_metadata=ml_metadata,
        ml_trained=ml_clf.is_trained
    )


# =====================================================================
# REST API ENDPOINTS
# =====================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check and model telemetry endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Dialogue Classifier 2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected",
        "models": {
            "ml_trained": ml_clf.is_trained,
            "selected_model": ml_clf.metadata.get("selected_model", "None"),
            "accuracy": ml_clf.metadata.get("selected_accuracy"),
            "f1_score": ml_clf.metadata.get("selected_f1_score")
        },
        "supported_categories": Config.CATEGORIES
    })


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """
    Main text classification endpoint.
    Accepts text, classification mode, and optional save flag.
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "hybrid").lower()
    save_history = data.get("save_history", True)

    if not text:
        return jsonify({
            "error": "Input text cannot be empty.",
            "category": "EMPTY INPUT"
        }), 400

    if len(text) > 10000:
        return jsonify({
            "error": "Input text exceeds maximum length of 10,000 characters."
        }), 400

    # 1. Linguistic Statistics
    stats = preprocessor.get_text_statistics(text)

    # 2. Dialogue Intent Classification
    if mode == "rule_based_original":
        classification_result = original_rule_clf.classify(text)
    elif mode == "rule_based_enhanced":
        classification_result = enhanced_rule_clf.classify(text)
    elif mode == "machine_learning":
        classification_result = ml_clf.predict(text)
    else:  # hybrid
        classification_result = hybrid_clf.classify(text)

    # 3. Sentiment Analysis
    sentiment_result = sentiment_analyzer.analyze(text)

    # 4. Emotion Detection
    emotion_result = emotion_detector.detect(text)

    # 5. Persist to SQLite
    record_id = None
    if save_history and classification_result["category"] != "EMPTY INPUT":
        conf_val = classification_result.get("confidence")
        if conf_val is None:
            conf_val = classification_result.get("confidence_score")

        record_id = db.insert_classification(
            input_text=text,
            predicted_category=classification_result["category"],
            confidence_score=conf_val if isinstance(conf_val, (int, float)) else None,
            sentiment=sentiment_result["sentiment"],
            sentiment_polarity=sentiment_result.get("polarity_score"),
            emotion=emotion_result["emotion"],
            classification_mode=mode,
            explanation=classification_result.get("explanation", ""),
            keywords=classification_result.get("matched_keywords", [])
        )

    response = {
        "record_id": record_id,
        "input_text": text,
        "category": classification_result["category"],
        "confidence": classification_result.get("confidence"),
        "confidence_score": classification_result.get("confidence_score"),
        "probabilities": classification_result.get("probabilities"),
        "explanation": classification_result.get("explanation"),
        "matched_keywords": classification_result.get("matched_keywords", []),
        "matched_patterns": classification_result.get("matched_patterns", []),
        "is_rule_based": classification_result.get("is_rule_based", False),
        "mode": mode,
        "subsystem_used": classification_result.get("subsystem_used"),
        "sentiment": sentiment_result,
        "emotion": emotion_result,
        "statistics": stats,
        "sentence_breakdown": classification_result.get("sentence_breakdown", [])
    }

    return jsonify(response)


@app.route("/api/batch/inspect", methods=["POST"])
def api_batch_inspect():
    """Inspects an uploaded CSV file and lists column candidates."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are supported."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(Config.UPLOAD_FOLDER, f"temp_{filename}")
    file.save(save_path)

    try:
        inspection = batch_processor.inspect_csv(save_path)
        inspection["temp_filename"] = f"temp_{filename}"
        return jsonify(inspection)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": str(e)}), 400


@app.route("/api/classify/batch", methods=["POST"])
def api_classify_batch():
    """Processes an inspected CSV file."""
    data = request.form
    temp_filename = data.get("temp_filename")
    text_column = data.get("text_column")
    mode = data.get("mode", "hybrid")
    save_to_db = data.get("save_to_db", "true").lower() == "true"

    if not temp_filename:
        # Check direct file upload
        if "file" in request.files:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                temp_filename = f"batch_{secure_filename(file.filename)}"
                save_path = os.path.join(Config.UPLOAD_FOLDER, temp_filename)
                file.save(save_path)
            else:
                return jsonify({"error": "Invalid or missing file."}), 400
        else:
            return jsonify({"error": "Missing temp_filename parameter."}), 400

    file_path = os.path.join(Config.UPLOAD_FOLDER, temp_filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Uploaded file not found on server."}), 404

    try:
        # If text_column not provided, inspect first
        if not text_column:
            inspection = batch_processor.inspect_csv(file_path)
            text_column = inspection["suggested_column"]

        result = batch_processor.process_file(
            input_csv_path=file_path,
            text_column=text_column,
            mode=mode,
            save_to_db=save_to_db
        )

        result["download_url"] = f"/download/{result['processed_file_name']}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>", methods=["GET"])
def download_processed_file(filename):
    """Safely serves processed batch CSV files."""
    safe_name = secure_filename(filename)
    return send_from_directory(Config.UPLOAD_FOLDER, safe_name, as_attachment=True)


@app.route("/api/history", methods=["GET"])
def api_history():
    """Paginated classification history."""
    try:
        page = max(1, int(request.args.get("page", 1)))
        limit = min(100, max(1, int(request.args.get("limit", 20))))
    except ValueError:
        page, limit = 1, 20

    offset = (page - 1) * limit
    category = request.args.get("category")
    sentiment = request.args.get("sentiment")
    mode = request.args.get("mode")
    search = request.args.get("search")

    records, total = db.get_history(
        limit=limit,
        offset=offset,
        category=category,
        sentiment=sentiment,
        mode=mode,
        search=search
    )

    total_pages = (total + limit - 1) // limit if total > 0 else 1

    return jsonify({
        "records": records,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages
    })


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_history(record_id):
    """Deletes a single history record."""
    success = db.delete_record(record_id)
    if success:
        return jsonify({"message": f"Record {record_id} deleted successfully."})
    return jsonify({"error": "Record not found."}), 404


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears all history records."""
    deleted_count = db.clear_history()
    return jsonify({"message": f"Cleared {deleted_count} records."})


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Returns analytics telemetry directly from stored SQLite records."""
    metrics = analytics_engine.get_summary_metrics()
    return jsonify(metrics)


@app.route("/api/train", methods=["POST"])
def api_train_model():
    """Retrains the machine learning models on dialogue dataset."""
    try:
        dataset_path = Config.DATASET_PATH
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Training dataset not found at {dataset_path}"}), 400

        metrics = ml_clf.train(dataset_path)
        return jsonify({
            "message": "Model retrained successfully.",
            "metrics": metrics
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """Exports all stored classification history records as a downloadable CSV."""
    records, _ = db.get_history(limit=10000, offset=0)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Input Text", "Predicted Category", "Confidence Score",
        "Sentiment", "Sentiment Polarity", "Emotion",
        "Classification Mode", "Explanation", "Keywords", "Created At"
    ])

    for r in records:
        writer.writerow([
            r["id"],
            r["input_text"],
            r["predicted_category"],
            r["confidence_score"] or "",
            r["sentiment"] or "",
            r["sentiment_polarity"] or "",
            r["emotion"] or "",
            r["classification_mode"],
            r["explanation"],
            r["keywords"],
            r["created_at"]
        ])

    output.seek(0)
    bytes_data = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(
        bytes_data,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"classification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )


@app.route("/api/export/report", methods=["GET"])
def api_export_report():
    """Exports a text summary report of system telemetry and analytics."""
    metrics = analytics_engine.get_summary_metrics()
    records, total = db.get_history(limit=15)

    report_lines = [
        "=" * 60,
        "   DIALOGUE CLASSIFIER 2.0 - TELEMETRY & AUDIT REPORT",
        "=" * 60,
        f"Generated At (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Analyzed Texts: {metrics['total_analyzed']}",
        f"Top Dialogue Category: {metrics['top_category']} ({metrics['top_category_count']} occurrences)",
        f"Average Confidence: {metrics['avg_confidence']}",
        "",
        "--- CATEGORY DISTRIBUTION ---"
    ]

    for cat, count in metrics["category_distribution"].items():
        pct = round((count / metrics["total_analyzed"]) * 100, 1) if metrics["total_analyzed"] else 0
        report_lines.append(f"  {cat.ljust(15)} : {count} ({pct}%)")

    report_lines.extend([
        "",
        "--- SENTIMENT BREAKDOWN ---",
        f"  POSITIVE : {metrics['sentiment_distribution'].get('POSITIVE', 0)}",
        f"  NEGATIVE : {metrics['sentiment_distribution'].get('NEGATIVE', 0)}",
        f"  NEUTRAL  : {metrics['sentiment_distribution'].get('NEUTRAL', 0)}",
        "",
        "--- CLASSIFICATION MODE USAGE ---"
    ])

    for mode_name, count in metrics["mode_distribution"].items():
        report_lines.append(f"  {mode_name.ljust(22)} : {count}")

    report_lines.extend([
        "",
        "--- RECENT SAMPLES (LAST 15) ---"
    ])

    for r in records:
        text_preview = (r['input_text'][:40] + '...') if len(r['input_text']) > 40 else r['input_text']
        report_lines.append(
            f"  [#{r['id']}] [{r['predicted_category']}] [{r['sentiment']}] \"{text_preview}\""
        )

    report_lines.append("\n" + "=" * 60)
    report_content = "\n".join(report_lines)

    bytes_data = io.BytesIO(report_content.encode("utf-8"))
    return send_file(
        bytes_data,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"dialogue_classifier_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
