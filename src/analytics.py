"""
Analytics and Telemetry Aggregator for Dialogue Classifier 2.0.
Computes real distribution statistics, daily trends, mode usage,
and sentiment/emotion metrics directly from SQLite records.
"""

from typing import Dict, Any, List, Optional
from src.database import Database


class AnalyticsEngine:
    """Computes real metrics, distributions, and historical trends."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Computes aggregated metrics from actual stored classification records.
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Total Count
            cursor.execute("SELECT COUNT(*) FROM classifications;")
            total_records = cursor.fetchone()[0]

            if total_records == 0:
                return {
                    "total_analyzed": 0,
                    "top_category": "N/A",
                    "top_category_count": 0,
                    "avg_confidence": 0.0,
                    "category_distribution": {},
                    "sentiment_distribution": {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0},
                    "emotion_distribution": {},
                    "mode_distribution": {},
                    "daily_timeline": [],
                    "is_empty": True
                }

            # 2. Category Distribution
            cursor.execute("""
                SELECT predicted_category, COUNT(*) as cnt
                FROM classifications
                GROUP BY predicted_category
                ORDER BY cnt DESC;
            """)
            cat_rows = cursor.fetchall()
            category_distribution = {r["predicted_category"]: r["cnt"] for r in cat_rows}
            top_category = cat_rows[0]["predicted_category"] if cat_rows else "N/A"
            top_category_count = cat_rows[0]["cnt"] if cat_rows else 0

            # 3. Sentiment Distribution
            cursor.execute("""
                SELECT sentiment, COUNT(*) as cnt
                FROM classifications
                GROUP BY sentiment;
            """)
            sentiment_rows = cursor.fetchall()
            sentiment_distribution = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
            for r in sentiment_rows:
                s_key = (r["sentiment"] or "NEUTRAL").upper()
                sentiment_distribution[s_key] = r["cnt"]

            # 4. Emotion Distribution
            cursor.execute("""
                SELECT emotion, COUNT(*) as cnt
                FROM classifications
                WHERE emotion IS NOT NULL AND emotion != ''
                GROUP BY emotion
                ORDER BY cnt DESC;
            """)
            emotion_rows = cursor.fetchall()
            emotion_distribution = {r["emotion"]: r["cnt"] for r in emotion_rows}

            # 5. Classification Mode Usage
            cursor.execute("""
                SELECT classification_mode, COUNT(*) as cnt
                FROM classifications
                GROUP BY classification_mode
                ORDER BY cnt DESC;
            """)
            mode_rows = cursor.fetchall()
            mode_distribution = {r["classification_mode"]: r["cnt"] for r in mode_rows}

            # 6. Daily Activity Timeline (Last 14 days)
            cursor.execute("""
                SELECT SUBSTR(created_at, 1, 10) as day_date, COUNT(*) as cnt
                FROM classifications
                GROUP BY day_date
                ORDER BY day_date ASC
                LIMIT 14;
            """)
            timeline_rows = cursor.fetchall()
            daily_timeline = [{"date": r["day_date"], "count": r["cnt"]} for r in timeline_rows]

            # 7. Average Confidence (where available)
            cursor.execute("""
                SELECT AVG(confidence_score)
                FROM classifications
                WHERE confidence_score IS NOT NULL;
            """)
            avg_conf_row = cursor.fetchone()[0]
            avg_confidence = round(float(avg_conf_row), 3) if avg_conf_row is not None else 0.0

        return {
            "total_analyzed": total_records,
            "top_category": top_category,
            "top_category_count": top_category_count,
            "avg_confidence": avg_confidence,
            "category_distribution": category_distribution,
            "sentiment_distribution": sentiment_distribution,
            "emotion_distribution": emotion_distribution,
            "mode_distribution": mode_distribution,
            "daily_timeline": daily_timeline,
            "is_empty": False
        }
