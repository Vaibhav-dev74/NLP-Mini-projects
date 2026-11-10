"""
Database Management Layer for Dialogue Classifier 2.0.
Uses SQLite with parameterized queries, schema creation, history querying,
filtering, searching, and deletion.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

from config import Config


class Database:
    """SQLite Database helper for storing and retrieving classification telemetry."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the SQLite database with Row factory."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database tables and indexes."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT NOT NULL,
                    predicted_category TEXT NOT NULL,
                    confidence_score REAL,
                    sentiment TEXT,
                    sentiment_polarity REAL,
                    emotion TEXT,
                    classification_mode TEXT NOT NULL,
                    explanation TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_created_at
                ON classifications (created_at DESC);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_category
                ON classifications (predicted_category);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_mode
                ON classifications (classification_mode);
            """)
            conn.commit()

    def insert_classification(
        self,
        input_text: str,
        predicted_category: str,
        confidence_score: Optional[float],
        sentiment: str,
        sentiment_polarity: Optional[float],
        emotion: str,
        classification_mode: str,
        explanation: str,
        keywords: List[str]
    ) -> int:
        """Inserts a new classification record using parameterized SQL."""
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords or "")
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO classifications (
                    input_text,
                    predicted_category,
                    confidence_score,
                    sentiment,
                    sentiment_polarity,
                    emotion,
                    classification_mode,
                    explanation,
                    keywords,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                input_text,
                predicted_category,
                confidence_score,
                sentiment,
                sentiment_polarity,
                emotion,
                classification_mode,
                explanation,
                keywords_str,
                now_utc
            ))
            conn.commit()
            return cursor.lastrowid

    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        mode: Optional[str] = None,
        search: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieves paginated classification history with optional filters and keyword search.
        Returns: (records, total_count)
        """
        conditions = []
        params: List[Any] = []

        if category and category.upper() != "ALL":
            conditions.append("predicted_category = ?")
            params.append(category.upper())

        if sentiment and sentiment.upper() != "ALL":
            conditions.append("sentiment = ?")
            params.append(sentiment.upper())

        if mode and mode.lower() != "all":
            conditions.append("classification_mode = ?")
            params.append(mode)

        if search and search.strip():
            conditions.append("(input_text LIKE ? OR explanation LIKE ? OR keywords LIKE ?)")
            term = f"%{search.strip()}%"
            params.extend([term, term, term])

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM classifications{where_clause};"

        select_sql = f"""
            SELECT * FROM classifications
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?;
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(count_sql, params)
            total_count = cursor.fetchone()[0]

            query_params = list(params) + [limit, offset]
            cursor.execute(select_sql, query_params)
            rows = cursor.fetchall()
            records = [dict(r) for r in rows]

        return records, total_count

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single record by its primary key ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classifications WHERE id = ?;", (record_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_record(self, record_id: int) -> bool:
        """Deletes a specific classification record by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM classifications WHERE id = ?;", (record_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> int:
        """Deletes all records from classifications table."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM classifications;")
            conn.commit()
            return cursor.rowcount
