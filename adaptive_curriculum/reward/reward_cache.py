"""
SQLite-backed reward cache to avoid re-scoring identical (checkpoint, image, question) triples.
"""
import hashlib
import sqlite3
import json
import time
from pathlib import Path
from typing import Optional


class RewardCache:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rewards (
                cache_key TEXT PRIMARY KEY,
                model_checkpoint_id TEXT,
                item_id TEXT,
                bucket TEXT,
                image_path TEXT,
                question TEXT,
                expected TEXT,
                predicted TEXT,
                correct INTEGER,
                weight REAL,
                score REAL,
                timestamp REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_item ON rewards(item_id, bucket)")
        conn.commit()

    @staticmethod
    def make_key(model_checkpoint_id: str, image_hash: str, item_id: str, question: str) -> str:
        raw = f"{model_checkpoint_id}|{image_hash}|{item_id}|{question}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def hash_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return hashlib.sha256(image_path.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT predicted, correct, weight, score FROM rewards WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        return {"predicted": row[0], "correct": bool(row[1]), "weight": row[2], "score": row[3]}

    def put(
        self,
        cache_key: str,
        model_checkpoint_id: str,
        item_id: str,
        bucket: str,
        image_path: str,
        question: str,
        expected: str,
        predicted: str,
        correct: bool,
        weight: float,
        score: float,
    ):
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO rewards
            (cache_key, model_checkpoint_id, item_id, bucket, image_path,
             question, expected, predicted, correct, weight, score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key, model_checkpoint_id, item_id, bucket, image_path,
                question, expected, predicted, int(correct), weight, score,
                time.time(),
            ),
        )
        conn.commit()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
