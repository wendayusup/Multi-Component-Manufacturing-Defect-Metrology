"""
SQLite Storage Layer for Inspection History and Quality Audit Records.
Persists minimal inspection metadata, AI predictions, and operator review notes.
Does not persist raw image binary data unless specifically configured.
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional
import config


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database and ensures tables exist."""
    db_path = Path(config.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database schema."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                module TEXT NOT NULL,
                component_type TEXT NOT NULL,
                status TEXT NOT NULL,
                predicted_label TEXT,
                confidence_score REAL,
                scores_json TEXT,
                model_version TEXT,
                quality_status TEXT,
                quality_details_json TEXT,
                processing_ms REAL,
                warnings_json TEXT,
                operator_review TEXT,
                operator_note TEXT
            )
        """)
        conn.commit()


def save_inspection(data: Dict):
    """Inserts a new inspection record."""
    init_db()
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO inspections (
                id, created_at, module, component_type, status,
                predicted_label, confidence_score, scores_json,
                model_version, quality_status, quality_details_json,
                processing_ms, warnings_json, operator_review, operator_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("inspection_id"),
            data.get("created_at"),
            data.get("module", "casting_defect"),
            data.get("component_type", "casting"),
            data.get("status"),
            data.get("predicted_label"),
            data.get("confidence_score"),
            json.dumps(data.get("scores", {})),
            data.get("model_version"),
            data.get("quality_status"),
            json.dumps(data.get("quality_details", {})),
            data.get("processing_ms"),
            json.dumps(data.get("warnings", [])),
            data.get("operator_review"),
            data.get("operator_note")
        ))
        conn.commit()


def update_manual_review(inspection_id: str, operator_review: str, operator_note: str) -> bool:
    """Updates manual verification feedback from quality inspector."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.execute("""
            UPDATE inspections
            SET operator_review = ?, operator_note = ?
            WHERE id = ?
        """, (operator_review, operator_note, inspection_id))
        conn.commit()
        return cursor.rowcount > 0


def get_recent_inspections(limit: int = 15) -> List[Dict]:
    """Retrieves list of recent inspection records."""
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM inspections
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        results = []
        for r in rows:
            results.append({
                "inspection_id": r["id"],
                "created_at": r["created_at"],
                "module": r["module"],
                "status": r["status"],
                "predicted_label": r["predicted_label"],
                "confidence_score": r["confidence_score"],
                "scores": json.loads(r["scores_json"]) if r["scores_json"] else {},
                "model_version": r["model_version"],
                "quality_status": r["quality_status"],
                "processing_ms": r["processing_ms"],
                "operator_review": r["operator_review"],
                "operator_note": r["operator_note"]
            })
        return results

