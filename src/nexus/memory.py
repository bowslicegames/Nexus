from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from .models import Observation


class EpisodicMemory:
    """Append-only experience history stored in SQLite."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, timestamp REAL NOT NULL, source TEXT NOT NULL,
                state_json TEXT NOT NULL, action TEXT, outcome_json TEXT,
                prediction_json TEXT, prediction_error REAL
            )
        """)
        self.db.commit()

    def record_observation(self, observation: Observation) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO episodes(id,timestamp,source,state_json) VALUES(?,?,?,?)",
            (observation.id, observation.timestamp, observation.source,
             json.dumps(observation.state, sort_keys=True))
        )
        self.db.commit()

    def record_outcome(self, observation_id: str, action: str,
                       outcome: dict[str, Any], prediction: dict[str, Any] | None,
                       prediction_error: float | None) -> None:
        self.db.execute(
            "UPDATE episodes SET action=?, outcome_json=?, prediction_json=?, prediction_error=? WHERE id=?",
            (action, json.dumps(outcome, sort_keys=True),
             json.dumps(prediction, sort_keys=True) if prediction is not None else None,
             prediction_error, observation_id)
        )
        self.db.commit()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.db.close()


class SemanticMemory:
    """Explicit beliefs with confidence and evidence provenance."""

    def __init__(self) -> None:
        self._beliefs: dict[str, dict[str, Any]] = {}

    def learn(self, key: str, value: Any, confidence: float, evidence_id: str) -> None:
        confidence = max(0.0, min(1.0, confidence))
        old = self._beliefs.get(key)
        evidence = list(old["evidence"]) if old else []
        if evidence_id not in evidence:
            evidence.append(evidence_id)
        if old is None or confidence >= old["confidence"]:
            self._beliefs[key] = {
                "value": value, "confidence": confidence,
                "evidence": evidence, "updated_at": time.time()
            }
        else:
            old["evidence"] = evidence

    def get(self, key: str) -> dict[str, Any] | None:
        belief = self._beliefs.get(key)
        return dict(belief) if belief else None

    def all(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._beliefs.items()}
