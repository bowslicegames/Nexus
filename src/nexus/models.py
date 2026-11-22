from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass(frozen=True)
class Observation:
    state: dict[str, Any]
    source: str = "environment"
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class Action:
    name: str
    effects: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0


@dataclass(frozen=True)
class Goal:
    target: dict[str, Any]
    priority: float = 1.0
    description: str = "reach target state"


@dataclass
class Prediction:
    action: str
    predicted_state: dict[str, Any]
    confidence: float
    model_version: int


@dataclass
class CycleResult:
    observation_id: str
    selected_action: str | None
    prediction: Prediction | None
    prediction_error: float | None
    rationale: str
