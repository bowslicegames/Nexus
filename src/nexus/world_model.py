from __future__ import annotations
from collections import defaultdict
from typing import Any
from .models import Prediction


class WorldModel:
    """Inspectable baseline model of numeric action-conditioned state changes.

    This learns correlations in observed transitions, not general causal laws.
    """

    def __init__(self) -> None:
        self._deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.version = 0

    def predict(self, state: dict[str, Any], action: str) -> Prediction:
        predicted = dict(state)
        dimensions = self._deltas.get(action, {})
        observed = 0
        samples = 0
        for key, values in dimensions.items():
            if key in state and isinstance(state[key], (int, float)) and values:
                predicted[key] = state[key] + sum(values) / len(values)
                observed += 1
                samples += len(values)
        confidence = min(0.95, samples / (samples + 3.0)) if observed else 0.0
        return Prediction(action, predicted, confidence, self.version)

    def learn(self, before: dict[str, Any], action: str, after: dict[str, Any]) -> None:
        for key, old in before.items():
            new = after.get(key)
            if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                self._deltas[action][key].append(float(new - old))
        self.version += 1

    def prediction_error(self, predicted: dict[str, Any], actual: dict[str, Any]) -> float:
        errors = []
        for key, value in predicted.items():
            actual_value = actual.get(key)
            if isinstance(value, (int, float)) and isinstance(actual_value, (int, float)):
                errors.append(abs(float(value) - float(actual_value)))
        return sum(errors) / len(errors) if errors else 0.0

    def learned_effects(self) -> dict[str, dict[str, float]]:
        return {
            action: {key: sum(values) / len(values) for key, values in dims.items() if values}
            for action, dims in self._deltas.items()
        }
