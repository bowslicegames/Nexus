from __future__ import annotations
from .models import Action, Goal
from .world_model import WorldModel


def distance_to_goal(state: dict, goal: Goal) -> float:
    errors = []
    for key, target in goal.target.items():
        current = state.get(key)
        if isinstance(target, (int, float)) and isinstance(current, (int, float)):
            errors.append(abs(float(target) - float(current)))
        else:
            errors.append(0.0 if current == target else 1.0)
    return sum(errors) / len(errors) if errors else 0.0


class ActionSelector:
    def __init__(self, model: WorldModel, exploration_bonus: float = 0.05) -> None:
        self.model = model
        self.exploration_bonus = exploration_bonus

    def choose(self, state: dict, actions: list[Action], goal: Goal) -> tuple[Action | None, str]:
        if not actions:
            return None, "No actions were supplied by the environment."
        ranked = []
        current_distance = distance_to_goal(state, goal)
        for action in actions:
            prediction = self.model.predict(state, action.name)
            improvement = current_distance - distance_to_goal(prediction.predicted_state, goal)
            exploration = self.exploration_bonus * (1.0 - prediction.confidence)
            score = goal.priority * improvement + exploration - max(0.0, action.cost) * 0.01
            ranked.append((score, action, prediction, improvement, exploration))
        ranked.sort(key=lambda item: item[0], reverse=True)
        _, action, prediction, improvement, exploration = ranked[0]
        rationale = (
            f"Selected {action.name!r}: estimated goal-distance improvement "
            f"{improvement:.3f}, model confidence {prediction.confidence:.2f}, "
            f"exploration bonus {exploration:.3f}."
        )
        return action, rationale
