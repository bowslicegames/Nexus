from __future__ import annotations
from typing import Callable
from .memory import EpisodicMemory, SemanticMemory
from .models import Action, CycleResult, Goal, Observation
from .reasoning import ActionSelector
from .world_model import WorldModel


class NexusAgent:
    """Coordinates observe -> predict -> choose -> experience -> learn."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.episodic = EpisodicMemory(db_path)
        self.semantic = SemanticMemory()
        self.world = WorldModel()
        self.selector = ActionSelector(self.world)
        self.current_state: dict = {}
        self.last_observation: Observation | None = None
        self.cycle_count = 0
        self.last_prediction_error: float | None = None

    def observe(self, state: dict, source: str = "environment") -> Observation:
        observation = Observation(dict(state), source=source)
        self.episodic.record_observation(observation)
        self.current_state = dict(state)
        self.last_observation = observation
        return observation

    def step(self, actions: list[Action], goal: Goal,
             environment_step: Callable[[dict, Action], dict]) -> CycleResult:
        if self.last_observation is None:
            self.observe(self.current_state)
        assert self.last_observation is not None
        before = dict(self.current_state)
        action, rationale = self.selector.choose(before, actions, goal)
        if action is None:
            return CycleResult(self.last_observation.id, None, None, None, rationale)
        prediction = self.world.predict(before, action.name)
        after = dict(environment_step(before, action))
        error = self.world.prediction_error(prediction.predicted_state, after)
        self.episodic.record_outcome(
            self.last_observation.id, action.name, after, prediction.predicted_state, error
        )
        self.world.learn(before, action.name, after)
        self.current_state = after
        self.cycle_count += 1
        self.last_prediction_error = error
        next_observation = self.observe(after, source="action_outcome")
        for key, value in after.items():
            self.semantic.learn(f"state.{key}", value, 1.0, next_observation.id)
        return CycleResult(next_observation.id, action.name, prediction, error, rationale)

    def status(self) -> dict:
        return {
            "cycles": self.cycle_count,
            "state": dict(self.current_state),
            "world_model_version": self.world.version,
            "learned_action_effects": self.world.learned_effects(),
            "last_prediction_error": self.last_prediction_error,
            "beliefs": self.semantic.all(),
        }

    def close(self) -> None:
        self.episodic.close()
