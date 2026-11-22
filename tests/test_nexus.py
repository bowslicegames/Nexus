from nexus.agent import NexusAgent
from nexus.models import Action, Goal


def transition(state, action):
    result = dict(state)
    for key, delta in action.effects.items():
        result[key] = result.get(key, 0) + delta
    return result


def test_agent_observes_learns_and_persists(tmp_path):
    db = tmp_path / "episodes.sqlite3"
    agent = NexusAgent(str(db))
    agent.observe({"energy": 1})
    actions = [Action("recharge", {"energy": 2}), Action("drain", {"energy": -1})]
    result = agent.step(actions, Goal({"energy": 5}), transition)
    assert result.selected_action == "recharge"
    assert agent.current_state["energy"] == 3
    assert agent.world.version == 1
    assert agent.last_prediction_error == 0
    assert agent.episodic.recent()
    agent.close()


def test_world_model_learns_action_effect():
    agent = NexusAgent()
    agent.observe({"x": 2})
    agent.step([Action("move", {"x": 3})], Goal({"x": 10}), transition)
    prediction = agent.world.predict({"x": 5}, "move")
    assert prediction.predicted_state["x"] == 8
    agent.close()


def test_empty_actions_are_handled():
    agent = NexusAgent()
    agent.observe({"energy": 1})
    result = agent.step([], Goal({"energy": 5}), transition)
    assert result.selected_action is None
    assert "No actions" in result.rationale
    agent.close()
