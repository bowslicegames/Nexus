from __future__ import annotations
import argparse
import json
import shlex
from .agent import NexusAgent
from .models import Action, Goal


def demo_transition(state: dict, action: Action) -> dict:
    result = dict(state)
    for key, value in action.effects.items():
        if isinstance(value, (int, float)) and isinstance(result.get(key, 0), (int, float)):
            result[key] = result.get(key, 0) + value
        else:
            result[key] = value
    return result


def parse_value(value: str):
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value


def parse_assignments(text: str) -> dict:
    result = {}
    for item in shlex.split(text):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = parse_value(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="NEXUS experimental cognitive architecture")
    parser.add_argument("--db", default="nexus_memory.sqlite3", help="SQLite episodic-memory file")
    args = parser.parse_args()
    agent = NexusAgent(args.db)
    actions: list[Action] = []
    goal = Goal({"energy": 10})
    print("NEXUS prototype. Commands: observe k=v..., goal k=v..., actions a,b, step, status, quit")
    try:
        while True:
            try:
                line = input("nexus> ").strip()
            except EOFError:
                break
            if not line:
                continue
            command, _, rest = line.partition(" ")
            if command in {"quit", "exit"}:
                break
            if command == "observe":
                state = parse_assignments(rest)
                agent.observe(state, "user")
                print(json.dumps(state))
            elif command == "goal":
                goal = Goal(parse_assignments(rest))
                print(f"Goal set: {goal.target}")
            elif command == "actions":
                names = [item.strip() for item in rest.split(",") if item.strip()]
                presets = {"recharge": {"energy": 2}, "wait": {"energy": 0},
                           "explore": {"energy": -1, "knowledge": 1}}
                actions = [Action(name, presets.get(name, {})) for name in names]
                print(f"Available actions: {[item.name for item in actions]}")
            elif command == "step":
                result = agent.step(actions, goal, demo_transition)
                print(json.dumps({
                    "action": result.selected_action,
                    "prediction": result.prediction.predicted_state if result.prediction else None,
                    "prediction_error": result.prediction_error,
                    "rationale": result.rationale
                }, indent=2))
            elif command == "status":
                print(json.dumps(agent.status(), indent=2, default=str))
            else:
                print("Unknown command.")
    finally:
        agent.close()


if __name__ == "__main__":
    main()
