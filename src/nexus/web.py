from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


class NexusWebServer:
    """Small standard-library HTTP interface for the NEXUS research prototype."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 db_path: str = "nexus_memory.sqlite3") -> None:
        self.agent = NexusAgent(db_path)
        self.goal = Goal({"energy": 10})
        self.actions = [
            Action("recharge", {"energy": 2}),
            Action("wait", {"energy": 0}),
            Action("explore", {"energy": -1, "knowledge": 1}),
        ]
        self.lock = threading.RLock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
                body = payload if isinstance(payload, bytes) else (
                    json.dumps(payload, default=str).encode("utf-8")
                )
                self.send_response(status)
                self.send_header("Content-Type", content_type + ("; charset=utf-8" if "charset=" not in content_type else ""))
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > 65536:
                        raise ValueError("Request body is too large")
                    data = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("JSON body must be an object")
                    return data
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid request: {exc}") from exc

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/":
                    html = Path(__file__).with_name("index.html").read_bytes()
                    self._send(200, html, "text/html")
                    return
                if path == "/api/status":
                    with outer.lock:
                        self._send(200, {
                            **outer.agent.status(),
                            "goal": outer.goal.target,
                            "actions": [{"name": a.name, "effects": a.effects} for a in outer.actions],
                        })
                    return
                self._send(404, {"error": "Not found"})

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                try:
                    data = self._body()
                    with outer.lock:
                        if path == "/api/observe":
                            state = data.get("state")
                            if not isinstance(state, dict) or not state:
                                raise ValueError("state must be a non-empty JSON object")
                            if not all(isinstance(k, str) and isinstance(v, (int, float, str, bool)) for k, v in state.items()):
                                raise ValueError("state values must be numbers, strings, or booleans")
                            outer.agent.observe(state, "web_ui")
                            result = {"message": "Observation recorded", **outer.agent.status()}
                        elif path == "/api/goal":
                            target = data.get("target")
                            if not isinstance(target, dict) or not target:
                                raise ValueError("target must be a non-empty JSON object")
                            if not all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in target.items()):
                                raise ValueError("goal values must be numeric")
                            outer.goal = Goal(target, description="web interface goal")
                            result = {"message": "Goal updated", **outer.agent.status(), "goal": outer.goal.target}
                        elif path == "/api/actions":
                            items = data.get("actions")
                            if not isinstance(items, list) or not items:
                                raise ValueError("actions must be a non-empty list")
                            parsed = []
                            for item in items:
                                if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                                    raise ValueError("each action needs a name")
                                effects = item.get("effects", {})
                                if not isinstance(effects, dict) or not all(isinstance(v, (int, float, str, bool)) for v in effects.values()):
                                    raise ValueError("action effects must be a JSON object of simple values")
                                parsed.append(Action(item["name"].strip(), effects))
                            outer.actions = parsed
                            result = {"message": "Actions updated", **outer.agent.status(), "actions": [{"name": a.name, "effects": a.effects} for a in parsed]}
                        elif path == "/api/step":
                            cycle = outer.agent.step(outer.actions, outer.goal, demo_transition)
                            result = {
                                "message": "Cognitive cycle completed",
                                "cycle": {
                                    "observation_id": cycle.observation_id,
                                    "selected_action": cycle.selected_action,
                                    "prediction": cycle.prediction.predicted_state if cycle.prediction else None,
                                    "prediction_confidence": cycle.prediction.confidence if cycle.prediction else None,
                                    "prediction_error": cycle.prediction_error,
                                    "rationale": cycle.rationale,
                                },
                                **outer.agent.status(),
                            }
                        else:
                            self._send(404, {"error": "Not found"})
                            return
                    self._send(200, result)
                except ValueError as exc:
                    self._send(400, {"error": str(exc)})
                except Exception as exc:
                    self._send(500, {"error": f"Internal error: {type(exc).__name__}"})

            def log_message(self, format: str, *args: Any) -> None:
                print("[NEXUS web] " + format % args)

        self.httpd = ThreadingHTTPServer((host, port), Handler)

    def serve_forever(self) -> None:
        print(f"NEXUS interface running at http://{self.httpd.server_address[0]}:{self.httpd.server_address[1]}")
        try:
            self.httpd.serve_forever()
        finally:
            self.agent.close()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.agent.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the NEXUS browser interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default="nexus_memory.sqlite3")
    args = parser.parse_args()
    NexusWebServer(args.host, args.port, args.db).serve_forever()


if __name__ == "__main__":
    main()
