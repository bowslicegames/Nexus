from urllib.request import urlopen, Request
import json
import threading

from nexus.web import NexusWebServer


def test_web_api_serves_ui_and_runs_cycle(tmp_path):
    server = NexusWebServer("127.0.0.1", 0, str(tmp_path / "web.sqlite3"))
    thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.httpd.server_address[1]}"
    try:
        with urlopen(base + "/") as response:
            assert response.status == 200
            assert b"NEXUS" in response.read()
        request = Request(base + "/api/observe",
                          data=json.dumps({"state": {"energy": 1, "knowledge": 0}}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request) as response:
            assert json.load(response)["state"]["energy"] == 1
        request = Request(base + "/api/step", data=b"{}",
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request) as response:
            result = json.load(response)
            assert result["cycles"] == 1
            assert result["cycle"]["selected_action"] in {"recharge", "wait", "explore"}
    finally:
        server.httpd.shutdown()
        server.httpd.server_close()
        server.agent.close()
        thread.join(timeout=2)
