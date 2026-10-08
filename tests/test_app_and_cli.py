import json
import threading
import urllib.request

from tests.fake_comfy import FakeComfy
from triggercollide import HOST
from triggercollide.app import create_app
from triggercollide.cli import main
from triggercollide.server import make_bound_server


def test_server_on_loopback_returns_200(tmp_path):
    assert HOST == "127.0.0.1"
    server = make_bound_server(create_app(data_dir=tmp_path), 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        assert host == "127.0.0.1"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            assert response.status == 200
            assert "TriggerCollide" in response.read().decode()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/favicon.ico", timeout=5) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_api_demo_scan_workflow_and_exports(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    scan = client.post("/api/scan/demo", json={}).get_json()
    assert scan["summary"]["loras"] == 11
    assert scan["summary"]["severity"]["high"] >= 1
    wf = client.get("/api/demo/workflow").get_json()
    report = client.post("/api/workflow/check", json={"scan_id": scan["scan_id"], "workflow": wf}).get_json()
    assert report["issue_counts"]["high"] >= 1
    csv = client.get(f"/api/scan/{scan['scan_id']}/collisions.csv")
    assert csv.status_code == 200 and csv.data.decode().startswith("severity,score")
    assert client.get(f"/api/scan/{scan['scan_id']}.json").status_code == 200
    bad = client.post("/api/workflow/check", json={"scan_id": scan["scan_id"], "workflow": "{not json"})
    assert bad.status_code == 400
    assert client.post("/api/scan/folder", json={"path": str(tmp_path / "missing")}).status_code == 400


def test_api_comfy_scan(tmp_path):
    client = create_app(data_dir=tmp_path).test_client()
    with FakeComfy() as fake:
        scan = client.post("/api/scan/comfy", json={"url": fake.url}).get_json()
    assert scan["summary"]["loras"] == 3
    assert all(m == "GET" for m, _ in fake.requests)
    err = client.post("/api/scan/comfy", json={"url": "http://127.0.0.1:9", "timeout": 1})
    assert err.status_code == 400 and "Could not reach" in err.get_json()["error"]


def test_cli_scan_and_check(tmp_path, demo_dir, capsys):
    out = tmp_path / "scan.json"
    csv = tmp_path / "c.csv"
    assert main(["scan", "--folder", str(demo_dir), "--out", str(out), "--csv", str(csv)]) == 0
    assert json.loads(out.read_text())["summary"]["loras"] == 11
    assert csv.read_text().startswith("severity")
    capsys.readouterr()
    assert main(["scan", "--folder", str(demo_dir), "--counts-only"]) == 0
    printed = capsys.readouterr().out
    counts = json.loads(printed)
    assert counts["loras"] == 11 and "Watercolor" not in printed
    from triggercollide.demo import demo_workflow

    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(demo_workflow()))
    assert main(["check", "--folder", str(demo_dir), "--workflow", str(wf)]) == 1
    assert "HIGH" in capsys.readouterr().out
    assert main(["scan", "--comfy", "http://127.0.0.1:9", "--timeout", "1"]) == 2
