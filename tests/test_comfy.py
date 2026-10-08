import pytest

from tests.fake_comfy import FakeComfy
from triggercollide.comfy import ComfyClient, ComfyError, normalize_base_url, scan_comfy
from triggercollide.report import build_result


def test_scan_uses_get_only_and_finds_clash():
    with FakeComfy() as fake:
        records, stats = scan_comfy(fake.url)
    assert {m for m, _ in fake.requests} == {"GET"}
    assert stats["methods"] == ["GET"]
    assert stats["files"] == 3 and stats["errors"] == 0 and stats["no_trigger_info"] == 1
    payload = build_result(records, stats)
    assert payload["summary"]["severity"]["high"] == 1
    assert any("/view_metadata/loras?filename=styles%2Fglow_fox_xl.safetensors" in p for _, p in fake.requests)


def test_falls_back_to_object_info():
    with FakeComfy(list_endpoint=False) as fake:
        records, _ = scan_comfy(fake.url)
    assert len(records) == 3
    assert ("GET", "/object_info/LoraLoader") in fake.requests


def test_client_refuses_queue_endpoints():
    client = ComfyClient("http://127.0.0.1:9")
    for path in ("/prompt", "/queue", "/interrupt", "/free", "/upload/image"):
        with pytest.raises(ComfyError):
            client._get(path)
    assert client.requests_made == []


def test_source_never_posts():
    import inspect

    import triggercollide.comfy as comfy

    source = inspect.getsource(comfy)
    assert 'method="GET"' in source
    assert 'method="POST"' not in source and "data=" not in source


def test_unreachable_host_gives_friendly_error():
    with pytest.raises(ComfyError, match="Could not reach ComfyUI"):
        scan_comfy("http://127.0.0.1:9", timeout=1)


def test_url_normalization():
    assert normalize_base_url("192.168.1.20:8188") == "http://192.168.1.20:8188"
    assert normalize_base_url("http://host:8188/") == "http://host:8188"
    with pytest.raises(ComfyError):
        normalize_base_url("")
    with pytest.raises(ComfyError):
        normalize_base_url("http://host:8188/prompt")
