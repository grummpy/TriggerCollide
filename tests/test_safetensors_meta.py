import json

import pytest

from triggercollide.safetensors_meta import HeaderError, parse_header_bytes, read_header, write_safetensors


def test_round_trip_reads_metadata_and_tensor_names(tmp_path):
    path = write_safetensors(tmp_path / "a.safetensors", {"ss_output_name": "Alpha"}, {"w.lora_down": [1.0, 2.0]})
    header = read_header(path)
    assert header.metadata == {"ss_output_name": "Alpha"}
    assert header.tensor_names == ["w.lora_down"]


def test_reader_never_reads_tensor_bytes(tmp_path, monkeypatch):
    path = write_safetensors(tmp_path / "b.safetensors", {"k": "v"}, {"t": [0.0] * 50000})
    reads = []
    real_open = type(path).open

    def tracking_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        original = handle.read

        def read(n=-1):
            reads.append(n)
            return original(n)

        handle.read = read
        return handle

    monkeypatch.setattr(type(path), "open", tracking_open)
    read_header(path)
    assert reads[0] == 8
    assert all(n != -1 for n in reads)
    assert sum(reads) < 2000


def test_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"\xff" * 64)
    with pytest.raises(HeaderError):
        read_header(bad)
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01")
    with pytest.raises(HeaderError):
        read_header(short)


def test_header_length_past_end_is_rejected(tmp_path):
    p = tmp_path / "trunc.safetensors"
    p.write_bytes((5000).to_bytes(8, "little") + b"{}")
    with pytest.raises(HeaderError):
        read_header(p)


def test_non_string_metadata_values_are_kept_as_json():
    raw = json.dumps({"__metadata__": {"n": 3}, "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}})
    header = parse_header_bytes(raw.encode())
    assert header.metadata["n"] == "3"
