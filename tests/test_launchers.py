import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_files_exist_and_point_at_the_module():
    command = ROOT / "Launch TriggerCollide.command"
    windows = ROOT / "Launch TriggerCollide.bat"
    shell = ROOT / "launch.sh"
    desktop = ROOT / "triggercollide.desktop"
    for path in (command, windows, shell, desktop):
        assert path.is_file(), path
    assert os.access(command, os.X_OK)
    assert os.access(shell, os.X_OK)
    for path in (command, windows, shell):
        text = path.read_text(encoding="utf-8")
        assert "triggercollide" in text
        assert "https://www.python.org/downloads/" in text
        assert "-m triggercollide" in text
    desktop_text = desktop.read_text(encoding="utf-8")
    assert "launch.sh" in desktop_text
    assert "icon" in desktop_text.lower()
    assert b"\r\n" in windows.read_bytes()


def test_icons_and_cover_exist():
    for name in ("icon.svg", "icon.png", "icon-512.png", "icon-1024.png", "icon.icns", "icon.ico"):
        assert (ROOT / "assets" / name).is_file(), name
    assert (ROOT / "docs" / "cover.jpg").is_file()
    assert (ROOT / "triggercollide" / "web" / "favicon.ico").is_file()


def test_no_lan_address_is_hardcoded():
    for path in (ROOT / "triggercollide").rglob("*.py"):
        assert "192.168." not in path.read_text(encoding="utf-8"), path
