"""A content pattern starting with "-" is a pattern, never a ripgrep/grep flag."""

import json
import shutil

import pytest

from tools.file_tools import search_tool


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = tmp_path / "proj"
    d.mkdir()
    (d / "team_config.lua").write_text("rank_range = {-20, 20}\n", newline="\n")
    return d


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_rg_dash_prefixed_pattern(proj):
    r = json.loads(search_tool("-20|rank_range", path=str(proj), task_id="t-dash"))
    assert "error" not in r
    assert r["total_count"] == 1


def test_grep_fallback_dash_prefixed_pattern(proj, monkeypatch):
    from tools.file_operations import ShellFileOperations

    has_command = ShellFileOperations._has_command
    monkeypatch.setattr(ShellFileOperations, "_has_command",
                        lambda self, command: command != "rg" and has_command(self, command))
    r = json.loads(search_tool("-20|rank_range", path=str(proj), task_id="t-dash-grep"))
    assert "error" not in r
    assert r["total_count"] == 1
