"""The offline demo and the command line entry points."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from agent_supervision_tree import cli
from agent_supervision_tree.agentic.audit import (
    AuditLog,
    anchor_path_for,
    append_anchor,
    verify_file,
)
from agent_supervision_tree.core import tracing
from agent_supervision_tree.core.security import hash_api_key
from agent_supervision_tree.workload.demo import render_tree, run_supervision_demo

pytestmark = pytest.mark.integration


def test_the_demo_runs_offline_and_loses_nothing(tmp_path: Path) -> None:
    summary = run_supervision_demo(tmp_path, documents=24, fault_rate=0.3, seed=11)
    assert summary["lost"] == 0
    assert summary["duplicate_sink_writes"] == 0
    assert summary["injection_leaks"] == 0
    assert int(summary["accepted"]) == (  # type: ignore[arg-type]
        int(summary["posted"]) + int(summary["denied"]) + int(summary["quarantined"])  # type: ignore[arg-type]
    )
    assert int(summary["restarts"]) >= 1  # type: ignore[arg-type]  # a worker is crashed on purpose
    assert int(summary["quarantined"]) >= 1  # type: ignore[arg-type]  # the poison batch
    assert summary["budget_invariants_ok"] is True
    assert summary["exit_code"] == 0
    assert summary["audit_chain_ok"] is True
    assert verify_file(tmp_path / "audit.jsonl", expect_head=str(summary["audit_head"])).ok
    exported = (tmp_path / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(exported) == summary["quarantined"]


def test_the_demo_reports_the_live_tree_and_the_budget_burn(tmp_path: Path) -> None:
    summary = run_supervision_demo(tmp_path, documents=12, fault_rate=0.2, seed=3)
    tree = summary["tree"]
    assert isinstance(tree, dict)
    assert tree["path"] == "root"
    rendered = summary["tree_render"]
    assert isinstance(rendered, list)
    assert rendered[0].startswith("root [")  # type: ignore[union-attr]
    assert any("extractors" in str(line) for line in rendered)
    assert 0 < int(summary["budget_usd_micros_spent"]) <= int(summary["budget_usd_micros_limit"])  # type: ignore[arg-type]
    assert float(summary["goodput_per_second"]) > 0  # type: ignore[arg-type]


def test_render_tree_handles_a_minimal_node() -> None:
    assert render_tree({"path": "root", "running": False}) == ["root [#0 down]"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_one_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path), "--documents", "12", "--seed", "5"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["demo"] == "supervision"
    assert payload["lost"] == 0
    assert "tree" not in payload  # the nested tree stays out of the printed summary
    assert "supervision tree after the run:" in captured.err


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("root", "actor.restarted", {})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


def test_cli_audit_verify_uses_the_anchor_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An anchored log fails closed on truncation without the operator passing a hash."""
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for index in range(3):
        log.append("root", "actor.restarted", {"index": index})
    append_anchor(anchor_path_for(path), entries=3, head=log.head)
    assert cli.main(["audit", "verify", str(path)]) == 0
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:1]), encoding="utf-8", newline="\n")
    assert cli.main(["audit", "verify", str(path)]) == 1
    assert cli.main(["audit", "verify", str(path), "--no-anchor"]) == 0
    assert cli.main(["audit", "verify", str(path), "--anchor", str(anchor_path_for(path))]) == 1
    capsys.readouterr()


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False
    assert captured["app"].state.pipeline.config.workers >= 1


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["agent_supervision_tree", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("agent_supervision_tree", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
