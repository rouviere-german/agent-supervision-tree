"""Command line entry point: ``agent-supervision-tree`` or ``python -m agent_supervision_tree``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from agent_supervision_tree.core.config import get_settings  # noqa: PLC0415
    from agent_supervision_tree.core.logging import configure_logging  # noqa: PLC0415
    from agent_supervision_tree.core.tracing import install_global_tracer_provider  # noqa: PLC0415
    from agent_supervision_tree.service import create_service_app  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_service_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _demo(workdir: Path, documents: int, fault_rate: float, seed: int) -> int:
    from agent_supervision_tree.core.logging import configure_logging  # noqa: PLC0415
    from agent_supervision_tree.workload.demo import run_supervision_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    summary = run_supervision_demo(workdir, documents=documents, fault_rate=fault_rate, seed=seed)
    rendered = summary.get("tree_render")
    if isinstance(rendered, list):
        print("supervision tree after the run:", file=sys.stderr)
        for line in rendered:
            print(f"  {line}", file=sys.stderr)
    printable = {key: value for key, value in summary.items() if key != "tree"}
    print(json.dumps({"demo": "supervision", **printable}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-supervision-tree")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)
    verify.add_argument("--anchor", type=Path, default=None, help="default: <log>.anchor")
    verify.add_argument("--no-anchor", action="store_true", help="verify the chain alone")

    demo = sub.add_parser("demo", help="run the offline fault-injection demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))
    demo.add_argument("--documents", type=int, default=48)
    demo.add_argument("--fault-rate", type=float, default=0.3)
    demo.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "generate-api-key":
        from agent_supervision_tree.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "audit":
        from agent_supervision_tree.agentic.audit import main as audit_main  # noqa: PLC0415

        forwarded = ["verify", str(args.path)]
        if args.expect_head:
            forwarded += ["--expect-head", args.expect_head]
        if args.anchor is not None:
            forwarded += ["--anchor", str(args.anchor)]
        if args.no_anchor:
            forwarded.append("--no-anchor")
        return audit_main(forwarded)
    return _demo(args.workdir, args.documents, args.fault_rate, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
