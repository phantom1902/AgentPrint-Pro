#!/usr/bin/env python3
"""Exercise original run_episode -> run_cli_task -> bwrap -> all 8 mocks -> PCAP."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PORTS = (18080, 18081, 18082, 7770, 7780, 8010, 9999, 3000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default="A9900")
    args = parser.parse_args()
    if len(args.episode) != 5 or args.episode[0] != "A" or not args.episode[1:].isdigit():
        parser.error("use an A plus four-digit episode ID")
    required = ("scripts/runner/run_episode.py", "scripts/runner/run_cli_task.py",
                "scripts/env/mock_services.sh", "scripts/label_pcap.py",
                "scripts/visualize_episode.py")
    missing = [name for name in required
               if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
    if missing or len(list((ROOT / "scripts/mock").glob("mock_*_server.py"))) != 8:
        parser.error("collection runtime is incomplete; see README.md and IMPORT_REPORT.json")
    out = ROOT / "episodes" / args.episode
    if out.exists():
        parser.error(f"refusing to overwrite {out}; use another --episode")
    # Do not stop/reuse services belonging to existing experiments.
    for port in PORTS:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", port))
            except OSError:
                parser.error(f"port {port} is occupied; run on an idle collection host")
    fixture = ROOT / ".venv" / "bin" / "flowbench_smoke_fixture"
    if fixture.exists():
        parser.error(f"refusing to overwrite {fixture}")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "tools" / "fixture_agent.py", fixture)
    fixture.chmod(0o755)
    # Test HOME is private and contains no real model credentials or session history.
    with tempfile.TemporaryDirectory(prefix="flowbench-smoke-home-") as home:
        env = dict(os.environ, HOME=home, FLOWBENCH_CLAUDE_CODE_CMD=str(fixture))
        for name in list(env):
            if name.endswith(("_API_KEY", "_AUTH_TOKEN")):
                env.pop(name)
        try:
            subprocess.run([sys.executable, str(ROOT / "scripts/runner/run_episode.py"),
                            args.episode, "--runner", "cli", "--instance",
                            str(ROOT / "examples/mock_smoke.yaml"), "--harness", "claude_code",
                            "--timeout", "30", "--no-tls-keylog"],
                           cwd=ROOT, env=env, check=True, timeout=120)
        finally:
            fixture.unlink(missing_ok=True)
    response = json.loads((out / "cli_response.json").read_text())
    assert response["completed"] and "FLOWBENCH_MOCK_SMOKE_OK" in response["stdout"], response
    assert (out / "events.jsonl").stat().st_size > 0
    packets = subprocess.run(["tcpdump", "-nn", "-r", str(out / "network.pcap")],
                             capture_output=True, text=True, check=True).stdout
    for port in PORTS:
        assert f".{port}:" in packets or f".{port} " in packets, f"no captured traffic for port {port}"
    for script in ("label_pcap.py", "visualize_episode.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), str(out)], check=True)
    for name in ("labeled_packets.csv", "label_summary.csv", "episode_chart.png"):
        assert (out / name).stat().st_size > 0, name
    namespaces = subprocess.run(["sudo", "-n", "ip", "netns", "list"],
                                capture_output=True, text=True, check=True).stdout
    assert f"fb_{args.episode}" not in namespaces
    print(f"PASS: original runner, bwrap, 8 mocks, PCAP, events, response, labels, chart, namespace cleanup: {out}")


if __name__ == "__main__":
    main()
