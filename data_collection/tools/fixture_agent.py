#!/usr/bin/python3
"""Offline CLI fixture for smoke_collection.py, accepts/ignores harness CLI flags."""
import json
import os
import urllib.request

assert os.geteuid() != 0, "task must not run as root"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for port in (18080, 18081, 18082, 7770, 7780, 8010, 9999, 3000):
    path = "/health" if port >= 18080 else "/"
    with opener.open(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        assert response.status == 200, port
        print(json.dumps({"mock_port": port, "status": "ok"}), flush=True)
print("FLOWBENCH_MOCK_SMOKE_OK", flush=True)
