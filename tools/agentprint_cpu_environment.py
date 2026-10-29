#!/usr/bin/env python3
"""Print dependency-import/version evidence as JSON in a short-lived process.

The CPU acceptance orchestrator launches this command before training so it
does not retain imported framework memory while an RF training process runs.
"""
from contextlib import redirect_stdout
import importlib
from importlib.metadata import version
import json
import platform
import sys


def main():
    packages = {}
    # Keep third-party import diagnostics away from the JSON protocol on stdout.
    with redirect_stdout(sys.stderr):
        for module, distribution in [('torch', 'torch'), ('numpy', 'numpy'), ('yaml', 'PyYAML'),
                                     ('dpkt', 'dpkt'), ('sklearn', 'scikit-learn'),
                                     ('tensorboard', 'tensorboard'), ('scapy.all', 'scapy'),
                                     ('unittest', None)]:
            try:
                imported = importlib.import_module(module)
                packages[module] = {'import': 'ok', 'version': version(distribution)
                                    if distribution else platform.python_version()}
                if module == 'torch':
                    packages[module]['tensor_cpu_probe'] = float(imported.tensor([1.0]).sum())
            except Exception as exc:
                packages[module] = {'import': 'failed', 'error': str(exc)}
    print(json.dumps(packages, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
