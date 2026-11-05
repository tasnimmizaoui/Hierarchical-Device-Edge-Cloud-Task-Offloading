"""Print required/optional package versions and a few shared model checks."""

from __future__ import annotations

import importlib.metadata
import sys


PACKAGES = ("numpy", "pandas", "matplotlib", "PyYAML", "simpy", "gymnasium", "stable-baselines3")
print(f"Python: {sys.version.split()[0]}")
for package in PACKAGES:
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

try:
    from mec_offloading.domain.metrics import transmission_time_s

    print(f"Smoke formula: 1 MB over 8 Mbps = {transmission_time_s(1.0, 8.0):.3f} s")
except ImportError as exc:
    raise SystemExit("Project package is not installed. Run: pip install -e '.[dev]'") from exc
