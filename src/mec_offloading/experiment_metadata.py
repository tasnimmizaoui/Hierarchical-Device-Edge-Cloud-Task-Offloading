"""Stable hashes for experiment protocol snapshots."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def configuration_hash(configuration: Mapping[str, object]) -> str:
    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
