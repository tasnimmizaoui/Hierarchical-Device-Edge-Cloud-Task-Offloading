"""Validate a local Alibaba batch_instance.csv subset and report its audit counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from mec_offloading.trace_workloads import load_alibaba_batch_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    result = load_alibaba_batch_instances(args.input)
    print(f"source={result.source}")
    print(f"selected={result.selected_count}")
    print(f"rejected={result.rejected_count}")
    for rejection in result.rejected:
        print(f"row={rejection.row_number}: {rejection.reason}")


if __name__ == "__main__":
    main()
