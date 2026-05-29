#!/usr/bin/env python3
"""Parse QAT training logs, filtering out tqdm progress bar noise.

Usage:
    python scripts/qat_log_parser.py /tmp/qat-debug.log
    python scripts/qat_log_parser.py /tmp/qat-debug.log --errors-only
"""

import re
import sys


def parse_log(path: str, errors_only: bool = False) -> list[str]:
    tqdm_pattern = re.compile(r"\d+%|it/s|s/it|\|[█▏▎▍▌▋▊▉ ]+\|")
    with open(path) as f:
        lines = f.readlines()

    results = []
    for line in lines:
        clean = line.rstrip()
        # Skip pure tqdm progress bars
        if tqdm_pattern.search(clean) and not re.search(r"INFO|ERROR|WARN|loss|metric", clean):
            continue
        # Skip empty lines and ANSI escapes only
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", clean).strip()
        if not stripped:
            continue
        if errors_only and "ERROR" not in clean and "Traceback" not in clean:
            continue
        results.append(clean)
    return results


if __name__ == "__main__":
    path = sys.argv[1]
    errors_only = "--errors-only" in sys.argv
    for line in parse_log(path, errors_only):
        print(line)
