"""Audit lifetime chunk timing coverage independently from scientific results."""

import json
import math
from pathlib import Path


def summarize_timing(path, chunks):
    """Sum valid events, marking missing, duplicate or malformed logs incomplete."""
    totals = dict.fromkeys(
        ("compile_and_execute_seconds", "execute_seconds", "checkpoint_seconds"), 0.0
    )
    seen, invalid = set(), 0
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    for line in lines:
        try:
            event = json.loads(line)
            chunk = event["chunk"]
            if type(chunk) is not int or not 1 <= chunk <= chunks or chunk in seen:
                raise ValueError("invalid chunk")
            execution = [
                key for key in totals if key != "checkpoint_seconds" and key in event
            ]
            if len(execution) != 1:
                raise ValueError("missing or ambiguous execution timing")
            values = {key: event[key] for key in [execution[0], "checkpoint_seconds"]}
            if any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in values.values()
            ):
                raise ValueError("invalid duration")
            seen.add(chunk)
            for key, value in values.items():
                totals[key] += value
        except (ValueError, KeyError, TypeError):
            invalid += 1
    return {
        "complete": invalid == 0 and len(seen) == chunks,
        "expected_chunks": chunks,
        "recorded_chunks": len(seen),
        "invalid_events": invalid,
        **totals,
    }
