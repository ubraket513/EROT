"""Incomplete timing records must never silently become complete measurements."""

import json

from erot.runtime.timing import summarize_timing


def test_summarizes_complete_timing(tmp_path):
    path = tmp_path / "timing.jsonl"
    events = [
        {"chunk": 1, "compile_and_execute_seconds": 2.0, "checkpoint_seconds": 0.1},
        {"chunk": 2, "execute_seconds": 0.5, "checkpoint_seconds": 0.2},
    ]
    path.write_text("".join(json.dumps(x) + "\n" for x in events))
    result = summarize_timing(path, 2)
    assert result["complete"]
    assert result["compile_and_execute_seconds"] == 2
    assert result["execute_seconds"] == 0.5
    assert abs(result["checkpoint_seconds"] - 0.3) < 1e-12


def test_missing_duplicate_or_truncated_events_are_incomplete(tmp_path):
    path = tmp_path / "timing.jsonl"
    assert not summarize_timing(path, 2)["complete"]
    event = json.dumps({"chunk": 1, "execute_seconds": 0.1, "checkpoint_seconds": 0.1})
    for content in [event + "\n", event + "\n" + event + "\n", event + '\n{"chunk":']:
        path.write_text(content)
        assert not summarize_timing(path, 2)["complete"]
