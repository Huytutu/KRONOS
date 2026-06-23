"""Smoke test for the prediction runner via the no-GPU 'mock' system."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_run_multihop_mock(tmp_path):
    qa = ROOT / "data" / "multihop_qa" / "qa.jsonl"
    if not qa.exists():
        import pytest
        pytest.skip("qa.jsonl not generated")

    out = tmp_path / "preds.jsonl"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_multihop.py"),
         "--system", "mock", "--limit", "5", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    preds = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert len(preds) == 5
    assert all({"id", "answer", "cause", "trace"} <= set(p) for p in preds)
