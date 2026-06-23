"""Smoke test for the eval CLI — runs scripts/eval_multihop.py end-to-end."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_eval_multihop_cli(tmp_path):
    qa = tmp_path / "qa.jsonl"
    pred = tmp_path / "pred.jsonl"
    out = tmp_path / "report.json"

    qa.write_text(json.dumps({
        "id": "n1", "finding_a": "Calcification", "finding_b": "Aortic enlargement",
        "answer": "No", "gold_causes": [], "support_edges": [],
        "hops": 2, "single_cause": False,
    }) + "\n", encoding="utf-8")
    pred.write_text(json.dumps({
        "id": "n1", "answer": "No", "cause": None, "trace": [],
    }) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval_multihop.py"),
         "--qa", str(qa), "--pred", str(pred), "--system", "t", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["system"] == "t"
    assert data["n"] == 1
    assert data["binary_accuracy"] == 1.0
    assert "load_bearing_rate" in data
