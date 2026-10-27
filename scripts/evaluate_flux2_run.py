"""Evaluate a completed Klein base-only run and check zero-compute recovery."""
import argparse
from pathlib import Path

from allbench.evaluate import evaluate
from allbench.io import atomic_json, file_hash

parser = argparse.ArgumentParser()
parser.add_argument("run_root")
root = Path(parser.parse_args().run_root)
manifest = root / "generated-final/benchmark.json"
features = root / "features/features.json"
first = evaluate(manifest, features, root / "report")
replay = evaluate(manifest, features, root / "report")
assert replay["resumed"] and not replay["computed"]
recovery = root / "report-recovered"
if not (recovery / "checkpoint.json").exists():
    try:
        evaluate(manifest, features, recovery, interrupt_after_checkpoint=True)
    except RuntimeError as exc:
        assert str(exc) == "INJECTED_INTERRUPTION_AFTER_CHECKPOINT"
    else:
        raise AssertionError("failure injection did not fire")
    assert not (recovery / "completed.json").exists()
recovered = evaluate(manifest, features, recovery)
assert recovered["resumed"] and not recovered["computed"]
assert file_hash(first["report"]) == file_hash(recovered["report"])
summary = {"replay_computed": replay["computed"], "recovery_computed": recovered["computed"],
           "report_sha256": file_hash(first["report"]), "reports_equal": True}
atomic_json(root / "evaluation-verification.json", summary)
print(summary)
