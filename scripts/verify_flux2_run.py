"""Verify base-only generation, partial recovery and offline evaluation metadata."""
import argparse
from pathlib import Path

from allbench.io import atomic_json, file_hash, read_json
from allbench.protocol import validate

parser = argparse.ArgumentParser()
parser.add_argument("run_root")
parser.add_argument("--capture-partial", action="store_true")
args = parser.parse_args()
root = Path(args.run_root)
generated = root / "generated-final"
proof_path = root / "partial-recovery-proof.json"
if args.capture_partial:
    receipts = []
    for p in generated.glob("*.json"):
        record = read_json(p)
        if "job" in record:
            assert file_hash(generated / record["path"]) == record["sha256"]
            receipts.append({"receipt": p.name, "receipt_sha256": file_hash(p), "image": record["path"], "sha256": record["sha256"]})
    assert len(receipts) == 2
    atomic_json(proof_path, {"completed_before_resume": receipts})
    print({"partial_committed": 2})
else:
    spec, protocol_id = validate(generated / "benchmark.json")
    assert spec["evaluation_mode"] == "base_only" and set(spec["models"]) == {"base"}
    assert len(spec["images"]) == 6 and all(r["role"] == "base" for r in spec["images"])
    for r in read_json(proof_path)["completed_before_resume"]:
        assert file_hash(generated / r["receipt"]) == r["receipt_sha256"]
        assert file_hash(generated / r["image"]) == r["sha256"]
    report = read_json(root / "report/report.json")
    features = read_json(root / "features/features.json")
    assert features["protocol_digest"] == protocol_id and len(features["images"]) == 6
    assert not report["synthetic_features"] and report["input_origin"] == "model_generated"
    assert report["evaluation_mode"] == "base_only" and len(report["groups"]) == 1
    assert report["groups"][0]["diversity"]["status"] == "measured"
    assert all(v["status"] == "not_measured" for v in report["groups"][0]["fidelity"].values())
    completion = read_json(root / "completion-attempt.json")
    replay = read_json(generated / "generation-summary.json")
    assert completion["generated_this_attempt"] == 4 and completion["reused"] == 2
    assert replay["generated_this_attempt"] == 0 and replay["reused"] == 6
    assert file_hash(root / "report/report.json") == file_hash(root / "report-recovered/report.json")
    receipts = [read_json(generated / r["generation_receipt"]) for r in spec["images"]]
    summary = {"expected_images": 6, "generated_images": len(spec["images"]), "prompt_count": 3,
               "seeds": spec["seeds"], "generation": spec["generation"],
               "generator_provenance": spec["generator_provenance"], "partial_committed": 2,
               "resumed_new_images": 4, "replayed_new_images": 0,
               "generation_seconds": completion["generation_seconds_total"],
               "resume_load_seconds": completion["load_seconds_this_attempt"],
               "per_image_seconds": [r["seconds"] for r in receipts],
               "peak_allocated_bytes": max(r["metrics"]["peak_allocated_bytes"] for r in receipts),
               "dino_images": len(features["images"]), "dino_runtime": features["runtime"],
               "report_sha256": file_hash(root / "report/report.json"),
               "recovered_report_equal": True, "lora_tested": False, "model_effect_validated": False,
               "unmeasured": ["reference_fidelity", "clip_alignment", "paired_style_drift", "lora_effect"]}
    atomic_json(root / "verification-summary.json", summary)
    print(summary)
