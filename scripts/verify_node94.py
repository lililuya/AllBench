"""Metadata-only verification of the isolated synthetic node94 smoke run."""
import argparse
import json
from pathlib import Path

from allbench.encode import encode
from allbench.evaluate import evaluate
from allbench.io import atomic_json, file_hash, read_json

parser = argparse.ArgumentParser()
parser.add_argument("run_root")
parser.add_argument("--dino", required=True)
args = parser.parse_args()
root = Path(args.run_root)
manifest = root / "synthetic/benchmark.json"
features = root / "dino-features/features.json"
meta = read_json(features)
report = read_json(root / "dino-results/report.json")
assert len(meta["images"]) == 26 and len(report["groups"]) == 12
assert report["synthetic_images"] and not report["synthetic_features"]
assert all(r["fidelity"]["holdout"]["status"] == "measured" for r in report["groups"])
assert all(r["text_alignment"]["status"] == "not_measured" for r in report["groups"])
assert all(r["diversity"]["status"] == "measured" for r in report["groups"])
encoding_replay = encode(manifest, root / "dino-features", dino=args.dino, device="cuda:0")
assert encoding_replay["resumed"] and encoding_replay["encoded_images"] == 0
recovery = root / "dino-recovery"
if not (recovery / "checkpoint.json").exists():
    try:
        evaluate(manifest, features, recovery, interrupt_after_checkpoint=True)
    except RuntimeError as exc:
        assert "INJECTED" in str(exc)
    else:
        raise AssertionError("failure injection did not fire")
    assert not (recovery / "completed.json").exists()
resumed = evaluate(manifest, features, recovery)
assert resumed["resumed"] and not resumed["computed"]
assert file_hash(root / "dino-results/report.json") == file_hash(recovery / "report.json")
summary = {"synthetic_images": 26, "generation_jobs": 24, "report_groups": 12,
           "pretrained_encoder": meta["encoders"]["dino"], "runtime": meta["runtime"],
           "report_sha256": file_hash(recovery / "report.json"),
           "recovered_without_metric_recompute": True, "encoding_replay_new_images": 0,
           "clip_pretrained_tested": False, "vgg19_pretrained_tested": False,
           "real_lora_inference": False, "model_effect_validated": False}
atomic_json(root / "verification-summary.json", summary)
print(json.dumps(summary, ensure_ascii=False))
