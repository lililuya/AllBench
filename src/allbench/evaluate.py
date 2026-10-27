"""Offline grouped evaluation with an immutable, resumable result identity."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import __version__
from .features import load_bundle
from .io import atomic_json, digest, file_hash, read_json, require
from .metrics import image_similarity, paired_style, reference_style, text_alignment, vendi
from .protocol import KINDS, validate


def ok(value, n, **extra):
    return {"status": "measured", "value": value, "n": n, **extra}


def missing(reason):
    return {"status": "not_measured", "value": None, "reason": reason}


def compute(spec, bundle, arrays):
    prompts = {p["id"]: p for p in spec["prompts"]}
    images = arrays["images"]
    groups = defaultdict(list)
    references = defaultdict(list)
    base = {}
    for image in spec["images"]:
        if image["role"] == "reference":
            references[(image["concept_id"], image["split"])].append(image)
        else:
            kind = prompts[image["prompt_id"]]["kind"]
            groups[(image["concept_id"], kind, image["role"])].append(image)
            if image["role"] == "base":
                base[(image["prompt_id"], image["seed"])] = image
    rows = []
    for (concept, kind, role), samples in sorted(groups.items()):
        samples.sort(key=lambda r: (r["prompt_id"], r["seed"]))
        row = {"concept_id": concept, "prompt_kind": kind, "role": role,
               "images": len(samples), "prompts": len({s["prompt_id"] for s in samples})}
        def vectors(items, key):
            return np.stack([images[s["id"]][key] for s in items])
        row["fidelity"] = {}
        row["style_fidelity"] = {}
        for split in ("train", "holdout"):
            refs = references[(concept, split)]
            if not refs:
                value = missing("NO_" + split.upper() + "_REFERENCE")
            elif "dino" not in bundle["encoders"]:
                value = missing("DINO_NOT_CONFIGURED")
            else:
                value = ok(image_similarity(vectors(refs, "dino"), vectors(samples, "dino")),
                           len(samples), reference_n=len(refs))
            row["fidelity"][split] = value
            if not refs:
                style = missing("NO_" + split.upper() + "_REFERENCE")
            elif "style" not in bundle["encoders"]:
                style = missing("VGG19_NOT_CONFIGURED")
            else:
                style = ok(reference_style(vectors(refs, "style"), vectors(samples, "style")),
                           len(samples), reference_n=len(refs))
            row["style_fidelity"][split] = style
        if "clip" in bundle["encoders"]:
            text = np.stack([arrays["texts"][s["prompt_id"]]["clip"] for s in samples])
            row["text_alignment"] = ok(text_alignment(text, vectors(samples, "clip")), len(samples))
        else:
            row["text_alignment"] = missing("CLIP_NOT_CONFIGURED")
        per_prompt = defaultdict(list)
        for s in samples:
            per_prompt[s["prompt_id"]].append(s)
        if "dino" not in bundle["encoders"]:
            row["diversity"] = missing("DINO_NOT_CONFIGURED")
        elif len(spec["seeds"]) < 2:
            row["diversity"] = missing("DIVERSITY_REQUIRES_TWO_SEEDS")
        else:
            scores = {pid: vendi(vectors(items, "dino")) for pid, items in sorted(per_prompt.items())}
            row["diversity"] = ok(float(np.mean(list(scores.values()))), len(samples),
                                  per_prompt=scores, seeds_per_prompt=len(spec["seeds"]))
        if role == "base":
            row["base_style_drift"] = missing("BASE_IS_COMPARATOR")
        elif "style" not in bundle["encoders"]:
            row["base_style_drift"] = missing("VGG19_NOT_CONFIGURED")
        else:
            paired = [base[(s["prompt_id"], s["seed"])] for s in samples]
            row["base_style_drift"] = ok(paired_style(vectors(paired, "style"), vectors(samples, "style")), len(samples))
        row["exact_reference_copies"] = sum(s["sha256"] in {r["sha256"] for key, ref in references.items()
                                                         if key[0] == concept for r in ref} for s in samples)
        rows.append(row)
    coverage = {c["id"]: {k: sum(p["concept_id"] == c["id"] and p["kind"] == k for p in spec["prompts"])
                           for k in sorted(KINDS)} for c in spec["concepts"]}
    return {"schema_version": "1.0", "protocol_digest": digest(spec), "benchmark_id": spec["benchmark_id"],
            "evaluation_mode": spec.get("evaluation_mode", "paired"),
            "input_origin": spec.get("input_origin", "unspecified"),
            "generator_provenance": spec.get("generator_provenance"),
            "synthetic_images": spec["synthetic"],
            "synthetic_features": any(e["synthetic"] for e in bundle["encoders"].values()),
            "model_effect_validated": False,
            "interpretation": "代理指标；需独立人工校准。合成输入不能证明 LoRA 效果。",
            "protocol": {k: spec[k] for k in ("dataset_release", "models", "generation", "training", "seeds")},
            "encoders": bundle["encoders"], "encoding_runtime": bundle.get("runtime"), "prompt_coverage": coverage,
            "aggregation": "每概念/提示类型/模型分别报告；Vendi逐提示后宏平均；不计算总分。",
            "directions": {"cosine": "higher", "squared_centroid_distance": "lower",
                           "text_alignment": "higher", "diversity": "higher_is_more_diverse",
                           "base_style_drift": "lower_is_less_drift", "style_fidelity": "lower"}, "groups": rows}


def evaluate(manifest, features, out, interrupt_after_checkpoint=False):
    spec, protocol_id = validate(manifest)
    bundle, arrays = load_bundle(features, spec, protocol_id)
    implementation = digest([(p.name, file_hash(p)) for p in sorted(Path(__file__).parent.glob("*.py"))])
    identity = digest({"version": __version__, "implementation": implementation,
                       "protocol": protocol_id, "bundle": file_hash(features)})
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    # Exclusive create; never steal a possibly active process's lock.
    with lock.open("x", encoding="utf-8") as f:
        f.write(identity)
    try:
        done, checkpoint = root / "completed.json", root / "checkpoint.json"
        if done.exists():
            record = read_json(done)
            require(record["identity"] == identity, "RESULT_INPUT_IDENTITY_MISMATCH")
            require(file_hash(root / "report.json") == record["report_sha256"], "RESULT_CHECKSUM_MISMATCH")
            return {"identity": identity, "resumed": True, "computed": False, "report": str(root / "report.json")}
        if checkpoint.exists():
            saved = read_json(checkpoint)
            require(saved["identity"] == identity, "CHECKPOINT_INPUT_IDENTITY_MISMATCH")
            require(digest(saved["report"]) == saved["report_digest"], "CHECKPOINT_CHECKSUM_MISMATCH")
            report, computed = saved["report"], False
        else:
            report, computed = compute(spec, bundle, arrays), True
            atomic_json(checkpoint, {"identity": identity, "report_digest": digest(report), "report": report})
        if interrupt_after_checkpoint:
            raise RuntimeError("INJECTED_INTERRUPTION_AFTER_CHECKPOINT")
        atomic_json(root / "report.json", report)
        atomic_json(done, {"identity": identity, "report_sha256": file_hash(root / "report.json")})
        return {"identity": identity, "resumed": not computed, "computed": computed, "report": str(root / "report.json")}
    finally:
        lock.unlink(missing_ok=True)
