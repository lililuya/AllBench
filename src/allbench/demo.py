"""Synthetic images/vectors only; deliberately no real LoRA inference."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .features import save_arrays
from .io import atomic_json, file_hash
from .protocol import validate


def make_demo(out):
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require_empty = root / "benchmark.json"
    if require_empty.exists():
        raise FileExistsError("DEMO_ALREADY_EXISTS")
    prompts = []
    for kind, text in (("train", "TOK toy on a table"), ("trigger", "TOK toy"),
                       ("alter", "TOK toy beside a blue cube"), ("style", "watercolor of TOK toy"),
                       ("agnostic-class", "a toy beside a blue cube"), ("agnostic", "a forest")):
        prompts.append({"id": kind, "concept_id": "toy", "kind": kind,
                        "text": text, "evaluation_text": text.replace("TOK", "a red wooden")})
    spec = {"schema_version": "1.0", "benchmark_id": "synthetic-smoke-v1", "synthetic": True,
            "dataset_release": "synthetic-release-v1", "seeds": [7, 11],
            "models": {"base": {"checkpoint": "synthetic-drawing", "revision": "1"},
                       "adapted": {"checkpoint": "synthetic-drawing", "revision": "1", "adapter": {
                           "checkpoint": "no-real-lora", "revision": "1", "method": "synthetic", "scale": 1.0}}},
            "training": {"unique_samples": None, "sample_exposures": None, "steps": None, "seed": None},
            "generation": {"engine": "Pillow-synthetic", "engine_revision": "1", "scheduler": "none",
                           "steps": 1, "width": 224, "height": 224, "cfg": 0.0,
                           "negative_prompt": "", "precision": "uint8"},
            "concepts": [{"id": "toy", "kind": "subject", "trigger": "TOK"}], "prompts": prompts, "images": []}
    def image(row, shift):
        im = Image.new("RGB", (224, 224), (35 + shift, 95, 135))
        draw = ImageDraw.Draw(im)
        draw.rectangle((24 + shift, 32, 120 + shift, 150), fill=(215, 55 + shift, 45))
        draw.ellipse((130, 130 - shift, 205, 205 - shift), fill=(45, 170, 210))
        path = root / (row["id"] + ".png")
        im.save(path)
        row.update(path=path.name, sha256=file_hash(path))
        spec["images"].append(row)
    for i, split in enumerate(("train", "holdout")):
        image({"id": "ref-" + split, "role": "reference", "concept_id": "toy", "split": split,
               "group_id": "independent-" + split}, i * 9)
    for role in ("base", "adapted"):
        for i, p in enumerate(prompts):
            for seed in spec["seeds"]:
                image({"id": f"{role}-{p['id']}-{seed}", "role": role, "concept_id": "toy",
                       "prompt_id": p["id"], "seed": seed}, i * 4 + seed + (role == "adapted") * 2)
    atomic_json(root / "benchmark.json", spec)
    _, identity = validate(root / "benchmark.json")
    features = {"schema_version": "1.0", "protocol_digest": identity,
                "encoders": {k: {"id": "synthetic-vector-fixture", "weights_digest": "no-weights-v1",
                                  "synthetic": True, "preprocess": "fixed-vectors-not-image-encoding"}
                             for k in ("dino", "clip", "style")}, "images": {}, "texts": {}}
    for i, row in enumerate(spec["images"]):
        shift = i / 100
        vector = np.array([1.0, shift, 0.25])
        features["images"][row["id"]] = save_arrays(root / "toy-features", f"image-{i:06}",
            {"dino": vector, "clip": vector, "style": np.array([shift, shift**2, 0.4])})
    for i, p in enumerate(prompts):
        features["texts"][p["id"]] = save_arrays(root / "toy-features", f"text-{i:06}", {"clip": np.array([1., i / 10, .25])})
    atomic_json(root / "toy-features" / "features.json", features)
    return {"manifest": str(root / "benchmark.json"), "features": str(root / "toy-features" / "features.json"),
            "images": len(spec["images"]), "real_lora_inference": False}
