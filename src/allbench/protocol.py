"""Versioned manifest: explicit prompts, matched base/adapted cells and references."""
import math
from pathlib import Path

from PIL import Image

from .io import digest, file_hash, read_json, require

KINDS = {"train", "trigger", "alter", "style", "agnostic-class", "agnostic"}


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def unique(rows, name):
    require(isinstance(rows, list) and rows, "EMPTY_" + name)
    result = {}
    for row in rows:
        require(nonempty(row.get("id")) and row["id"] not in result, "DUPLICATE_OR_INVALID_" + name)
        result[row["id"]] = row
    return result


def validate(path, require_outputs=True):
    path = Path(path).resolve()
    spec = read_json(path)
    digest(spec)  # Reject all non-finite nested metadata too.
    require(spec.get("schema_version") == "1.0", "SCHEMA_VERSION")
    require(nonempty(spec.get("benchmark_id")), "BENCHMARK_ID")
    require(type(spec.get("synthetic")) is bool, "SYNTHETIC_FLAG_REQUIRED")
    require(nonempty(spec.get("dataset_release")), "DATASET_RELEASE_REQUIRED")
    mode = spec.get("evaluation_mode", "paired")
    require(mode in ("paired", "base_only"), "EVALUATION_MODE")
    roles = {"base", "adapted"} if mode == "paired" else {"base"}
    require(set(spec.get("models", {})) == roles, "BASE_AND_ADAPTED_REQUIRED")
    for model in spec["models"].values():
        require(all(nonempty(model.get(k)) for k in ("checkpoint", "revision")), "MODEL_IDENTITY_REQUIRED")
    require("adapter" not in spec["models"]["base"], "BASE_MUST_NOT_HAVE_ADAPTER")
    if mode == "paired":
        require(spec["models"]["base"]["checkpoint"] == spec["models"]["adapted"]["checkpoint"]
                and spec["models"]["base"]["revision"] == spec["models"]["adapted"]["revision"],
                "ADAPTED_BASE_MISMATCH")
        adapter = spec["models"]["adapted"].get("adapter", {})
        require(all(nonempty(adapter.get(k)) for k in ("checkpoint", "revision", "method")), "ADAPTER_IDENTITY_REQUIRED")
        require(type(adapter.get("scale")) in (int, float) and math.isfinite(adapter["scale"]), "ADAPTER_SCALE_REQUIRED")
    training = spec.get("training", {})
    for key in ("unique_samples", "sample_exposures", "steps", "seed"):
        require(key in training and (training[key] is None or
                type(training[key]) is int and training[key] >= 0), "TRAINING_BUDGET_" + key)
    g = spec.get("generation", {})
    for key in ("engine", "engine_revision", "scheduler", "negative_prompt", "precision"):
        require(isinstance(g.get(key), str) and (bool(g[key]) or key == "negative_prompt"), "GENERATION_" + key)
    for key in ("steps", "width", "height"):
        require(type(g.get(key)) is int and g[key] > 0, "GENERATION_" + key)
    require(type(g.get("cfg")) in (int, float) and math.isfinite(g["cfg"]) and g["cfg"] >= 0, "GENERATION_CFG")
    seeds = spec.get("seeds")
    require(isinstance(seeds, list) and seeds and all(type(s) is int and 0 <= s < 2**63 for s in seeds)
            and len(seeds) == len(set(seeds)), "INVALID_SEEDS")
    concepts = unique(spec.get("concepts"), "CONCEPT")
    prompts = unique(spec.get("prompts"), "PROMPT")
    for c in concepts.values():
        require(c.get("kind") in ("subject", "style") and nonempty(c.get("trigger")), "CONCEPT_KIND_OR_TRIGGER")
    for p in prompts.values():
        require(p.get("concept_id") in concepts and p.get("kind") in KINDS, "PROMPT_CLASSIFICATION")
        require(nonempty(p.get("text")) and nonempty(p.get("evaluation_text")), "PROMPT_TEXT_REQUIRED")
        trigger = concepts[p["concept_id"]]["trigger"]
        if p["kind"].startswith("agnostic"):
            require(trigger not in p["text"], "AGNOSTIC_CONTAINS_TRIGGER")
        else:
            require(trigger in p["text"], "CONCEPT_PROMPT_MISSING_TRIGGER")
        require(trigger not in p["evaluation_text"], "EVALUATION_TEXT_REQUIRES_SEMANTIC_REPLACEMENT")
    images = ({} if mode == "base_only" and spec.get("images") == []
              else unique(spec.get("images"), "IMAGE"))
    seen_cells, ref_splits, groups, refs, ref_members = set(), {}, {}, set(), set()
    for row in images.values():
        role = row.get("role")
        require(role in roles | {"reference"}, "IMAGE_ROLE")
        require(row.get("concept_id") in concepts, "IMAGE_CONCEPT")
        if role == "reference":
            require(row.get("split") in ("train", "holdout") and nonempty(row.get("group_id")), "REFERENCE_SPLIT_GROUP")
            refs.add(row["concept_id"])
            group = row["group_id"]
            require(group not in groups or groups[group] == row["split"], "CROSS_SPLIT_GROUP_LEAK")
            groups[group] = row["split"]
        else:
            require(row.get("prompt_id") in prompts and type(row.get("seed")) is int and row["seed"] in seeds, "UNKNOWN_OUTPUT_CELL")
            p = prompts[row["prompt_id"]]
            require(p["concept_id"] == row["concept_id"], "OUTPUT_CONCEPT_MISMATCH")
            cell = (role, row["prompt_id"], row["seed"])
            require(cell not in seen_cells, "DUPLICATE_OUTPUT_CELL")
            seen_cells.add(cell)
        require(nonempty(row.get("path")), "IMAGE_PATH_REQUIRED")
        image_path = (path.parent / row["path"]).resolve()
        require(image_path.is_file(), "IMAGE_MISSING:" + row["id"])
        require(row.get("sha256") == file_hash(image_path), "IMAGE_CHECKSUM_MISMATCH:" + row["id"])
        if "generation_receipt" in row:
            receipt_path = (path.parent / row["generation_receipt"]).resolve()
            require(file_hash(receipt_path) == row.get("receipt_sha256"), "GENERATION_RECEIPT_CHECKSUM_MISMATCH")
            receipt = read_json(receipt_path)
            require(role != "reference" and receipt["sha256"] == row["sha256"]
                    and receipt["job"]["role"] == role and receipt["job"]["prompt_id"] == row["prompt_id"]
                    and receipt["job"]["seed"] == row["seed"]
                    and receipt["job"]["text"] == prompts[row["prompt_id"]]["text"], "GENERATION_RECEIPT_PAIR_MISMATCH")
            provenance = spec.get("generator_provenance", {})
            require(receipt["identity"] == provenance.get("identity")
                    and receipt["components"] == provenance.get("components"), "GENERATION_PROVENANCE_MISMATCH")
        with Image.open(image_path) as im:
            im.verify()
        if role == "reference":
            h = row["sha256"]
            require(h not in ref_splits or ref_splits[h] == row["split"], "CROSS_SPLIT_EXACT_LEAK")
            ref_splits[h] = row["split"]
            member = (row["concept_id"], row["split"], h)
            require(member not in ref_members, "DUPLICATE_REFERENCE_IMAGE")
            ref_members.add(member)
        else:
            with Image.open(image_path) as im:
                require(im.size == (g["width"], g["height"]), "OUTPUT_SIZE_MISMATCH")
    if mode == "paired":
        require(refs == set(concepts), "REFERENCE_REQUIRED_FOR_EACH_CONCEPT")
    expected = {(role, pid, seed) for role in roles for pid in prompts for seed in seeds}
    if require_outputs:
        require(seen_cells == expected, "INCOMPLETE_OUTPUT_MATRIX")
    return spec, digest(spec)


def plan(path):
    spec, identity = validate(path, require_outputs=False)
    return {"schema_version": "1.0", "protocol_digest": identity,
            "generation": spec["generation"], "models": spec["models"],
            "jobs": [{"role": role, "concept_id": p["concept_id"], "prompt_id": p["id"],
                      "text": p["text"], "seed": seed}
                     for role in spec["models"] for p in spec["prompts"] for seed in spec["seeds"]]}
