"""Checksummed feature bundles; no pickle or implicit encoder fallback."""
from pathlib import Path

import numpy as np

from .io import atomic_json, file_hash, read_json, require


def save_arrays(root, name, arrays):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Callers hold the bundle lock. Incomplete files never gain a manifest reference.
    path = root / (name + ".npz")
    tmp = root / (name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return {"path": path.name, "sha256": file_hash(path)}


def load_bundle(path, spec, identity):
    path = Path(path)
    meta = read_json(path)
    require(meta.get("schema_version") == "1.0" and meta.get("protocol_digest") == identity,
            "FEATURE_INPUT_IDENTITY_MISMATCH")
    encoders = meta.get("encoders", {})
    require(encoders and set(encoders) <= {"dino", "clip", "style"}, "ENCODER_NAMES")
    for encoder in encoders.values():
        require(all(encoder.get(k) for k in ("id", "weights_digest", "preprocess"))
                and type(encoder.get("synthetic")) is bool, "ENCODER_IDENTITY_REQUIRED")
    expected_images = {r["id"] for r in spec["images"]}
    expected_text = {r["id"] for r in spec["prompts"]} if "clip" in encoders else set()
    require(set(meta.get("images", {})) == expected_images, "FEATURE_IMAGE_MEMBERS_MISMATCH")
    require(set(meta.get("texts", {})) == expected_text, "FEATURE_TEXT_MEMBERS_MISMATCH")
    loaded = {}
    dims = {}
    for kind in ("images", "texts"):
        loaded[kind] = {}
        for sid, record in meta[kind].items():
            file = (path.parent / record["path"]).resolve()
            require(file_hash(file) == record["sha256"], "FEATURE_CHECKSUM_MISMATCH")
            with np.load(file, allow_pickle=False) as arrays:
                expected = set(encoders) if kind == "images" else {"clip"}
                require(set(arrays.files) == expected, "FEATURE_ENCODERS_MISMATCH")
                loaded[kind][sid] = {}
                for key in arrays.files:
                    a = np.asarray(arrays[key], dtype=np.float64)
                    require(a.ndim == 1 and len(a) and np.isfinite(a).all(), "INVALID_FEATURE_VECTOR")
                    require(key not in dims or dims[key] == len(a), "FEATURE_DIMENSION_MISMATCH")
                    if key != "style":
                        require(np.linalg.norm(a) > 0, "ZERO_FEATURE_VECTOR")
                    dims[key] = len(a)
                    loaded[kind][sid][key] = a
    return meta, loaded
