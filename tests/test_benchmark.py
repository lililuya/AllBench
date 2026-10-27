import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from allbench.demo import make_demo
from allbench.evaluate import evaluate
from allbench.io import atomic_json, file_hash, read_json
from allbench.metrics import gram_features, image_similarity, paired_style, text_alignment, vendi
from allbench.protocol import plan, validate


@pytest.fixture
def demo(tmp_path):
    return make_demo(tmp_path / "demo")


def test_hand_calculated_metrics():
    a = [[1, 0], [0, 1]]
    assert image_similarity(a, a) == {"cosine": .5, "squared_centroid_distance": 0.0}
    assert image_similarity([[1, 0]], [[-1, 0]]) == {"cosine": -1., "squared_centroid_distance": 4.}
    assert text_alignment(a, a) == pytest.approx(1)
    assert vendi(a) == pytest.approx(2)
    assert vendi([[1, 0], [1, 0]]) == pytest.approx(1)
    assert vendi(np.eye(3)) == pytest.approx(3)
    assert paired_style([[1, 2]], [[3, 4]]) == pytest.approx(8)
    grams = gram_features([np.array([[[2., 0.]], [[0., 2.]]])])
    assert grams.tolist() == [1., 0., 0., 1.]


@pytest.mark.parametrize("bad", [[], [[0, 0]], [[float('nan'), 1]], [[float('inf'), 1]], [1, 2]])
def test_bad_vectors_rejected(bad):
    with pytest.raises(ValueError):
        image_similarity(bad, [[1, 0]])


def test_vendi_one_seed_and_mismatched_pair_rejected():
    with pytest.raises(ValueError, match="TWO_SEEDS"):
        vendi([[1, 0]])
    with pytest.raises(ValueError, match="SHAPE"):
        text_alignment([[1, 0]], [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="SHAPE"):
        paired_style([[1, 0]], [[1, 0, 1]])


def test_end_to_end_and_exact_replay(demo, tmp_path):
    spec, _ = validate(demo["manifest"])
    assert len(plan(demo["manifest"])["jobs"]) == 24
    first = evaluate(demo["manifest"], demo["features"], tmp_path / "results")
    report = read_json(first["report"])
    assert len(report["groups"]) == 12
    assert report["synthetic_images"] and report["synthetic_features"]
    assert not report["model_effect_validated"]
    assert all(row["diversity"]["seeds_per_prompt"] == 2 for row in report["groups"])
    assert all(sum(report["prompt_coverage"][c["id"]].values()) == 6 for c in spec["concepts"])
    before = file_hash(first["report"])
    resumed = evaluate(demo["manifest"], demo["features"], tmp_path / "results")
    assert resumed["resumed"] and not resumed["computed"]
    assert file_hash(first["report"]) == before


def test_checkpoint_recovery_without_recompute(demo, tmp_path, monkeypatch):
    out = tmp_path / "results"
    with pytest.raises(RuntimeError, match="INJECTED"):
        evaluate(demo["manifest"], demo["features"], out, True)
    assert not (out / "completed.json").exists()
    assert not (out / ".lock").exists()
    (out / "orphan.tmp").write_text("incomplete")
    monkeypatch.setattr("allbench.evaluate.compute", lambda *args: pytest.fail("recomputed"))
    assert not evaluate(demo["manifest"], demo["features"], out)["computed"]


@pytest.mark.parametrize("mutation,code", [
    (lambda s: s["images"].pop(), "INCOMPLETE_OUTPUT_MATRIX"),
    (lambda s: s["images"].append(copy.deepcopy(s["images"][-1])), "DUPLICATE_OR_INVALID_IMAGE"),
    (lambda s: s["images"][-1].update(seed=999), "UNKNOWN_OUTPUT_CELL"),
    (lambda s: s["seeds"].append(7), "INVALID_SEEDS"),
    (lambda s: s["images"][1].update(group_id=s["images"][0]["group_id"]), "GROUP_LEAK"),
    (lambda s: s["images"][1].update(path=s["images"][0]["path"], sha256=s["images"][0]["sha256"]), "EXACT_LEAK"),
    (lambda s: s["prompts"][0].update(evaluation_text="TOK toy"), "SEMANTIC_REPLACEMENT"),
    (lambda s: s["prompts"][-1].update(text="TOK forest"), "AGNOSTIC_CONTAINS_TRIGGER"),
    (lambda s: s["generation"].update(steps=True), "GENERATION_steps"),
    (lambda s: s["training"].pop("steps"), "TRAINING_BUDGET"),
    (lambda s: s["models"]["adapted"].update(revision="other"), "BASE_MISMATCH"),
])
def test_contract_failures(demo, mutation, code):
    spec = read_json(demo["manifest"])
    mutation(spec)
    atomic_json(demo["manifest"], spec)
    with pytest.raises(ValueError, match=code):
        validate(demo["manifest"])


def test_protocol_change_refuses_old_features(demo, tmp_path):
    spec = read_json(demo["manifest"])
    spec["generation"]["cfg"] += 1
    atomic_json(demo["manifest"], spec)
    with pytest.raises(ValueError, match="FEATURE_INPUT_IDENTITY"):
        evaluate(demo["manifest"], demo["features"], tmp_path / "results")


def test_image_mutation_rejected(demo):
    from pathlib import Path
    spec = read_json(demo["manifest"])
    image = Path(demo["manifest"]).parent / spec["images"][0]["path"]
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="IMAGE_CHECKSUM"):
        validate(demo["manifest"])


def test_feature_and_report_corruption(demo, tmp_path):
    from pathlib import Path
    out = tmp_path / "results"
    result = evaluate(demo["manifest"], demo["features"], out)
    Path(result["report"]).write_text("{}")
    with pytest.raises(ValueError, match="RESULT_CHECKSUM"):
        evaluate(demo["manifest"], demo["features"], out)
    features = read_json(demo["features"])
    file = Path(demo["features"]).parent / next(iter(features["images"].values()))["path"]
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="FEATURE_CHECKSUM"):
        evaluate(demo["manifest"], demo["features"], tmp_path / "other")


def test_missing_encoders_are_unmeasured(demo, tmp_path):
    from pathlib import Path
    from allbench.features import save_arrays
    features = read_json(demo["features"])
    features["encoders"] = {"dino": features["encoders"]["dino"]}
    features["texts"] = {}
    root = Path(demo["features"]).parent
    for index, (sid, record) in enumerate(features["images"].items()):
        with np.load(root / record["path"], allow_pickle=False) as a:
            vector = a["dino"]
        features["images"][sid] = save_arrays(root, f"only-dino-{index}", {"dino": vector})
    atomic_json(demo["features"], features)
    result = evaluate(demo["manifest"], demo["features"], tmp_path / "results")
    report = read_json(result["report"])
    assert all(r["text_alignment"]["status"] == "not_measured" for r in report["groups"])
    assert all(r["base_style_drift"]["value"] is None for r in report["groups"])


def test_output_lock_not_stolen(demo, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / ".lock").write_text("other-worker")
    with pytest.raises(FileExistsError):
        evaluate(demo["manifest"], demo["features"], out)
    assert (out / ".lock").read_text() == "other-worker"


def test_cli_subprocess(demo, tmp_path):
    out = tmp_path / "jobs.json"
    completed = subprocess.run([sys.executable, "-m", "allbench.cli", "plan", demo["manifest"], "--out", str(out)],
                               check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout)["jobs"] == 24
    assert out.is_file()


def test_missing_holdout_is_not_zero(demo, tmp_path):
    spec = read_json(demo["manifest"])
    spec["images"] = [r for r in spec["images"] if r.get("split") != "holdout"]
    atomic_json(demo["manifest"], spec)
    from allbench.io import digest
    features = read_json(demo["features"])
    features["protocol_digest"] = digest(spec)
    features["images"].pop("ref-holdout")
    atomic_json(demo["features"], features)
    result = evaluate(demo["manifest"], demo["features"], tmp_path / "results")
    assert all(r["fidelity"]["holdout"]["value"] is None for r in read_json(result["report"])["groups"])


def test_result_refuses_different_feature_identity(demo, tmp_path):
    out = tmp_path / "results"
    evaluate(demo["manifest"], demo["features"], out)
    features = read_json(demo["features"])
    features["encoders"]["dino"]["weights_digest"] = "different-weights"
    atomic_json(demo["features"], features)
    with pytest.raises(ValueError, match="RESULT_INPUT_IDENTITY"):
        evaluate(demo["manifest"], demo["features"], out)


def test_damaged_checkpoint_rejected(demo, tmp_path):
    out = tmp_path / "results"
    with pytest.raises(RuntimeError):
        evaluate(demo["manifest"], demo["features"], out, True)
    checkpoint = read_json(out / "checkpoint.json")
    checkpoint["report"]["benchmark_id"] = "tampered"
    atomic_json(out / "checkpoint.json", checkpoint)
    with pytest.raises(ValueError, match="CHECKPOINT_CHECKSUM"):
        evaluate(demo["manifest"], demo["features"], out)


def test_duplicate_reference_weight_rejected(demo):
    spec = read_json(demo["manifest"])
    duplicate = dict(spec["images"][0], id="another-reference")
    spec["images"].append(duplicate)
    atomic_json(demo["manifest"], spec)
    with pytest.raises(ValueError, match="DUPLICATE_REFERENCE"):
        validate(demo["manifest"])


def test_missing_features_rejects_instead_of_reducing_denominator(demo, tmp_path):
    features = read_json(demo["features"])
    features["images"].pop(next(iter(features["images"])))
    atomic_json(demo["features"], features)
    with pytest.raises(ValueError, match="MEMBERS_MISMATCH"):
        evaluate(demo["manifest"], demo["features"], tmp_path / "results")


def test_vendi_is_per_prompt_not_between_prompts(demo, tmp_path):
    from pathlib import Path
    from allbench.features import save_arrays
    spec = read_json(demo["manifest"])
    features = read_json(demo["features"])
    root = Path(demo["features"]).parent
    # Both seeds collapse within every prompt, but different prompts span two modes.
    for i, r in enumerate(spec["images"]):
        with np.load(root / features["images"][r["id"]]["path"], allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        arrays["dino"] = np.array([1., 0., 0.]) if r.get("prompt_id") == "train" else np.array([0., 1., 0.])
        features["images"][r["id"]] = save_arrays(root, f"collapsed-{i}", arrays)
    atomic_json(demo["features"], features)
    result = evaluate(demo["manifest"], demo["features"], tmp_path / "results")
    assert all(r["diversity"]["value"] == pytest.approx(1.) for r in read_json(result["report"])["groups"])
