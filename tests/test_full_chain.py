import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from allbench.demo import make_demo
from allbench.encode import encode
from allbench.evaluate import evaluate
from allbench.generate import component_identity, generate
from allbench.io import atomic_json, file_hash, read_json
from allbench.metrics import reference_style
from allbench.pipeline import prepare, run
from allbench.report import export_report


class Encoders:
    calls = []

    def __init__(self, paths, device):
        self.keys = set(paths)

    def image(self, row, path):
        self.calls.append(row["id"])
        return {key: np.array([1., .2, .3]) for key in self.keys}

    def text(self, row):
        self.calls.append(row["id"])
        return {"clip": np.array([1., .1, .2])}


class Generator:
    calls = []
    runtime = {"diffusers": "test-1", "gpu": "synthetic-test", "adapter_load": {"weights_verified": True}}

    def __init__(self, config):
        assert config["adapter"]["revision"]

    def generate(self, job, generation):
        from PIL import Image
        self.calls.append(job)
        return Image.new("RGB", (generation["width"], generation["height"]), (job["seed"], job["role"] == "adapted", 0)), {
            "adapter_enabled": job["role"] == "adapted"}


@pytest.fixture
def case(tmp_path, monkeypatch):
    demo = make_demo(tmp_path / "case")
    spec = read_json(demo["manifest"])
    spec["images"] = [dict(r, path=str(Path(demo["manifest"]).parent / r["path"])) for r in spec["images"] if r["role"] == "reference"]
    for row in spec["images"]:
        row.pop("sha256")
    spec["generation"].update(engine="diffusers.Flux2KleinPipeline", engine_revision="test-1", precision="bfloat16",
                              scheduler="FlowMatchEulerDiscreteScheduler", max_sequence_length=512,
                              generator_device="cpu", is_distilled=True, cfg=1.)
    model = {"is_distilled": True, "device": "cuda:0", "cpu_offload": True}
    for key in ("transformer", "transformer_config", "text_encoder", "tokenizer", "vae", "scheduler_config"):
        path = tmp_path / (key + ".json")
        atomic_json(path, {"component": key})
        model[key] = str(path)
    for role in ("base", "adapted"):
        spec["models"][role].pop("revision")
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"synthetic-weight-test")
    spec["models"]["adapted"]["adapter"] = {"checkpoint": str(adapter), "method": "lora", "scale": .8}
    atomic_json(demo["manifest"], spec)
    atomic_json(tmp_path / "model.json", model)
    config = {"manifest": demo["manifest"], "model_config": str(tmp_path / "model.json"),
              "generator_python": sys.executable, "encoder_python": sys.executable,
              "encoders": {"dino": model["transformer"], "clip": model["text_encoder"], "vgg": model["vae"], "device": "cpu"},
              "cuda_visible_devices": "0", "require_all_metrics": True}
    atomic_json(tmp_path / "run.json", config)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-1")
    return tmp_path / "run.json", tmp_path / "out"


def encoder_args(case):
    config, root = case
    prep = prepare(config, root / "prepared")
    result = generate(prep["manifest"], prep["model_config"], root / "generated", backend_factory=Generator)
    return {"manifest": result["manifest"], "out": root / "features", "backend_factory": Encoders,
            "runtime_override": {"test_runtime": True}, **prep["config"]["encoders"]}


def test_all_cross_style_distance():
    assert reference_style([[0, 0], [2, 0]], [[1, 2]]) == pytest.approx(5)
    assert reference_style([[1, 2]], [[1, 2]]) == 0


def test_real_paired_contract_and_adapter_receipts(case):
    config, root = case
    prep = prepare(config, root / "prepared")
    Generator.calls = []
    with pytest.raises(RuntimeError, match="INJECTED_GENERATION"):
        generate(prep["manifest"], prep["model_config"], root / "generated", 13, Generator)
    result = generate(prep["manifest"], prep["model_config"], root / "generated", backend_factory=Generator)
    assert result["reused"] == 13 and result["generated_this_attempt"] == 11
    assert len(Generator.calls) == 24 and {r["role"] for r in Generator.calls} == {"base", "adapted"}
    spec = read_json(result["manifest"])
    assert "adapter" in spec["generator_provenance"]["components"]
    for row in spec["images"]:
        if row["role"] != "reference":
            receipt = read_json(root / "generated" / row["generation_receipt"])
            assert receipt["metrics"]["adapter_enabled"] == (row["role"] == "adapted")


@pytest.mark.parametrize("after", [3, 28])
def test_partial_image_and_text_encoding_recovery(case, after):
    args = encoder_args(case)
    Encoders.calls = []
    with pytest.raises(RuntimeError, match="INJECTED_ENCODING"):
        encode(**args, interrupt_after=after)
    assert len(Encoders.calls) == after
    resumed = encode(**args)
    assert resumed["encoded_images"] + resumed["encoded_texts"] == 32 - after
    assert len(Encoders.calls) == 32
    again = encode(**args)
    assert again["encoded_images"] == again["encoded_texts"] == 0
    assert len(Encoders.calls) == 32


def test_encoding_checkpoint_corruption_and_weight_change(case):
    args = encoder_args(case)
    with pytest.raises(RuntimeError):
        encode(**args, interrupt_after=1)
    file = next(Path(args["out"]).glob("*.npz"))
    file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="FEATURE_CHECKSUM"):
        encode(**args)
    Path(args["dino"]).write_text("changed")
    with pytest.raises(ValueError, match="ENCODING_IDENTITY"):
        encode(**args)


def test_prepared_input_rejects_changed_reference_and_config(case):
    config, root = case
    prep = prepare(config, root)
    before = file_hash(prep["manifest"])
    c = read_json(config)
    c["cuda_visible_devices"] = "1"
    atomic_json(config, c)
    with pytest.raises(ValueError, match="PIPELINE_INPUT_IDENTITY"):
        prepare(config, root)
    assert file_hash(prep["manifest"]) == before


def test_require_all_metrics_catches_missing_encoder(case):
    config, root = case
    c = read_json(config)
    c["encoders"].pop("clip")
    atomic_json(config, c)
    with pytest.raises(ValueError, match="ALL_THREE_ENCODERS"):
        prepare(config, root)


def test_unsupported_adapter_and_changed_weight_fail_before_generation(case):
    config, root = case
    prep = prepare(config, root / "prepared")
    adapter = Path(read_json(prep["manifest"])["models"]["adapted"]["adapter"]["checkpoint"])
    adapter.write_bytes(b"changed-after-freeze")
    with pytest.raises(ValueError, match="ADAPTER_IDENTITY_MISMATCH"):
        generate(prep["manifest"], prep["model_config"], root / "generated", backend_factory=Generator)
    c = read_json(config)
    spec = read_json(c["manifest"])
    spec["models"]["adapted"]["adapter"]["method"] = "lokr"
    atomic_json(c["manifest"], spec)
    with pytest.raises(ValueError, match="ONLY_STANDARD_LORA"):
        prepare(config, root / "other")


def test_loader_rejects_unsupported_keys_instead_of_filtering(monkeypatch):
    from types import SimpleNamespace
    from allbench.generate import load_adapter
    class Open:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def keys(self):
            return ["lora_unet.layer.dora_scale", "lora_unet.layer.lora_down.weight"]
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(safe_open=lambda *a, **k: Open()))
    with pytest.raises(ValueError, match="UNSUPPORTED_LORA_KEYS"):
        load_adapter(None, {"method": "lora", "checkpoint": "test-only.safetensors"})


def test_python_launcher_keeps_configured_path(case, tmp_path):
    config, root = case
    c = read_json(config)
    launcher = tmp_path / "venv-python.exe"
    try:
        launcher.symlink_to(sys.executable)
    except OSError:
        pytest.skip("Creating symlinks requires local Windows permission")
    c["generator_python"] = str(launcher)
    atomic_json(config, c)
    assert prepare(config, root)["config"]["generator_python"] == str(launcher)


def test_reports_are_bound_to_authoritative_input_and_escape_html(case):
    args = encoder_args(case)
    features = encode(**args)["features"]
    root = Path(args["out"]).parent
    report = evaluate(args["manifest"], features, root / "evaluation")["report"]
    exported = export_report(args["manifest"], report, root / "report")
    rows = list(csv.DictReader(Path(exported["csv"]).open(encoding="utf-8-sig", newline="")))
    assert rows and all(r["value"] for r in rows if r["status"] == "measured")
    assert all(r["style_fidelity"]["holdout"]["status"] == "measured" for r in read_json(report)["groups"])
    assert export_report(args["manifest"], report, root / "report")["reused"]
    assert all(r["concept_success"] is None for r in read_json(root / "report/review-template.json")["items"])
    Path(exported["html"]).write_text("corrupt")
    with pytest.raises(ValueError, match="EXPORT_CHECKSUM"):
        export_report(args["manifest"], report, root / "report")


def test_report_escapes_prompts_and_refuses_other_protocol(case):
    config, root = case
    spec_path = read_json(config)["manifest"]
    spec = read_json(spec_path)
    spec["prompts"][0]["text"] += '<script>alert("x")</script>'
    atomic_json(spec_path, spec)
    args = encoder_args(case)
    features = encode(**args)["features"]
    report = evaluate(args["manifest"], features, root / "evaluation")["report"]
    exported = export_report(args["manifest"], report, root / "report")
    html = Path(exported["html"]).read_text(encoding="utf-8")
    assert '<script>' not in html and '&lt;script&gt;' in html
    generated = read_json(args["manifest"])
    generated["input_origin"] = "a-different-protocol"
    atomic_json(args["manifest"], generated)
    with pytest.raises(ValueError, match="EXPORT_INPUT_MISMATCH"):
        export_report(args["manifest"], report, root / "other-report")


def fake_child(command, log, env):
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    stage = command[3]
    if stage == "preflight":
        return {"status": "ready", "synthetic_runtime_check": True}
    option = lambda name: command[command.index(name) + 1]
    out = option("--out")
    after = int(option("--interrupt-after")) if "--interrupt-after" in command else None
    if stage == "generate":
        return generate(command[4], option("--model-config"), out, after, Generator)
    if stage == "encode":
        return encode(command[4], out, option("--dino"), option("--clip"), option("--vgg"), "cpu", after,
                      Encoders, {"test_runtime": True})
    if stage == "evaluate":
        return evaluate(command[4], option("--features"), out, "--interrupt-after-checkpoint" in command)
    return export_report(command[4], option("--report"), out)


@pytest.mark.parametrize("stage", ["generate", "encode", "evaluate"])
def test_single_entry_failure_resume_and_zero_replay(case, stage):
    config, root = case
    with pytest.raises(RuntimeError, match="INJECTED"):
        run(config, root, stage, runner=fake_child)
    assert read_json(root / "pipeline.json")["status"] == "failed"
    assert not (root / ".lock").exists()
    result = run(config, root, runner=fake_child)
    assert result["status"] == "completed" and Path(result["report"]).is_file()
    replay = run(config, root, runner=fake_child)
    assert replay["stages"]["generate"]["generated_this_attempt"] == 0
    assert replay["stages"]["encode"]["encoded_images"] == replay["stages"]["encode"]["encoded_texts"] == 0
    assert not replay["stages"]["evaluate"]["computed"]
    assert replay["stages"]["report"]["reused"]
