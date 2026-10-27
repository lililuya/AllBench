from pathlib import Path

from PIL import Image
import pytest

from allbench.demo import make_demo
from allbench.generate import component_identity, generate
from allbench.io import atomic_json, read_json
from allbench.protocol import plan, validate


@pytest.fixture
def setup(tmp_path, monkeypatch):
    demo = make_demo(tmp_path / "input")
    spec = read_json(demo["manifest"])
    spec.update(evaluation_mode="base_only", input_origin="model_generated", images=[])
    spec["models"].pop("adapted")
    spec["generation"].update(engine="diffusers.Flux2KleinPipeline", engine_revision="test-1", precision="bfloat16",
                              scheduler="FlowMatchEulerDiscreteScheduler", max_sequence_length=512,
                              generator_device="cpu", is_distilled=True, cfg=1.0)
    config = {"is_distilled": True, "device": "cuda:0", "cpu_offload": True}
    for key in ("transformer", "transformer_config", "text_encoder", "tokenizer", "vae", "scheduler_config"):
        file = tmp_path / (key + ".json")
        atomic_json(file, {"component": key})
        config[key] = str(file)
    spec["models"]["base"]["revision"] = component_identity(config)["transformer"]
    atomic_json(demo["manifest"], spec)
    atomic_json(tmp_path / "config.json", config)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-1")
    return Path(demo["manifest"]), tmp_path / "config.json", tmp_path / "output"


class FakeBackend:
    calls = []
    runtime = {"diffusers": "test-1", "gpu": "synthetic-test"}

    def __init__(self, config):
        pass

    def generate(self, job, generation):
        self.calls.append(job)
        return Image.new("RGB", (generation["width"], generation["height"]), (job["seed"], 10, 30)), {}


def test_base_only_plan_and_no_phantom_adapted(setup):
    manifest, config, out = setup
    assert len(plan(manifest)["jobs"]) == 12
    with pytest.raises(ValueError, match="INCOMPLETE_OUTPUT"):
        validate(manifest)
    result = generate(manifest, config, out, backend_factory=FakeBackend)
    spec, _ = validate(result["manifest"])
    assert len(spec["images"]) == 12 and all(r["role"] == "base" for r in spec["images"])


def test_generation_recovers_without_replaying_completed_jobs(setup):
    manifest, config, out = setup
    FakeBackend.calls = []
    with pytest.raises(RuntimeError, match="INJECTED_GENERATION"):
        generate(manifest, config, out, interrupt_after=2, backend_factory=FakeBackend)
    assert len(FakeBackend.calls) == 2
    result = generate(manifest, config, out, backend_factory=FakeBackend)
    assert result["generated_this_attempt"] == 10 and result["reused"] == 2
    assert len(FakeBackend.calls) == 12
    def never_load(config):
        pytest.fail("Loaded backend on completed replay")
    assert generate(manifest, config, out, backend_factory=never_load)["generated_this_attempt"] == 0


def test_generation_refuses_changed_inputs(setup):
    manifest, config, out = setup
    generate(manifest, config, out, backend_factory=FakeBackend)
    spec = read_json(manifest)
    spec["generation"]["steps"] += 1
    atomic_json(manifest, spec)
    with pytest.raises(ValueError, match="GENERATION_INPUT_IDENTITY"):
        generate(manifest, config, out, backend_factory=FakeBackend)


def test_generation_refuses_damaged_output(setup):
    manifest, config, out = setup
    generate(manifest, config, out, backend_factory=FakeBackend)
    next(out.glob("*.png")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="GENERATED_IMAGE_CHECKSUM"):
        generate(manifest, config, out, backend_factory=FakeBackend)


def test_base_only_cannot_contain_adapter_or_adapted_images(setup):
    manifest, _, _ = setup
    spec = read_json(manifest)
    spec["models"]["base"]["adapter"] = {}
    atomic_json(manifest, spec)
    with pytest.raises(ValueError, match="BASE_MUST_NOT"):
        validate(manifest, require_outputs=False)


def test_base_only_report_missing_reference(setup):
    from allbench.evaluate import compute
    import numpy as np
    manifest, config, out = setup
    generated = generate(manifest, config, out, backend_factory=FakeBackend)
    spec = read_json(generated["manifest"])
    bundle = {"encoders": {"dino": {"synthetic": True}}}
    arrays = {"images": {r["id"]: {"dino": np.array([1., .1])} for r in spec["images"]}, "texts": {}}
    report = compute(spec, bundle, arrays)
    assert report["evaluation_mode"] == "base_only"
    assert all(r["fidelity"]["holdout"]["value"] is None for r in report["groups"])
    assert {r["role"] for r in report["groups"]} == {"base"}


def test_receipt_tampering_rejected_by_evaluation_input(setup):
    manifest, config, out = setup
    generated = generate(manifest, config, out, backend_factory=FakeBackend)
    spec = read_json(generated["manifest"])
    receipt = out / spec["images"][0]["generation_receipt"]
    value = read_json(receipt)
    value["job"]["seed"] += 1
    atomic_json(receipt, value)
    with pytest.raises(ValueError, match="RECEIPT_CHECKSUM"):
        validate(generated["manifest"])


def test_distilled_guidance_is_not_silently_ignored(setup):
    manifest, config, out = setup
    spec = read_json(manifest)
    spec["generation"]["cfg"] = 3.5
    atomic_json(manifest, spec)
    with pytest.raises(ValueError, match="GUIDANCE_MUST_BE_ONE"):
        generate(manifest, config, out, backend_factory=FakeBackend)


def test_tokenizer_chat_template_changes_invalidate_generation(setup, tmp_path):
    manifest, config_path, out = setup
    config = read_json(config_path)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    atomic_json(tokenizer / "tokenizer_config.json", {"model": "test"})
    template = tokenizer / "chat_template.jinja"
    template.write_text("first-template", encoding="utf-8")
    config["tokenizer"] = str(tokenizer)
    atomic_json(config_path, config)
    generate(manifest, config_path, out, backend_factory=FakeBackend)
    template.write_text("changed-template", encoding="utf-8")
    with pytest.raises(ValueError, match="GENERATION_INPUT_IDENTITY"):
        generate(manifest, config_path, out, backend_factory=FakeBackend)
