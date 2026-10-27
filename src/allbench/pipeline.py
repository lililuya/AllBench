"""One entry point; generator and encoders keep their own Python environments."""
import copy
import json
import os
import subprocess
from pathlib import Path

from .encode import weights_digest
from .generate import component_identity
from .io import atomic_json, digest, file_hash, read_json, require
from .protocol import validate


def resolve(parent, value):
    return str((Path(parent) / value).resolve())


def prepare(config_path, out):
    config_path, root = Path(config_path).resolve(), Path(out).resolve()
    config = read_json(config_path)
    require(set(config) == {"manifest", "model_config", "generator_python", "encoder_python", "encoders",
                            "cuda_visible_devices", "require_all_metrics"}, "PIPELINE_CONFIG_KEYS")
    require(type(config["require_all_metrics"]) is bool, "REQUIRE_ALL_METRICS_BOOLEAN")
    require(isinstance(config["cuda_visible_devices"], str) and config["cuda_visible_devices"], "CUDA_VISIBLE_DEVICES_REQUIRED")
    manifest = Path(resolve(config_path.parent, config["manifest"]))
    model_path = Path(resolve(config_path.parent, config["model_config"]))
    model = read_json(model_path)
    for key in ("transformer", "transformer_config", "text_encoder", "tokenizer", "vae", "scheduler_config"):
        model[key] = resolve(model_path.parent, model[key])
    components = component_identity(model)
    spec = copy.deepcopy(read_json(manifest))
    base = spec["models"]["base"]
    base.setdefault("revision", components["transformer"])
    require(base["revision"] == components["transformer"], "PREPARE_BASE_REVISION_MISMATCH")
    if spec.get("evaluation_mode", "paired") == "paired":
        adapted = spec["models"]["adapted"]
        adapted.setdefault("checkpoint", base["checkpoint"])
        adapted.setdefault("revision", base["revision"])
        adapter = adapted["adapter"]
        adapter["checkpoint"] = resolve(manifest.parent, adapter["checkpoint"])
        require(adapter["method"].lower() == "lora", "KLEIN_ONLY_STANDARD_LORA_SUPPORTED")
        adapter_hash = weights_digest(adapter["checkpoint"])
        adapter.setdefault("revision", adapter_hash)
        require(adapter["revision"] == adapter_hash, "PREPARE_ADAPTER_REVISION_MISMATCH")
        components["adapter"] = adapter_hash
    for row in spec["images"]:
        require(row["role"] == "reference", "PREPARE_REFERENCES_ONLY")
        row["path"] = resolve(manifest.parent, row["path"])
        sha = file_hash(row["path"])
        row.setdefault("sha256", sha)
        require(row["sha256"] == sha, "PREPARE_REFERENCE_CHECKSUM_MISMATCH")
    runtime = {}
    for key in ("generator_python", "encoder_python"):
        # Preserve venv launcher symlinks: resolving them may escape pyvenv.cfg.
        runtime[key] = os.path.abspath(config_path.parent / config[key])
        require(Path(runtime[key]).is_file(), "PYTHON_EXECUTABLE_MISSING:" + key)
    encoders = dict(config["encoders"])
    require(set(encoders) <= {"dino", "clip", "vgg", "device"} and "device" in encoders, "ENCODER_CONFIG_KEYS")
    require(set(encoders) - {"device"}, "SELECT_AT_LEAST_ONE_LOCAL_ENCODER")
    if config["require_all_metrics"]:
        require(set(encoders) == {"dino", "clip", "vgg", "device"}, "ALL_THREE_ENCODERS_REQUIRED")
        require(len(spec["seeds"]) >= 2, "ALL_METRICS_REQUIRE_TWO_SEEDS")
        for concept in spec["concepts"]:
            require({r.get("split") for r in spec["images"] if r["concept_id"] == concept["id"]} == {"train", "holdout"},
                    "ALL_METRICS_REQUIRE_TRAIN_AND_HOLDOUT")
    encoder_hashes = {}
    for key in set(encoders) - {"device"}:
        encoders[key] = resolve(config_path.parent, encoders[key])
        encoder_hashes[key] = weights_digest(encoders[key])
    implementation = digest([(p.name, file_hash(p)) for p in sorted(Path(__file__).parent.glob("*.py"))])
    frozen = {"manifest": spec, "model_config": model, **runtime, "encoders": encoders,
              "cuda_visible_devices": config["cuda_visible_devices"], "require_all_metrics": config["require_all_metrics"],
              "components": components, "encoder_hashes": encoder_hashes, "implementation": implementation}
    identity = digest(frozen)
    receipt = root / "prepared.json"
    if receipt.exists():
        require(read_json(receipt)["identity"] == identity, "PIPELINE_INPUT_IDENTITY_MISMATCH_USE_NEW_DIRECTORY")
    root.mkdir(parents=True, exist_ok=True)
    # Validate via a candidate before replacing any committed prepared input.
    candidate = root / "input.candidate.json"
    atomic_json(candidate, spec)
    try:
        validate(candidate, require_outputs=False)
    finally:
        candidate.unlink(missing_ok=True)
    atomic_json(root / "input.json", spec)
    atomic_json(root / "model-config.json", model)
    atomic_json(receipt, {"identity": identity, "config": frozen})
    return {"identity": identity, "manifest": str(root / "input.json"),
            "model_config": str(root / "model-config.json"), "config": frozen}


def child(command, log, env):
    with Path(log).open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, shell=False)
    lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    if result.returncode:
        raise RuntimeError("STAGE_FAILED:" + str(result.returncode) + ":" + str(log) + "\n" + "\n".join(lines[-15:]))
    for line in reversed(lines):
        try:
            summary = json.loads(line)
            if isinstance(summary, dict):
                return summary
        except json.JSONDecodeError:
            pass
    raise RuntimeError("STAGE_SUMMARY_MISSING:" + str(log))


def run(config_path, out, interrupt_stage=None, interrupt_after=2, runner=None):
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        prepared = prepare(config_path, root / "prepared")
        frozen, identity = prepared["config"], prepared["identity"]
        journal = root / "pipeline.json"
        old = read_json(journal) if journal.exists() else {"identity": identity, "attempts": 0, "stages": {}}
        require(old["identity"] == identity, "PIPELINE_INPUT_IDENTITY_MISMATCH")
        old.update(status="running", attempts=old["attempts"] + 1)
        atomic_json(journal, old)
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=frozen["cuda_visible_devices"], PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        gen_python, enc_python = frozen["generator_python"], frozen["encoder_python"]
        cli = ["-m", "allbench.cli"]
        manifest = str(root / "generated/benchmark.json")
        features = str(root / "features/features.json")
        commands = [
            ("check-generator", [gen_python, *cli, "preflight", str(root / "prepared/prepared.json"), "--kind", "generator"]),
            ("check-encoder", [enc_python, *cli, "preflight", str(root / "prepared/prepared.json"), "--kind", "encoder"]),
            ("generate", [gen_python, *cli, "generate", prepared["manifest"], "--model-config", prepared["model_config"], "--out", str(root / "generated")]),
            ("encode", [enc_python, *cli, "encode", manifest, "--out", str(root / "features")]),
            ("evaluate", [enc_python, *cli, "evaluate", manifest, "--features", features, "--out", str(root / "evaluation")]),
            ("report", [enc_python, *cli, "report", manifest, "--report", str(root / "evaluation/report.json"), "--out", str(root / "report")])]
        for key, value in frozen["encoders"].items():
            commands[3][1].extend(["--" + key, value])
        for stage, command in commands:
            if stage == interrupt_stage:
                command.extend(["--interrupt-after-checkpoint"] if stage == "evaluate" else ["--interrupt-after", str(interrupt_after)])
            old.update(current_stage=stage)
            atomic_json(journal, old)
            print("pipeline " + stage, flush=True)
            try:
                summary = (runner or child)(command, logs / f"{old['attempts']:03}-{stage}.log", env)
            except Exception as exc:
                old.update(status="failed", error=str(exc))
                atomic_json(journal, old)
                raise
            old["stages"][stage] = summary
            atomic_json(journal, old)
        old.update(status="completed", current_stage=None)
        old.pop("error", None)
        atomic_json(journal, old)
        return {"status": "completed", "identity": identity, "stages": old["stages"],
                "report": str(root / "report/index.html"), "model_effect_validated": False}
    finally:
        lock.unlink(missing_ok=True)
