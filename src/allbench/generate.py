"""Local FLUX.2-klein generation with immutable per-image receipts."""
import os
import time
from pathlib import Path

from PIL import Image

from .encode import weights_digest
from .io import atomic_json, digest, file_hash, read_json, require
from .protocol import plan, validate


def component_identity(config):
    keys = ("transformer", "transformer_config", "text_encoder", "tokenizer", "vae", "scheduler_config")
    require(set(config) == set(keys) | {"is_distilled", "device", "cpu_offload"}, "GENERATOR_CONFIG_KEYS")
    require(type(config["is_distilled"]) is bool and type(config["cpu_offload"]) is bool, "GENERATOR_CONFIG_FLAGS")
    require(isinstance(config["device"], str) and config["device"].startswith("cuda:"), "GENERATOR_CUDA_REQUIRED")
    return {k: weights_digest(config[k]) for k in keys}


class KleinBackend:
    def __init__(self, config):
        import diffusers
        import torch
        import transformers
        from diffusers import (AutoencoderKLFlux2, Flux2KleinPipeline,
                               Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler)
        from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast

        self.torch = torch
        self.device = config["device"]
        self.runtime = {"diffusers": diffusers.__version__, "transformers": transformers.__version__,
                        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(self.device),
                        "device": self.device, "precision": "bfloat16", "cpu_offload": config["cpu_offload"]}
        transformer = Flux2Transformer2DModel.from_single_file(
            config["transformer"], config=config["transformer_config"],
            torch_dtype=torch.bfloat16, local_files_only=True)
        vae = AutoencoderKLFlux2.from_pretrained(config["vae"], torch_dtype=torch.bfloat16, local_files_only=True)
        text_encoder = Qwen3ForCausalLM.from_pretrained(config["text_encoder"], torch_dtype=torch.bfloat16, local_files_only=True)
        tokenizer = Qwen2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(read_json(config["scheduler_config"]))
        self.pipe = Flux2KleinPipeline(transformer=transformer, vae=vae, text_encoder=text_encoder,
                                       tokenizer=tokenizer, scheduler=scheduler, is_distilled=config["is_distilled"])
        if config["cpu_offload"]:
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, job, generation):
        self.torch.cuda.reset_peak_memory_stats(self.device)
        image = self.pipe(prompt=job["text"], height=generation["height"], width=generation["width"],
                          num_inference_steps=generation["steps"], guidance_scale=generation["cfg"],
                          max_sequence_length=generation["max_sequence_length"],
                          generator=self.torch.Generator(device="cpu").manual_seed(job["seed"])).images[0]
        self.torch.cuda.synchronize(self.device)
        return image, {"peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device)}


def generate(manifest, model_config, out, interrupt_after=None, backend_factory=None):
    spec, _ = validate(manifest, require_outputs=False)
    require(spec.get("evaluation_mode") == "base_only", "KLEIN_GENERATOR_BASE_ONLY_CURRENTLY_SUPPORTED")
    require(all(r["role"] == "reference" for r in spec["images"]), "GENERATOR_REQUIRES_INPUT_WITHOUT_OUTPUTS")
    g = spec["generation"]
    require(g["engine"] == "diffusers.Flux2KleinPipeline" and g["precision"] == "bfloat16", "KLEIN_ENGINE_PRECISION")
    require(g["scheduler"] == "FlowMatchEulerDiscreteScheduler" and g["negative_prompt"] == "", "KLEIN_SAMPLER_CONFIG")
    require(g["width"] % 16 == 0 and g["height"] % 16 == 0, "KLEIN_SIZE_ALIGNMENT")
    require(type(g.get("max_sequence_length")) is int and 1 <= g["max_sequence_length"] <= 512, "KLEIN_MAX_SEQUENCE")
    config = read_json(model_config)
    components = component_identity(config)
    require(spec["models"]["base"]["revision"] == components["transformer"], "GENERATOR_MODEL_IDENTITY_MISMATCH")
    require(g.get("generator_device") == "cpu", "GENERATOR_RNG_DEVICE_MUST_BE_CPU")
    require(g.get("is_distilled") == config["is_distilled"], "GENERATOR_DISTILLATION_MISMATCH")
    require(not config["is_distilled"] or g["cfg"] == 1.0, "DISTILLED_KLEIN_GUIDANCE_MUST_BE_ONE")
    import importlib.metadata
    runtime_identity = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "transformers")}
    require(g["engine_revision"] == runtime_identity["diffusers"], "GENERATOR_ENGINE_VERSION_MISMATCH")
    implementation = digest([(p.name, file_hash(p)) for p in sorted(Path(__file__).parent.glob("*.py"))])
    identity = digest({"spec": spec, "components": components, "config": config,
                       "runtime": runtime_identity, "implementation": implementation})
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    try:
        identity_path = root / "identity.json"
        if identity_path.exists():
            require(read_json(identity_path)["identity"] == identity, "GENERATION_INPUT_IDENTITY_MISMATCH")
        else:
            atomic_json(identity_path, {"identity": identity, "components": components, "runtime": runtime_identity})
        jobs = plan(manifest)["jobs"]
        completed = []
        pending = []
        for job in jobs:
            key = digest(job)
            receipt_path = root / (key + ".json")
            image_path = root / (key + ".png")
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                require(receipt["identity"] == identity and receipt["job"] == job, "GENERATION_RECEIPT_IDENTITY_MISMATCH")
                require(image_path.is_file() and file_hash(image_path) == receipt["sha256"], "GENERATED_IMAGE_CHECKSUM_MISMATCH")
                completed.append(receipt)
            else:
                pending.append((job, key, receipt_path, image_path))
        backend = None
        generated = 0
        load_seconds = 0.0
        for job, key, receipt_path, image_path in pending:
            if backend is None:
                start = time.perf_counter()
                backend = (backend_factory or KleinBackend)(config)
                load_seconds = time.perf_counter() - start
                require(backend.runtime["diffusers"] == g["engine_revision"], "BACKEND_VERSION_MISMATCH")
                require(all(r["runtime"] == backend.runtime for r in completed), "GENERATION_RESUME_RUNTIME_MISMATCH")
            start = time.perf_counter()
            image, metrics = backend.generate(job, g)
            require(isinstance(image, Image.Image) and image.size == (g["width"], g["height"]), "GENERATED_IMAGE_SIZE_MISMATCH")
            tmp = root / (key + ".tmp.png")
            with tmp.open("wb") as f:
                image.save(f, format="PNG")
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(image_path)
            receipt = {"identity": identity, "job": job, "path": image_path.name,
                       "sha256": file_hash(image_path), "seconds": time.perf_counter() - start,
                       "runtime": backend.runtime, "metrics": metrics, "components": components}
            atomic_json(receipt_path, receipt)
            completed.append(receipt)
            generated += 1
            print(f"generated {len(completed)}/{len(jobs)}", flush=True)
            if interrupt_after is not None and generated >= interrupt_after:
                raise RuntimeError("INJECTED_GENERATION_INTERRUPTION")
        final = dict(spec)
        final["generator_provenance"] = {"identity": identity, "components": components,
                                         "runtime": completed[0]["runtime"]}
        final["images"] = [dict(r, path=str((Path(manifest).resolve().parent / r["path"]).resolve())) for r in spec["images"]]
        for receipt in sorted(completed, key=lambda r: digest(r["job"])):
            job = receipt["job"]
            final["images"].append({"id": digest(job), "role": job["role"], "concept_id": job["concept_id"],
                                    "prompt_id": job["prompt_id"], "seed": job["seed"],
                                    "path": receipt["path"], "sha256": receipt["sha256"],
                                    "generation_receipt": digest(job) + ".json",
                                    "receipt_sha256": file_hash(root / (digest(job) + ".json"))})
        atomic_json(root / "benchmark.json", final)
        validate(root / "benchmark.json")
        summary = {"identity": identity, "manifest": str(root / "benchmark.json"), "expected": len(jobs),
                   "generated_this_attempt": generated, "reused": len(jobs) - generated,
                   "load_seconds_this_attempt": load_seconds,
                   "generation_seconds_total": sum(r["seconds"] for r in completed),
                   "model_effect_validated": False, "evaluation_mode": "base_only"}
        atomic_json(root / "generation-summary.json", summary)
        return summary
    finally:
        lock.unlink(missing_ok=True)
