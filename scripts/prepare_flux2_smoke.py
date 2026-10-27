"""Prepare a harmless base-only smoke manifest using explicit existing local assets."""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from allbench.generate import component_identity
from allbench.io import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--transformer", required=True)
parser.add_argument("--text-encoder", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--vae", required=True)
parser.add_argument("--config-revision", required=True)
args = parser.parse_args()
from transformers import Qwen2TokenizerFast
tokenizer = Qwen2TokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
if not tokenizer.chat_template:
    raise ValueError("KLEIN_TOKENIZER_CHAT_TEMPLATE_REQUIRED: provide the complete official tokenizer directory")
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
config_root = out / "config-cache"
files = {}
for name in ("transformer/config.json", "scheduler/scheduler_config.json"):
    files[name] = hf_hub_download("black-forest-labs/FLUX.2-klein-9B", name,
                                  revision=args.config_revision, cache_dir=str(config_root))
config = {"transformer": args.transformer, "transformer_config": str(Path(files["transformer/config.json"]).parent),
          "text_encoder": args.text_encoder, "tokenizer": args.tokenizer, "vae": args.vae,
          "scheduler_config": files["scheduler/scheduler_config.json"],
          "is_distilled": True, "device": "cuda:0", "cpu_offload": True}
identity = component_identity(config)
import diffusers
texts = [("robot", "A small red wooden toy robot standing on a blue table, studio photography."),
         ("robot-garden", "A small red wooden toy robot standing in a sunny garden, watercolor painting."),
         ("forest", "A peaceful green forest with a winding path and warm morning light, landscape photography.")]
spec = {"schema_version": "1.0", "benchmark_id": "flux2-klein-9b-base-smoke-v1", "synthetic": True,
        "input_origin": "model_generated", "evaluation_mode": "base_only", "dataset_release": "no-training-data-base-smoke",
        "models": {"base": {"checkpoint": "black-forest-labs/FLUX.2-klein-9B", "revision": identity["transformer"]}},
        "training": {"unique_samples": None, "sample_exposures": None, "steps": None, "seed": None},
        "generation": {"engine": "diffusers.Flux2KleinPipeline", "engine_revision": diffusers.__version__,
                       "scheduler": "FlowMatchEulerDiscreteScheduler", "steps": 4, "cfg": 1.0,
                       "width": 512, "height": 512, "negative_prompt": "", "precision": "bfloat16",
                       "max_sequence_length": 512, "generator_device": "cpu", "is_distilled": True},
        "seeds": [7, 11], "concepts": [{"id": "smoke", "kind": "subject", "trigger": "<not-trained>"}],
        "prompts": [{"id": pid, "concept_id": "smoke", "kind": "agnostic", "text": text, "evaluation_text": text}
                    for pid, text in texts], "images": []}
atomic_json(out / "model-config.json", config)
atomic_json(out / "input.json", spec)
print({"manifest": str(out / "input.json"), "model_config": str(out / "model-config.json"),
       "jobs": 6, "components": identity, "config_revision": args.config_revision}, flush=True)
