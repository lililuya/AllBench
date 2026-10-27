"""Cheap runtime/token checks before a long generation batch; no model weights loaded."""
from .io import read_json, require


def preflight(prepared_path, kind):
    frozen = read_json(prepared_path)["config"]
    spec = frozen["manifest"]
    if kind == "generator":
        import diffusers
        import peft
        import torch
        from diffusers import Flux2KleinPipeline
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        require(diffusers.__version__ == spec["generation"]["engine_revision"], "GENERATOR_ENGINE_VERSION_MISMATCH")
        tokenizer = Qwen2TokenizerFast.from_pretrained(frozen["model_config"]["tokenizer"], local_files_only=True)
        require(bool(tokenizer.chat_template), "KLEIN_TOKENIZER_CHAT_TEMPLATE_REQUIRED")
        require(torch.cuda.is_available(), "GENERATOR_CUDA_UNAVAILABLE")
        return {"status": "ready", "kind": kind, "diffusers": diffusers.__version__, "peft": peft.__version__,
                "gpu": torch.cuda.get_device_name(frozen["model_config"]["device"])}
    import torch
    import torchvision
    import transformers
    from transformers import CLIPConfig, CLIPProcessor, Dinov2Model
    if "clip" in frozen["encoders"]:
        path = frozen["encoders"]["clip"]
        processor = CLIPProcessor.from_pretrained(path, local_files_only=True)
        config = CLIPConfig.from_pretrained(path, local_files_only=True)
        for prompt in spec["prompts"]:
            tokens = processor.tokenizer(prompt["evaluation_text"], truncation=False)["input_ids"]
            require(len(tokens) <= config.text_config.max_position_embeddings, "CLIP_TEXT_TOO_LONG:" + prompt["id"])
    device = frozen["encoders"]["device"]
    if device.startswith("cuda"):
        require(torch.cuda.is_available(), "ENCODER_CUDA_UNAVAILABLE")
        torch.cuda.get_device_name(device)
    return {"status": "ready", "kind": kind, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "transformers": transformers.__version__}
