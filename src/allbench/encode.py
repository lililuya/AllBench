"""Optional local-only pretrained encoders. No automatic model download."""
import platform
from pathlib import Path

from .features import load_bundle, save_arrays
from .io import atomic_json, digest, file_hash, read_json, require
from .protocol import validate


def weights_digest(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()
                                               and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model", ".jinja"})
    require(bool(files), "NO_LOCAL_MODEL_FILES")
    return digest([(p.name if path.is_file() else p.relative_to(path).as_posix(), file_hash(p)) for p in files])


def encode(manifest, out, dino=None, clip=None, vgg=None, device="cpu"):
    import numpy as np
    import torch
    import torchvision
    import transformers
    from PIL import Image, ImageOps
    from torchvision import transforms

    spec, protocol_id = validate(manifest)
    require(any((dino, clip, vgg)), "SELECT_AT_LEAST_ONE_LOCAL_ENCODER")
    paths = {k: p for k, p in (("dino", dino), ("clip", clip), ("style", vgg)) if p}
    encoders = {k: {"id": {"dino": "hf-dinov2-cls", "clip": "hf-clip-projection", "style": "torchvision-vgg19-gram"}[k],
                    "weights_digest": weights_digest(p), "synthetic": False,
                    "preprocess": {"dino": "224-bicubic;reference-letterbox-black;generated-stretch;imagenet-normalize",
                                   "clip": "local-CLIPProcessor;semantic-evaluation-text;reject-truncation",
                                   "style": "short-edge224-bilinear;imagenet-normalize;conv-0,5,10,19,28;Gram/(CHW)"}[k]}
                for k, p in paths.items()}
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "transformers": transformers.__version__,
               "device": str(device), "precision": "float32", "batch_size": 1}
    if str(device).startswith("cuda"):
        runtime["gpu"] = torch.cuda.get_device_name(torch.device(device))
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    with lock.open("x") as f:
        f.write(protocol_id)
    try:
        meta_path = root / "features.json"
        if meta_path.exists():
            old = read_json(meta_path)
            require(old["encoders"] == encoders and old.get("runtime") == runtime, "ENCODING_IDENTITY_MISMATCH")
            load_bundle(meta_path, spec, protocol_id)
            return {"features": str(meta_path), "resumed": True, "encoded_images": 0}
        models, processors = {}, {}
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if dino:
            from transformers import Dinov2Model
            models["dino"] = Dinov2Model.from_pretrained(dino, local_files_only=True).eval().to(device)
        if clip:
            from transformers import CLIPModel, CLIPProcessor
            models["clip"] = CLIPModel.from_pretrained(clip, local_files_only=True).eval().to(device)
            processors["clip"] = CLIPProcessor.from_pretrained(clip, local_files_only=True)
        if vgg:
            from torchvision.models import vgg19
            model = vgg19(weights=None)
            model.load_state_dict(torch.load(vgg, map_location="cpu", weights_only=True), strict=True)
            models["style"] = model.features[:29].eval().to(device)
            del model
        meta = {"schema_version": "1.0", "protocol_digest": protocol_id,
                "encoders": encoders, "runtime": runtime, "images": {}, "texts": {}}
        with torch.inference_mode():
            for index, row in enumerate(spec["images"]):
                image_path = (Path(manifest).resolve().parent / row["path"]).resolve()
                require(file_hash(image_path) == row["sha256"], "IMAGE_CHANGED_DURING_ENCODING")
                with Image.open(image_path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    arrays = {}
                    if dino:
                        sized = (ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
                                 if row["role"] == "reference" else im.resize((224, 224), Image.Resampling.BICUBIC))
                        result = models["dino"](pixel_values=normalize(sized).unsqueeze(0).to(device))
                        arrays["dino"] = result.last_hidden_state[0, 0].cpu().numpy()
                    if clip:
                        values = processors["clip"](images=im, return_tensors="pt").to(device)
                        arrays["clip"] = models["clip"].get_image_features(**values)[0].cpu().numpy()
                    if vgg:
                        x = normalize(transforms.Resize(224)(im)).unsqueeze(0).to(device)
                        grams = []
                        for layer_id, layer in enumerate(models["style"]):
                            x = layer(x)
                            if layer_id in (0, 5, 10, 19, 28):
                                _, c, h, w = x.shape
                                f = x.reshape(1, c, h * w)
                                grams.append((torch.bmm(f, f.transpose(1, 2)) / (c * h * w)).flatten().cpu().numpy())
                        arrays["style"] = np.concatenate(grams)
                    meta["images"][row["id"]] = save_arrays(root, f"image-{index:06}", arrays)
                print(f"encoded {index + 1}/{len(spec['images'])}", flush=True)
            if clip:
                for index, p in enumerate(spec["prompts"]):
                    processor = processors["clip"]
                    tokens = processor.tokenizer(p["evaluation_text"], truncation=False)["input_ids"]
                    require(len(tokens) <= models["clip"].config.text_config.max_position_embeddings, "CLIP_TEXT_TOO_LONG:" + p["id"])
                    text = processor(text=[p["evaluation_text"]], return_tensors="pt", padding=True, truncation=False).to(device)
                    feature = models["clip"].get_text_features(**text)[0].cpu().numpy()
                    meta["texts"][p["id"]] = save_arrays(root, f"text-{index:06}", {"clip": feature})
        # Revalidate files after inference before publishing their features.
        validate(manifest)
        atomic_json(meta_path, meta)
        load_bundle(meta_path, spec, protocol_id)
        return {"features": str(meta_path), "resumed": False, "encoded_images": len(spec["images"]), "runtime": runtime}
    finally:
        lock.unlink(missing_ok=True)
