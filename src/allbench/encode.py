"""Local pretrained encoders with checksummed per-item recovery."""
import platform
from pathlib import Path

from .features import load_bundle, load_record, save_arrays
from .io import atomic_json, digest, file_hash, read_json, require
from .protocol import validate


def weights_digest(path):
    path = Path(path).resolve()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()
                                               and p.suffix in {".json", ".safetensors", ".bin", ".pth", ".txt", ".model", ".jinja"})
    require(bool(files), "NO_LOCAL_MODEL_FILES")
    return digest([(p.name if path.is_file() else p.relative_to(path).as_posix(), file_hash(p)) for p in files])


def encoding_runtime(device):
    import torch
    import torchvision
    import transformers
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "transformers": transformers.__version__,
               "device": str(device), "precision": "float32", "batch_size": 1}
    if str(device).startswith("cuda"):
        runtime["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return runtime


class EncoderBackend:
    def __init__(self, paths, device):
        import torch
        from torchvision import transforms
        self.torch, self.transforms, self.device = torch, transforms, device
        self.models, self.processors = {}, {}
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if "dino" in paths:
            from transformers import Dinov2Model
            self.models["dino"] = Dinov2Model.from_pretrained(paths["dino"], local_files_only=True).eval().to(device)
        if "clip" in paths:
            from transformers import CLIPModel, CLIPProcessor
            self.models["clip"] = CLIPModel.from_pretrained(paths["clip"], local_files_only=True).eval().to(device)
            self.processors["clip"] = CLIPProcessor.from_pretrained(paths["clip"], local_files_only=True)
        if "style" in paths:
            from torchvision.models import vgg19
            model = vgg19(weights=None)
            model.load_state_dict(torch.load(paths["style"], map_location="cpu", weights_only=True), strict=True)
            self.models["style"] = model.features[:29].eval().to(device)

    def image(self, row, path):
        import numpy as np
        from PIL import Image, ImageOps
        torch, device = self.torch, self.device
        with torch.inference_mode(), Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert("RGB")
            arrays = {}
            if "dino" in self.models:
                sized = (ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
                         if row["role"] == "reference" else im.resize((224, 224), Image.Resampling.BICUBIC))
                result = self.models["dino"](pixel_values=self.normalize(sized).unsqueeze(0).to(device))
                arrays["dino"] = result.last_hidden_state[0, 0].cpu().numpy()
            if "clip" in self.models:
                values = self.processors["clip"](images=im, return_tensors="pt").to(device)
                arrays["clip"] = self.models["clip"].get_image_features(**values)[0].cpu().numpy()
            if "style" in self.models:
                x = self.normalize(self.transforms.Resize(224)(im)).unsqueeze(0).to(device)
                grams = []
                for layer_id, layer in enumerate(self.models["style"]):
                    x = layer(x)
                    if layer_id in (0, 5, 10, 19, 28):
                        _, c, h, w = x.shape
                        f = x.reshape(1, c, h * w)
                        grams.append((torch.bmm(f, f.transpose(1, 2)) / (c * h * w)).flatten().cpu().numpy())
                arrays["style"] = np.concatenate(grams)
            return arrays

    def text(self, prompt):
        processor, model = self.processors["clip"], self.models["clip"]
        tokens = processor.tokenizer(prompt["evaluation_text"], truncation=False)["input_ids"]
        require(len(tokens) <= model.config.text_config.max_position_embeddings, "CLIP_TEXT_TOO_LONG:" + prompt["id"])
        text = processor(text=[prompt["evaluation_text"]], return_tensors="pt", padding=True, truncation=False).to(self.device)
        with self.torch.inference_mode():
            return {"clip": model.get_text_features(**text)[0].cpu().numpy()}


def encode(manifest, out, dino=None, clip=None, vgg=None, device="cpu", interrupt_after=None,
           backend_factory=None, runtime_override=None):
    spec, protocol_id = validate(manifest)
    require(any((dino, clip, vgg)), "SELECT_AT_LEAST_ONE_LOCAL_ENCODER")
    paths = {k: p for k, p in (("dino", dino), ("clip", clip), ("style", vgg)) if p}
    encoders = {k: {"id": {"dino": "hf-dinov2-cls", "clip": "hf-clip-projection", "style": "torchvision-vgg19-gram"}[k],
                    "weights_digest": weights_digest(p), "synthetic": False,
                    "preprocess": {"dino": "224-bicubic;reference-letterbox-black;generated-stretch;imagenet-normalize",
                                   "clip": "local-CLIPProcessor;semantic-evaluation-text;reject-truncation",
                                   "style": "short-edge224-bilinear;imagenet-normalize;conv-0,5,10,19,28;Gram/(CHW)"}[k]}
                for k, p in paths.items()}
    runtime = runtime_override if runtime_override is not None else encoding_runtime(device)
    implementation = digest([(p.name, file_hash(p)) for p in sorted(Path(__file__).parent.glob("*.py"))])
    identity = digest({"protocol": protocol_id, "encoders": encoders, "runtime": runtime, "implementation": implementation})
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".lock"
    with lock.open("x") as f:
        f.write(identity)
    try:
        meta_path, checkpoint = root / "features.json", root / "checkpoint.json"
        if meta_path.exists():
            old = read_json(meta_path)
            require(old.get("identity") == identity, "ENCODING_IDENTITY_MISMATCH")
            load_bundle(meta_path, spec, protocol_id)
            return {"features": str(meta_path), "resumed": True, "encoded_images": 0, "encoded_texts": 0}
        if checkpoint.exists():
            saved = read_json(checkpoint)
            require(saved["identity"] == identity, "ENCODING_IDENTITY_MISMATCH")
            meta = saved["meta"]
            require(digest(meta) == saved["digest"], "ENCODING_CHECKPOINT_CHECKSUM_MISMATCH")
        else:
            meta = {"schema_version": "1.0", "protocol_digest": protocol_id, "identity": identity,
                    "encoders": encoders, "runtime": runtime, "images": {}, "texts": {}}
        expected = {"images": {r["id"] for r in spec["images"]},
                    "texts": {p["id"] for p in spec["prompts"]} if clip else set()}
        dims = {}
        for kind in ("images", "texts"):
            require(set(meta[kind]) <= expected[kind], "ENCODING_CHECKPOINT_MEMBERS")
            for record in meta[kind].values():
                load_record(root, record, set(encoders) if kind == "images" else {"clip"}, dims)
        backend, counts = None, {"images": 0, "texts": 0}
        for kind, rows in (("images", spec["images"]), ("texts", spec["prompts"] if clip else [])):
            for index, row in enumerate(rows):
                if row["id"] in meta[kind]:
                    continue
                if backend is None:
                    backend = (backend_factory or EncoderBackend)(paths, device)
                if kind == "images":
                    path = (Path(manifest).resolve().parent / row["path"]).resolve()
                    require(file_hash(path) == row["sha256"], "IMAGE_CHANGED_DURING_ENCODING")
                    arrays = backend.image(row, path)
                else:
                    arrays = backend.text(row)
                record = save_arrays(root, f"{kind}-{index:06}", arrays)
                load_record(root, record, set(encoders) if kind == "images" else {"clip"}, dims)
                meta[kind][row["id"]] = record
                atomic_json(checkpoint, {"identity": identity, "meta": meta, "digest": digest(meta)})
                counts[kind] += 1
                print(f"encoded {kind} {len(meta[kind])}/{len(rows)}", flush=True)
                if interrupt_after is not None and sum(counts.values()) >= interrupt_after:
                    raise RuntimeError("INJECTED_ENCODING_INTERRUPTION")
        validate(manifest)
        atomic_json(meta_path, meta)
        load_bundle(meta_path, spec, protocol_id)
        return {"features": str(meta_path), "resumed": sum(counts.values()) < sum(map(len, expected.values())),
                "encoded_images": counts["images"], "encoded_texts": counts["texts"], "runtime": runtime}
    finally:
        lock.unlink(missing_ok=True)
