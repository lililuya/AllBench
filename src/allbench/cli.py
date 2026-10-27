import argparse
import json
import sys

from .demo import make_demo
from .evaluate import evaluate
from .io import atomic_json
from .protocol import plan, validate


def main():
    parser = argparse.ArgumentParser(description="AllBench: 独立 LoRA 离线评测")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="生成合成流程夹具，不执行 LoRA")
    p.add_argument("--out", required=True)
    p = sub.add_parser("validate", help="校验完整图片矩阵")
    p.add_argument("manifest")
    p = sub.add_parser("plan", help="导出外部生成器任务")
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p = sub.add_parser("evaluate", help="按概念/提示类型计算代理指标")
    p.add_argument("manifest")
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interrupt-after-checkpoint", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("encode", help="显式本地预训练权重编码；不下载模型")
    p.add_argument("manifest")
    p.add_argument("--out", required=True)
    p.add_argument("--dino")
    p.add_argument("--clip")
    p.add_argument("--vgg")
    p.add_argument("--device", default="cpu")
    p.add_argument("--interrupt-after", type=int, help=argparse.SUPPRESS)
    p = sub.add_parser("generate", help="使用本地 FLUX.2-klein-9B 生成底模或 LoRA 配对矩阵")
    p.add_argument("manifest")
    p.add_argument("--model-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interrupt-after", type=int, help=argparse.SUPPRESS)
    for command, help_text in (("prepare", "冻结样本、权重摘要和统一运行配置"), ("run", "一条命令完成真实生成、编码、评测和报告")):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("config")
        p.add_argument("--out", required=True)
        if command == "run":
            p.add_argument("--interrupt-stage", choices=("generate", "encode", "evaluate"), help=argparse.SUPPRESS)
            p.add_argument("--interrupt-after", type=int, default=2, help=argparse.SUPPRESS)
    p = sub.add_parser("report", help="导出带图片配对的 HTML、CSV 和待审模板")
    p.add_argument("manifest")
    p.add_argument("--report", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("preflight", help="检查运行环境、tokenizer 和 CLIP 文本长度")
    p.add_argument("prepared")
    p.add_argument("--kind", choices=("generator", "encoder"), required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            result = make_demo(args.out)
        elif args.command == "validate":
            spec, identity = validate(args.manifest)
            result = {"valid": True, "protocol_digest": identity, "images": len(spec["images"])}
        elif args.command == "plan":
            result = plan(args.manifest)
            atomic_json(args.out, result)
            result = {"jobs": len(result["jobs"]), "out": args.out}
        elif args.command == "evaluate":
            result = evaluate(args.manifest, args.features, args.out, args.interrupt_after_checkpoint)
        elif args.command == "generate":
            from .generate import generate
            result = generate(args.manifest, args.model_config, args.out, args.interrupt_after)
        elif args.command == "report":
            from .report import export_report
            result = export_report(args.manifest, args.report, args.out)
        elif args.command == "preflight":
            from .preflight import preflight
            result = preflight(args.prepared, args.kind)
        elif args.command in ("prepare", "run"):
            from .pipeline import prepare, run
            result = (prepare(args.config, args.out) if args.command == "prepare" else
                      run(args.config, args.out, args.interrupt_stage, args.interrupt_after))
        else:
            from .encode import encode
            result = encode(args.manifest, args.out, args.dino, args.clip, args.vgg, args.device, args.interrupt_after)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
