"""Readable derivatives of the authoritative JSON; never invent success labels."""
import csv
import html
import io
import os
from pathlib import Path

from .io import atomic_json, digest, file_hash, read_json, require
from .protocol import validate


def metric_rows(report):
    for group in report["groups"]:
        prefix = {k: group[k] for k in ("concept_id", "prompt_kind", "role", "images", "prompts")}
        for metric in ("fidelity", "style_fidelity", "text_alignment", "diversity", "base_style_drift"):
            splits = group[metric] if metric in ("fidelity", "style_fidelity") else {"": group[metric]}
            for split, result in splits.items():
                values = result["value"] if isinstance(result["value"], dict) else {metric: result["value"]}
                for component, value in values.items():
                    yield {**prefix, "metric": metric, "component": component, "split": split,
                           "status": result["status"], "value": value, "reason": result.get("reason", "")}


def export_report(manifest, report_path, out):
    spec, protocol_id = validate(manifest)
    report = read_json(report_path)
    require(report.get("protocol_digest") == protocol_id and report["benchmark_id"] == spec["benchmark_id"] and report["protocol"] == {
        k: spec[k] for k in ("dataset_release", "models", "generation", "training", "seeds")}, "EXPORT_INPUT_MISMATCH")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = digest({"protocol": protocol_id, "report": file_hash(report_path), "format": "html-csv-v1"})
    receipt = root / "export.json"
    if receipt.exists():
        old = read_json(receipt)
        require(old["identity"] == identity, "EXPORT_IDENTITY_MISMATCH")
        require(all(file_hash(root / name) == sha for name, sha in old["files"].items()), "EXPORT_CHECKSUM_MISMATCH")
        return {"reused": True, "html": str(root / "index.html"), "csv": str(root / "metrics.csv")}
    rows = list(metric_rows(report))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_path = root / "metrics.csv"
    csv_tmp = root / "metrics.csv.tmp"
    csv_tmp.write_text(stream.getvalue(), encoding="utf-8-sig")
    csv_tmp.replace(csv_path)
    e = lambda v: html.escape(str(v), quote=True)
    table = "".join("<tr>" + "".join(f"<td>{e(v)}</td>" for v in r.values()) + "</tr>" for r in rows)
    header = "".join(f"<th>{e(k)}</th>" for k in rows[0])
    cards = []
    prompts = {p["id"]: p for p in spec["prompts"]}
    generated = [r for r in spec["images"] if r["role"] != "reference"]
    for row in sorted(generated, key=lambda r: (r["prompt_id"], r["seed"], r["role"])):
        path = (Path(manifest).resolve().parent / row["path"]).resolve()
        src = Path(os.path.relpath(path, root)).as_posix()
        cards.append(f'<figure><img loading="lazy" src="{e(src)}"><figcaption>{e(row["role"])} / '
                     f'{e(row["prompt_id"])} / seed {row["seed"]}<br>{e(prompts[row["prompt_id"]]["text"])}</figcaption></figure>')
    coverage = "".join(f"<li>{e(concept)}: {e(counts)}</li>" for concept, counts in report["prompt_coverage"].items())
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>AllBench {e(report["benchmark_id"])}</title>
<style>body{{font:15px system-ui;margin:28px;color:#182331;background:#f7f8fa}}table{{border-collapse:collapse;background:white}}td,th{{padding:7px;border:1px solid #ccd}}th{{position:sticky;top:0;background:#e4eaf3}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}figure{{margin:0;background:white;padding:10px}}img{{width:100%;height:260px;object-fit:contain}}figcaption{{overflow-wrap:anywhere}}.scroll{{overflow:auto}}</style>
<h1>AllBench · {e(report["benchmark_id"])}</h1><p>数据版本：{e(spec["dataset_release"])}；模式：{e(report["evaluation_mode"])}。</p>
<p>真实概念效果：待人工验收。参考夹具/合成标志：{e(report["synthetic_images"])}；人工向量：{e(report["synthetic_features"])}。</p>
<p>流程成功仅证明模块执行完成。余弦/CLIP 越高越相似，SCD/Gram 距离越低越相似，Vendi 表示多样性；无统一总分或通过阈值。</p>
<p>权威 JSON SHA256：{file_hash(report_path)}。<a href="metrics.csv">下载 CSV</a></p>
<h2>提示覆盖</h2><ul>{coverage}</ul><h2>指标（未测保留原因）</h2><div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{table}</tbody></table></div>
<h2>同提示/种子对照</h2><div class="gallery">{"".join(cards)}</div></html>'''
    temp = root / "index.html.tmp"
    temp.write_text(content, encoding="utf-8")
    temp.replace(root / "index.html")
    # Labels deliberately empty; reviewers may organize independent gold outside Git.
    atomic_json(root / "review-template.json", {"protocol_digest": protocol_id, "dataset_release": spec["dataset_release"],
        "status": "pending_human_review", "items": [{"image_id": r["id"], "concept_id": r["concept_id"],
            "prompt_id": r["prompt_id"], "seed": r["seed"], "concept_success": None,
            "prompt_success": None, "copy_suspected": None, "failure_tags": [], "reviewer": None} for r in generated]})
    files = {name: file_hash(root / name) for name in ("index.html", "metrics.csv", "review-template.json")}
    atomic_json(receipt, {"identity": identity, "files": files})
    return {"reused": False, "html": str(root / "index.html"), "csv": str(csv_path)}
