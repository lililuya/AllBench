# v0.1 验证记录

本页保留首次离线骨架验证；首轮真实底模接入见 [FLUX.2-klein-9B](flux2-klein-generator.md)，当前真实 LoRA 与全部编码器/报告的验收见[完整链路](full-pipeline-validation.md)。

日期：2026-10-27。这里只记录实际执行证据，不形成模型质量或吞吐结论。

## 本地 Windows

- Python 3.12，独立 `.venv`；依赖精确版本见 `requirements-test-lock.txt`。
- `python -m pytest -q --tb=short --basetemp E:/Codex_Data/AllBench/pytest-local-20261027-c`：32 项通过。
- 验证包括余弦/SCD/Vendi/Gram 的手算边界、零/非法特征、完整输出矩阵、跨划分字节/亲缘泄漏、参考重复、CLIP 语义替换、图片/特征/报告损坏、输入变化拒绝、缺编码器未测、锁不被抢占、检查点中断恢复和命令行子进程。
- 初次 pytest 默认临时目录权限失败，改用上述独立基目录后通过；未改系统临时目录权限。
- 独立合成 Demo：26 张图片、24 个 base/adapted 任务、6 类提示、2 个 seed、12 个指标分组，首次计算及再次重放成功。

## node94 真实预训练编码器

- 独立代码目录：`/net/node95/hd4/AllBench/20261027-v01/code`；运行和特征：同级 `runs/`，不在 Git。
- 只读使用既有 `/home/ubuntu/anaconda3/envs/comfy/bin/python`，没有向该环境安装或升级软件；pytest 依赖单独放在 `test-deps/`。
- 使用 `CUDA_VISIBLE_DEVICES=1`，实际空闲 GPU1 为 RTX 4090 24GB，进程内设备 `cuda:0`。未停止其他任务。
- 实际运行：Python 3.10.18、torch 2.8.0+cu128、torchvision 0.23.0+cu128、transformers 4.57.6、float32、batch=1。
- 既有 HF DINOv2-L snapshot：`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`；配置/权重文件聚合摘要 `e5b3a674cc978ae02da8c4aa29b276045c0d2448121ba3c7a68c8e60183d6d87`。
- 26/26 合成图片完成真实 DINOv2 编码，12/12 分组的保真与多样性代理指标生成。CLIP、VGG19 未配置，报告正确返回未测。
- 编码重放新增编码图片 0；对 DINO 特征执行提交前故障注入，从检查点恢复，指标重算 0，报告与首次完整运行逐字节一致。报告 SHA-256：`3828687bae33ac9641f8e775f869d6af1da5b59992b4bcb8110d3746261a8dc0`。
- 元数据汇总：`runs/verification-summary.json`；验证脚本：`scripts/verify_node94.py`。
- node94 使用独立 `test-deps` 运行同一完整回归：32 项通过，92.43 秒。该耗时包含网络盘合成文件读写，不是模型吞吐。

示例复核命令（既有代码目录中）：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src /home/ubuntu/anaconda3/envs/comfy/bin/python scripts/verify_node94.py /net/node95/hd4/AllBench/20261027-v01/runs --dino /home/ubuntu/.cache/huggingface/hub/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c
```

## 未验收部分

真实底模/LoRA 出图、目标概念成功率、CLIP/VGG19 预训练编码器 GPU 路径、风格 LoRA 参考风格相似度、人工阈值/一致性、置信区间、固定训练预算数据对照和正式生成吞吐均未验收。数值核心的合成测试不补齐这些证据。此版本不能宣称完整复现论文、验证真实数据质量或证明任何 LoRA/数据配方的收益。
