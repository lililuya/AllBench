# FLUX.2-klein-9B 真实生成器

2026-10-27，用户指定该模型作为真实生成器。此文记录独立生成与评测链路；真实 LoRA 概念/数据收益仍需对应权重、参考样本与人工金标。

本页保留首轮 base_only 历史证据；后续标准 LoRA、三个真实编码器和统一入口的实际验收见[完整链路记录](full-pipeline-validation.md)。

## 固定模型和组件

- 官方 `black-forest-labs/FLUX.2-klein-9B` 蒸馏版；HF 配置修订 `92196c8e11f7b6cf2b7493e037d8c5345c559216`。与非蒸馏 klein-base-9B 分开。
- 本地 transformer 与官方发布 SHA-256 一致：`0975d6b77b5f510b99547d6724a208e36527df654e8f6134f59ece3f9f30da58`。
- 本地 Qwen3-8B 文本编码器、VAE 和同一官方修订的完整 tokenizer；全部组件内容摘要进入身份。
- 首次加载发现既有 tokenizer 目录缺少 chat_template，未产图。随后只向 AllBench 独立目录获取官方7个 tokenizer 文件；原共享目录未修改。准备脚本检查模板存在，模板 `.jinja` 也参与内容摘要，变更后拒绝恢复。

## 协议和运行方式

`base_only` 模式，仅6个底模任务：3条公开中性提示（木制玩具机器人、花园水彩机器人、森林景观）×seed7/11；512×512，4步，guidance1.0，BF16，max_sequence_length512，CPU RNG，模型 CPU offload。没有 adapted 输出或参考图；保真/风格对照不会被伪造为零。

独立代码：`/net/node95/hd4/AllBench/20261027-flux2/code-final`，运行目录：同级 `run/`。只读复用 `qwen_image_workspace/envs/qwenimg`（Diffusers0.40.0、Transformers5.15.1、torch2.11.0+cu128）；使用 node94 空闲GPU1 RTX4090 24GB，通过 `CUDA_VISIBLE_DEVICES=1` 映射为进程 cuda:0。未停止其他GPU任务，没有向既有环境安装或升级包。

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src /net/node95/hd4/qwen_image_workspace/envs/qwenimg/bin/python3 -m allbench.cli generate /net/node95/hd4/AllBench/20261027-flux2/run/input.json --model-config /net/node95/hd4/AllBench/20261027-flux2/run/model-config.json --out /net/node95/hd4/AllBench/20261027-flux2/run/generated-final
```

完整生成后将 `generated-final/benchmark.json` 交给 `encode/evaluate`。输入文件与私有本地路径、图片、回执、特征均位于 Git 外。

## 验证范围

本地41项契约、指标、生成矩阵和恢复回归通过。新增覆盖：base_only完整矩阵、没有虚假adapted、逐图恢复不重复调用、损坏图片拒绝、回执篡改拒绝、模板变化失效、蒸馏guidance不被静默忽略。

真实生成采用提交两张后注入中断，再重放补四张；全量完成后再次重放零新生成。随后完成预训练 DINOv2-L 编码、逐提示 Vendi 报告及评测检查点恢复。

| 验证项 | 实际结果 |
|---|---|
| 完整生成矩阵 | 6/6张，512×512，3提示×2 seed |
| 部分中断恢复 | 先提交2张；恢复复用2张、新增4张；原图片及回执摘要不变 |
| 全量重放 | 新生成0张、复用6张、模型重新加载0次 |
| DINOv2-L | 6/6张预训练编码成功；Transformers4.57.6/torch2.8.0+cu128独立环境 |
| 评测重放及故障恢复 | 重算0次；两份报告逐字节相等 |
| 契约回归 | Windows41项通过；node94同套41项通过，84.22秒 |

每图记录的推理/传输/保存总计202.616秒；首张91.519秒，其余20.830–24.389秒。恢复阶段模型加载另计19.028秒，完整权重校验与首次加载不在202.616秒内；不能据此给出完整构建速度或生产吞吐。最大 `torch.cuda.max_memory_allocated` 为18,451,622,400字节（约18.45GB，17.18GiB），不是进程全部显存开销。

最终生成身份 `c02d06c6f2853b8ffd6fb8f1e0021f96b2d70af608b48985377d02fa09aea9d0`；报告 SHA-256 `24edcd8a4a758b067fd1c0fd26264c88c32ddc17176a3aca7e3cc512e5c9e39b`。原始汇总保存在 `run/verification-summary.json`，部分恢复证据在 `run/partial-recovery-proof.json`，完整阶段和零重放分别保存在 `run/completion-attempt.json` 与 `run/generated-final/generation-summary.json`。

复核：`scripts/evaluate_flux2_run.py <run_root>` 检查评测零重算与恢复；`scripts/verify_flux2_run.py <run_root>` 核对生成矩阵、原两张摘要、真实特征及报告一致性。最终代码快照与Git源码校验后发布。

本轮没有 LoRA、概念参考或人工金标，也没有启用 CLIP/VGG19。只能验收实际完成的底模生成和可用编码器链路，不能形成 LoRA 概念成功率、数据质量收益或完整四指标验收结论。每图计时包含推理与CPU offload传输，模型加载、完整权重摘要校验和网络盘时间需单列，不能当作生产吞吐。
