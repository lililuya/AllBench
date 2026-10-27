# 研究证据与实施边界

核查日期：2026-10-27。按 ICLR 2024 完整研究论文阅读，重点是工程复现条件，不做投稿评价。

## 来源

- 论文：[Navigating Text-To-Image Customization](https://arxiv.org/html/2309.14859v2)，重点 §4、§5.2、附录 D.4–D.6、E 与 G；[OpenReview](https://openreview.net/forum?id=wfzXa8e783) 本次有浏览器验证，使用 arXiv v2。
- 算法库：[KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)，核查 dev tree `b5d65a3a5b9866979292bd15f1a5e3c1e770181a`。它主要提供微调算法；当前 kernel benchmark 不能当作模型效果评测。
- 作者评测：[cyber-meow/LyCORIS-evaluation](https://github.com/cyber-meow/LyCORIS-evaluation/tree/7c5758c8338aa59c4601bb68db8d5112b15748c4)，MIT，Copyright 2023 CyberMeow。

## 作者已有实施方案

作者仓库公开了图像/文本编码、image similarity/SCD、text similarity、Vendi、style loss、Frechet distance、IQA 及生成脚本。主要源码为 `lycoris_eval/evaluation/eval_*.py`、`metrics/vendi.py`、`encoding/encoders.py`。因此评估并非只有论文描述；但其 setup.py 的 install_requires 为空，运行依赖需另建环境，文件组织和 WebUI 生成流程有实验约定，不能把 clone 成功当成可复现验收。

## AllBench 的映射与差异

| 能力 | 实施情况 | 尚需验证 |
|---|---|---|
| 图像相似度与 SCD | 独立数值实现；DINOv2 可选本地权重 | 目标领域与人工身份判断的相关性 |
| CLIP 文图相似度 | 数值核心及可选 HF CLIP 编码路径；显式语义替换 | DataComp 权重/预处理对齐及 GPU 验收 |
| Vendi | 同 prompt 多 seed，附样本数 | 原作者 train/trigger 使用池化，本版统一逐提示，是工程选择 |
| VGG19 style loss | 同提示同 seed 配对；五层卷积前激活 Gram | 预训练 VGG19 GPU 路径与人工风格校准 |
| 风格 LoRA 的参考风格相似度 | 当前只提供概念标记与通用保真代理 | 对训练/留出参考的 VGG style fidelity 尚未实现，不能以 DINO 替代 |
| Concept leak / 泛化 | agnostic、alter、style 分层与 base 对照 | 自动泄漏判定、关系/属性金标和人工阈值 |
| FD/FID、IQA、SHAP、跨类归一化 | 本版未移植 | 独立研究任务；不作为小样本核心门槛 |

AllBench 依据公开数学定义整理独立接口，未整体复制作者仓库及其图片/权重。作者代码来源和 MIT 原文记录在 THIRD_PARTY_NOTICES.md；其他模型依赖仍遵循各自许可。

## 结论适用范围

通用数据 benchmark 实际应衡量“固定训练条件下，数据版本改变后各能力的变化”。本次工具只实现其中离线评测层；固定训练执行、数据预算配平和多训练种子不在当前运行时。CLIP/DINO 是代理特征，风格保留与身份保真也不是同一个目标。建议保留人工盲评、任务属性/关系金标、错误标签与不确定项，逐领域校准；无通用成功阈值。

当前 DataPipeline 已有分层人工抽样及加权区间，但 M4 模型归因/补数任务未实现，不迁出其已验证审计代码，不勾选原 10.1–10.3。AllBench 接收 release ID 和外部结果；数据库结果导入及自动补数仍需后续接口实现。
