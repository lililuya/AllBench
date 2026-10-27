# 正式 LoRA 与样本接入

目标是把已训练好的 LoRA 和参考样本接入同一条生成→编码→指标→报告链路。工具不执行训练；数据版本训练收益仍需相同训练预算的独立对照。

## 准备输入

把 `examples/klein-case.json`、`klein-model.json`、`klein-run.json` 复制到仓库外的本轮输入目录，填写路径和内容。

1. `klein-model.json`：LoRA 训练所用的同一 Klein 底模、本地组件及完整 tokenizer。非蒸馏 base 与蒸馏模型分别设置 `is_distilled`；case 的同名字段必须一致。蒸馏版要求 CFG=1，示例非蒸馏50步/CFG4仅是起始配置，须按自己的固定协议选择，不能沿用本轮4步技术试点来断言效果。
2. `klein-case.json`：LoRA 文件、倍率、实际触发词、概念类型、数据 release；train 和独立 holdout 参考图片、真实亲缘组；六类提示及 `evaluation_text`。后者用自然语言解释触发词，不能包含原触发词，CLIP 超长文本在生成前拒绝。
3. `klein-run.json`：两个 Python 环境、三个本地评测权重、空闲物理 GPU 编号。`require_all_metrics=true` 强制三个编码器、每概念 train/holdout 及至少两 seed；缺项在出图前失败。

`prepare` 为缺省的模型 revision、LoRA revision、图片 SHA 自动计算内容摘要；已填写的摘要必须正确，不覆盖不匹配值。相对路径按各自配置文件目录解释。训练样本数、样本次、步数和训练 seed 不知道时保留 null，不能伪造预算。示例每划分1张只用于展示结构，不代表正式样本量充足。

## 一条命令运行

先安装 AllBench 的基础包。生成与编码使用不同环境；生成环境还需要支持相应版本的 Diffusers、Transformers、PEFT，编码环境需要 torchvision 和 Transformers4。可以在各环境安装对应 extra；本轮94复用只读既有环境，不修改它们。

```bash
python -m allbench.cli run /outside/repo/input/klein-run.json --out /outside/repo/runs/my-concept-v1
```

命令依次执行配置冻结、两个环境预检、base/adapted 生成、DINO/CLIP/VGG19 编码、评测和报告。生成阶段校验实际挂载的每个转换后 A/B 张量，并对 base 关闭、adapted 启用 LoRA。当前生成器支持标准 LoRA 的 Kohya/Musubi 和 Diffusers A/B safetensors；LoHa、LoKr、DoRA、文本编码器 LoRA、多 LoRA 组合和其他架构尚需适配，遇到这些格式明确失败。离线特征及生成结果导入仍可独立使用。

主要结果：

| 路径 | 用途 |
|---|---|
| `pipeline.json` | 阶段成功/失败、最近回执及尝试次数 |
| `prepared/` | 冻结后的输入、组件和编码器摘要 |
| `logs/` | 每次尝试的各阶段诊断日志 |
| `generated/benchmark.json` | 完整生成矩阵和逐图回执 |
| `features/features.json` | 三编码器特征索引和运行环境 |
| `evaluation/report.json` | 权威分层指标，含未测原因 |
| `report/index.html` | 指标表、覆盖情况和同提示/seed 图片对照 |
| `report/metrics.csv` | 便于后续比较的长表 |
| `report/review-template.json` | 尚无人工标签的待审记录；填写时复制到其他路径 |

HTML 使用相对路径引用 generated 图片，保留整个 run 目录一起查看。正式数据和待审标签留在 Git 外；不要修改带摘要的报告模板，复制后再标注。本版没有人工金标导入/自动校准或自动补数；不能把空模板解释为已经完成盲评。

## 恢复与换数据

中断后原样重跑 `run`：逐图回执、逐图片/文本特征、评测 checkpoint 依次验证并复用。全量重放不会重新加载生成模型/编码模型，不新生成图片、不重算特征或指标；环境预检和摘要校验仍执行。配置、代码、模型或参考文件变化时使用新结果目录，防止混入旧分数。损坏产物拒绝复用，日志定位到具体阶段。

每层 `.lock` 阻止并发。进程被硬杀时可能留锁；先确认对应进程已退出，再手工删除该 run 内对应阶段的锁，原命令恢复。正常失败会释放锁。工具不自动抢锁。

## 正式验收仍需的材料

模块贯通不等于概念生效。需要真实独立参考、覆盖泛化/泄漏/风格/无关能力的提示、人工概念/属性关系金标，并按领域校准代理指标；比较数据方案还需相同底模、训练 recipe、预算和多个训练 seed。当前不生成统一总分、成功阈值或显著性结论。
