# AllBench

独立的 LoRA 图像模型离线评测入口。参考 [LyCORIS 论文](https://arxiv.org/html/2309.14859v2)和[作者评测仓库](https://github.com/cyber-meow/LyCORIS-evaluation)，将固定协议、生成结果、图像/特征校验和分层指标报告放在一起。

**当前为 v0.1 骨架**：已提供 manifest 校验、生成任务清单、离线特征导入、数值指标、可选本地预训练编码器与恢复机制。没有内置训练器或通用生成器；真实底模/LoRA 生成和模型效果验收仍待接入。合成 Demo 的图片和向量均不代表模型效果。

## 快速运行

Python ≥3.10，建议独立环境：

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[test]'
allbench demo --out /outside/repo/allbench-demo
allbench validate /outside/repo/allbench-demo/benchmark.json
allbench plan /outside/repo/allbench-demo/benchmark.json --out /outside/repo/jobs.json
allbench evaluate /outside/repo/allbench-demo/benchmark.json --features /outside/repo/allbench-demo/toy-features/features.json --out /outside/repo/results
python -m pytest -q
```

将 `/outside/repo` 替换为真实仓库外目录；PowerShell 可以使用 `E:/Codex_Data/AllBench`。同一 `evaluate` 命令再次运行会验证输入和结果并直接恢复，不重算指标。Demo 目录必须是新的目录。

## 真实数据接入

1. 在 Demo 的 `benchmark.json` 结构基础上填写真实底模 revision、LoRA revision/方法/倍率、数据 release、训练预算与固定推理配置。未知训练统计保留 `null`，不得据此做等预算因果结论。
2. 为每个概念提供 `reference` 图片，明确 `train` 或 `holdout`、亲缘 `group_id`、文件路径和 SHA-256。至少有一张参考；建议独立留出参考。字节或亲缘组跨集合直接拒绝，近重复仍需外部筛查。
3. 填写 `prompts`：`train/trigger/alter/style/agnostic-class/agnostic`，每条关联概念，含实际 `text` 和语义替换后的 `evaluation_text`。触发词不作为 CLIP 已知语义；CLIP 不允许静默截断超长文本。
4. 只保留参考图也能运行 `plan`。由所选生成器执行 jobs，在同一配置/提示/seed 下分别用 base 和 adapted 出图，再向 images 填入 `role/prompt_id/seed/concept_id/path/sha256`。`validate` 要求全部预期单元齐全。计划阶段摘要与最终含输出的 manifest 摘要不同，最终编码绑定后者。
5. 使用本地权重 `encode` 或导入同结构的特征包，然后 `evaluate`。目前 base 和 adapted 共用一份固定生成配置；后端实际执行及配置一致性需要生成器回执和人工核验，本工具不能从像素反推出采样参数。

```bash
python -m pip install -e '.[encode]'
allbench encode /outside/repo/benchmark.json --out /outside/repo/features --dino /models/dinov2-large-snapshot --clip /models/clip-snapshot --vgg /models/vgg19-dcbb9e9d.pth --device cuda:0
allbench evaluate /outside/repo/benchmark.json --features /outside/repo/features/features.json --out /outside/repo/report
```

三个权重参数可独立省略，至少选一个。只读本地文件，不自动下载模型。DINO 使用 Hugging Face `Dinov2Model` CLS；CLIP 使用本地 `CLIPModel/CLIPProcessor` 的投影空间；VGG 需要 torchvision VGG19 完整 state_dict。权重来源、hash、预处理不同即不同协议，不能直接与论文数字比较。论文 CLIP 使用的是 DataComp ViT-L/14；其他 CLIP 权重只能标为工程变体。

## 指标与输出

| 指标 | 当前计算 | 解释 |
|---|---|---|
| 概念保真代理 | DINO 图像跨集合平均余弦、归一化平方质心距离 | train 与 holdout 分开；相似度不等于身份成功率 |
| 文图对齐 | 对应提示与生成图的 CLIP 余弦 | 需要语义替换文本与相同 CLIP 空间 |
| 多样性 | 每个独立 prompt 内多 seed 的线性核 Vendi，再宏平均 | 附 n，少于两个 seed 未测；和作者 train/trigger 池化口径有区别 |
| 底模风格变化 | 同 prompt/seed 的 VGG19 五层归一化 Gram 平方差之和 | `style` 提示最适合检查风格保留；其他提示作为诊断 |

输出 `report.json`，按概念、提示类别、base/adapted 分组；缺失指标是 `not_measured/null/reason`。没有总分或自动通过阈值。`prompt_coverage` 暴露缺失类型。`model_effect_validated` 始终为 false，表示本工具未替代人工效果验收。运行证据、来源和待补样本见 [研究与缺口](docs/research-evidence.md)、[样本组织清单](docs/sample-checklist.md)和[验证记录](docs/validation.md)。

## 恢复与边界

特征包以 ID 映射到 checksummed NPZ，`allow_pickle=False`；图片、协议、特征、软件版本参与结果身份。正常重放不重算指标。提交前中断可从校验过的 checkpoint 恢复；损坏结果或身份变化拒绝复用，改用新目录。

`.lock` 用独占创建阻止并发写入。硬杀进程后锁可能残留：先核实对应进程已经退出，再手工移除该独立结果目录中的 `.lock`，重放原命令。不要删除其他任务的锁。编码在最终发布 feature manifest 后支持零编码重放；中途编码失败需要重新编码，当前不做逐图缓存恢复。

Git 只放代码、文档和合成夹具的生成逻辑；真实图像、标注、模型、依赖、特征和结果在仓库外。DataPipeline 继续管理数据处理和发布；AllBench 用 `dataset_release` 回链，本版未接业务数据库或自动补数工单。
