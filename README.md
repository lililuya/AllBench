# AllBench

独立的 LoRA 图像模型离线评测入口。参考 [LyCORIS 论文](https://arxiv.org/html/2309.14859v2)和[作者评测仓库](https://github.com/cyber-meow/LyCORIS-evaluation)，将固定协议、生成结果、图像/特征校验和分层指标报告放在一起。

**当前支持完整执行链**：固定配置→FLUX.2-klein-9B 底模/标准 LoRA 配对生成→DINO、CLIP、VGG19 编码→指标→JSON/CSV/HTML 报告，生成/编码/评测均可恢复。统一入口及正式样本替换方式见[完整链路手册](docs/full-pipeline.md)。模块实测证据与概念效果验收分别记录；合成 Demo 不能代表模型效果。

```bash
python -m allbench.cli run /outside/repo/input/klein-run.json --out /outside/repo/runs/my-concept-v1
```

从 [examples/klein-run.json](examples/klein-run.json)、[样本配置](examples/klein-case.json)和[组件配置](examples/klein-model.json)复制模板到仓库外。正式 LoRA、触发词和参考集准备好后替换配置即可，不需要改评测代码。当前生成器支持匹配 Klein 底模的标准 LoRA，其他 LyCORIS 算法/架构需专用适配。

2026-10-27：[node94 完整实测](docs/full-pipeline-validation.md)通过24张真实配对生成、26张图/6条文本的三个真实编码器、全部指标、三阶段中断恢复和全量零新增重放；56项回归通过。参考为生成夹具，正式概念效果待你的独立样本和人工金标验收。

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
4. 只保留参考图也能运行 `plan`。由所选生成器执行 jobs，在同一配置/提示/seed 下分别用 base 和 adapted 出图，再向 images 填入 `role/prompt_id/seed/concept_id/path/sha256`。默认 `paired` 要求完整两组矩阵；显式 `evaluation_mode=base_only` 只要求 base，允许参考为空，不自动复制 adapted。计划阶段摘要与最终含输出的 manifest 摘要不同，最终编码绑定后者。
5. 使用本地权重 `encode` 或导入同结构的特征包，然后 `evaluate`。目前 base 和 adapted 共用一份固定生成配置；后端实际执行及配置一致性需要生成器回执和人工核验，本工具不能从像素反推出采样参数。

```bash
python -m pip install -e '.[encode]'
allbench encode /outside/repo/benchmark.json --out /outside/repo/features --dino /models/dinov2-large-snapshot --clip /models/clip-snapshot --vgg /models/vgg19-dcbb9e9d.pth --device cuda:0
allbench evaluate /outside/repo/benchmark.json --features /outside/repo/features/features.json --out /outside/repo/report
```

三个权重参数可独立省略，至少选一个。只读本地文件，不自动下载模型。DINO 使用 Hugging Face `Dinov2Model` CLS；CLIP 使用本地 `CLIPModel/CLIPProcessor` 的投影空间；VGG 需要 torchvision VGG19 完整 state_dict。权重来源、hash、预处理不同即不同协议，不能直接与论文数字比较。论文 CLIP 使用的是 DataComp ViT-L/14；其他 CLIP 权重只能标为工程变体。

## FLUX.2-klein-9B 真实生成

`generate` 支持 `base_only` 和 `paired` 文生图。采用本地单文件 transformer、配套 Qwen3-8B/tokenizer/VAE 和显式 Diffusers 配置；不隐式下载权重。官方蒸馏版示例为4步、guidance 1.0，与非蒸馏 `klein-base-9B` 区分。LoRA 必须匹配实际训练底模；运行时需 PEFT，加载后逐张量验证挂载，base 禁用适配器、adapted 启用指定倍率。

新环境可安装 `.[generate]`。当前生成器验证于 Transformers5.15.1，编码器验证于4.57.6；请使用独立环境分别运行，`encode` 与 `generate` 两个 extra 暂不合并安装。依赖安装不下载模型权重，GPU Torch 构建需匹配实际驱动。

```bash
allbench generate /outside/repo/input.json --model-config /outside/repo/model-config.json --out /outside/repo/generated
allbench encode /outside/repo/generated/benchmark.json --out /outside/repo/features --dino /models/dinov2-large-snapshot --device cuda:0
allbench evaluate /outside/repo/generated/benchmark.json --features /outside/repo/features/features.json --out /outside/repo/report
```

本地 `model-config.json` 指定 `transformer`（单文件）、`transformer_config`（config.json 所在目录）、`text_encoder`、`tokenizer`、`vae`（对应本地目录）、`scheduler_config`（文件）、`is_distilled`、`device` 和 `cpu_offload`。用 `scripts/prepare_flux2_smoke.py` 可从这些显式路径准备3提示×2 seed的小批输入；该准备脚本按指定 HF revision 获取少量官方配置，权重不下载。实际协议固定引擎版本、模型组件摘要、尺寸、步数、guidance、文本长度和 CPU RNG 种子。

tokenizer 需要完整官方文件及 `chat_template.jinja`；只有词表和 tokenizer.json 不足以运行 Klein。配置与 node94 证据见 [真实生成器记录](docs/flux2-klein-generator.md)。

生成器将每张图与逐项回执原子提交；已提交项经图片摘要验证后跳过。断在图片写入后、回执提交前的孤立产物会重新生成；回执已提交项不重算。完整后输出新的 `benchmark.json`，带回执摘要与生成来源，编码/评测继续验证这些关联。硬终止残留锁的处理同下文。无 LoRA 和参考图时，保真、风格变化仍为未测；真实底模出图不能代替 LoRA 概念验收。

## 指标与输出

| 指标 | 当前计算 | 解释 |
|---|---|---|
| 概念保真代理 | DINO 图像跨集合平均余弦、归一化平方质心距离 | train 与 holdout 分开；相似度不等于身份成功率 |
| 文图对齐 | 对应提示与生成图的 CLIP 余弦 | 需要语义替换文本与相同 CLIP 空间 |
| 多样性 | 每个独立 prompt 内多 seed 的线性核 Vendi，再宏平均 | 附 n，少于两个 seed 未测；和作者 train/trigger 池化口径有区别 |
| 底模风格变化 | 同 prompt/seed 的 VGG19 五层归一化 Gram 平方差之和 | `style` 提示最适合检查风格保留；其他提示作为诊断 |
| 参考风格相似度 | 生成图与 train/holdout 参考分别计算跨集合平均 Gram 平方距离 | 风格保真代理，越低越相似；不能代替人工风格判断 |

输出 `report.json`，按概念、提示类别、base/adapted 分组；`run` 还导出 CSV、HTML 图片对照和待审模板。缺失指标是 `not_measured/null/reason`。没有总分或自动通过阈值。`prompt_coverage` 暴露缺失类型。`model_effect_validated` 始终为 false，表示本工具未替代人工效果验收。运行证据、来源和待补样本见 [研究与缺口](docs/research-evidence.md)、[样本组织清单](docs/sample-checklist.md)和[验证记录](docs/validation.md)。

## 恢复与边界

特征包以 ID 映射到 checksummed NPZ，`allow_pickle=False`；图片、协议、特征、软件版本参与结果身份。正常重放不重算指标。提交前中断可从校验过的 checkpoint 恢复；损坏结果或身份变化拒绝复用，改用新目录。

`.lock` 用独占创建阻止并发写入。硬杀进程后锁可能残留：先核实对应进程已经退出，再手工移除该独立结果目录中的 `.lock`，重放原命令。不要删除其他任务的锁。编码按图片/文本原子提交，恢复只补缺项；完整重放零编码。`run` 保存各次尝试日志和失败阶段，改配置/输入使用新目录。

Git 只放代码、文档和合成夹具的生成逻辑；真实图像、标注、模型、依赖、特征和结果在仓库外。DataPipeline 继续管理数据处理和发布；AllBench 用 `dataset_release` 回链，本版未接业务数据库或自动补数工单。
