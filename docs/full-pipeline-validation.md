# node94 完整链路验收

日期：2026-10-27。结论：匹配 FLUX.2-klein-base-9B 的标准 LoRA 自动评测链已实际贯通。正式 LoRA、触发词、参考划分和提示替换配置后可运行；本轮参考为公开生成夹具，不能给出真实概念成功率。

## 实际执行

使用 node94 空闲 GPU1 RTX4090 24GB，BF16、CPU offload、512×512；非蒸馏 base-9B 和已有 rank32 Kohya/Musubi LoRA，倍率1。6类提示各1条×seed7/11×base/adapted=24张。4步/CFG4仅为模块验收预算，正式效果协议应另行固定；BFL 非蒸馏默认50步/CFG4与本轮不混用。

LoRA 的原始336个张量经 Diffusers 官方转换得到288个 A/B 张量、144个目标模块。加载后逐张量核对实际挂载值；base 禁用、adapted 启用，回执保存开关和倍率。12个配对的文件摘要均有变化，这只证明输出受到适配器影响，不证明概念语义正确。

2张参考来自此前公开机器人生成样本，明示 synthetic、fixture group 和未知训练预算；触发词为工程占位词，不假称真实训练触发词。没有读取真实参考图或制造人工标签。

| 验收项 | 结果 |
|---|---|
| 统一 `run` 入口 | 配置冻结→两环境预检→生成→编码→指标→报告全部完成 |
| 真生成矩阵 | 24/24，完整 base/adapted；6种提示类别全部覆盖 |
| 真实编码 | DINOv2-L、DataComp CLIP ViT-L/14、VGG19，26张图；CLIP 另编码6条语义文本 |
| 指标 | 12组，DINO cosine/SCD、CLIP、逐提示 Vendi、参考 Gram 距离全部实算；6组 adapted 另有配对风格变化 |
| CSV/HTML | CSV108行，其中102项 measured；6项是 base 自身不计算 base-style-drift 的正常 comparator 标记；HTML24张对照图路径均校验 |
| 生成恢复 | 提交2张后注入中断，重放复用2张、新增22张；原图片和回执摘要保持一致 |
| 编码恢复 | 提交2张特征后注入中断，重放只补24张图和6条文本；原特征摘要不变 |
| 评测恢复 | checkpoint 后注入中断，恢复 computed=false，报告内容与 checkpoint 完全一致 |
| 整链路重放 | 新生成0、生成模型加载0、图/文本编码0、指标重算0、导出报告复用 |
| 回归 | node94：56项全部通过，57.83秒；Windows：55通过、1项符号链接权限测试跳过，21.79秒 |
| 源码一致性 | 最终运行的13个 Python 源文件 SHA256 与本地发布源文件逐一相同 |

生成环境：Python3.11、torch2.11.0+cu128、Diffusers0.40.0、Transformers5.15.1、PEFT0.20.0。编码环境：Python3.10.18、torch2.8.0+cu128、torchvision0.23.0+cu128、Transformers4.57.6。复用只读环境，没有安装/升级共享环境，也未停止其他GPU任务。

逐图推理/传输/保存合计555.457秒；峰值 `torch.cuda.max_memory_allocated` 18,740,652,544字节。该时间不包含全量权重摘要、所有模型加载、编码和报告，不能换算为生产吞吐。

## 身份和复核路径

| 项目 | 摘要 |
|---|---|
| base transformer 文件身份 | `7510abea50829330ab6011a9a21884a1d01a57d52244a0b7aa69af34aea31805` |
| LoRA 文件身份 | `3560dee4d33226a95c21bfde4f043def997db5076cb8cf8b82a93a087f9fd048` |
| DINO 权重身份 | `e5b3a674cc978ae02da8c4aa29b276045c0d2448121ba3c7a68c8e60183d6d87` |
| CLIP 权重身份 | `b29cc48844abbaf987ee227eb6548fce6072013400380bdc68976d7311c7e126` |
| VGG 权重身份 | `7b449cc1f51c2917371124ec9062bd8ca14a1ae53d2a3002d725603335985a6b` |
| 生成身份 | `1c9cb900c67e992102f35a393ef41789d52e5e6cea67ee73a6dc6b815aae9bea` |
| report.json SHA256 | `4d98fbc7a9c0587c59c417a067ca3924c184264c8d96f233755d7162aab515b8` |

模型文件身份是 AllBench 的文件名/内容摘要组合，不是 raw safetensors SHA。参考、图片、权重、特征及逐项证据均在 Git 外。

94根目录 `/net/node95/hd4/AllBench/20261027-full`：最终代码 `code-v2`，运行 `run-v2`，证据 `evidence/verification-summary.json`、`runtime-summary.json`、各次尝试和部分恢复文件摘要。`scripts/verify_full_run.py <run> --evidence <evidence>` 可重新核验。失败状态对应3次明确注入的中断；最终完成及完整重放均成功。

94已准备启动器和正式输入模板。填写 `formal-input/case.json` 的 LoRA、概念、触发词、提示、参考路径和真实亲缘组后执行：

```bash
/net/node95/hd4/AllBench/20261027-full/allbench-run /net/node95/hd4/AllBench/20261027-full/formal-input/run.json --out /net/node95/hd4/AllBench/runs/your-formal-benchmark-v1
```

模板预填当前匹配 base-9B 组件和三个编码器，正式 case 默认50步/CFG4；需要变更协议时用新目录。适配范围和输入说明见[完整手册](full-pipeline.md)。真实概念效果、人工阈值、固定训练预算数据收益和多训练 seed 仍待正式材料验收。
