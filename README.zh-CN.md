# auto-subtitle

[English](README.md) | 中文

在 RTX 5090 32 GB 上，本地生成日语音视频的简体中文字幕。
保留流程：**Anime Whisper 识别 → Gemma 4 31B QAT Q4 整集翻译 →
Qwen 排版（必须保持文本逐字不变）**。

按修订后的语境评分重新验证五个原有候选，最终保留 round 93。
Astra 独立首评、复评均为 **3 分**，仍未达到 4 分门槛。
详见 [评分规则](QUALITY.md) 和 [流程说明](WORKFLOW.md)。尚未验证音源准确性和播放效果。

```bash
# 仅检查本地模型、输入和运行环境，不推理
./run.sh input/example.mp4 --context-file input/context.txt \
  --output-dir output/example

# 增加 --execute 才会生成；输出目录必须尚不存在
./run.sh input/example.mp4 --context-file input/context.txt \
  --output-dir output/example --execute
```

使用 `.venv`、`ffmpeg` 和 `profiles/selected-local.json` 配置。
模型位于 `models/`。运行前，本地 Warden 的 Qwen DFlash 必须处于活动状态；
程序按阶段切换 GPU 模型并恢复原模型。须显式提供原始背景文本；整集输入
超过 32K 上下文容量时明确失败，不截断。每次处理一个媒体文件，原文件保留。

生成不会自动调用外部模型或批准发布。Astra 仅在独立评分命令中读取已授权的
日文识别稿、中文译文和原始背景，返回分数；不参与写作或修改。

保留的实测字幕：`output/selected/final/episode.zh.srt`。
本次证据：`docs/benchmarks/contextual-selection-20260914/`。
旧实验、模型备选和输出已迁入 `archive/experiments-20260914/workspace/`，
保留原路径映射和记录，不再作为活动流程；未删除实验数据。


## 限额句子修复实验

新实验从保留版出发，只修改精确定位的句子片段。先保留多个可行的日语读法，
再对照修改前后的日文、中文验证补丁；耗时的人声分离及残差信号识别只在第二轮
复访时按疑点触发。Gemma 与 Qwen 分别担任写作、检查，并在第三轮交换角色。
当前实验绑定已保存的 66 个原始片段，尚未替换默认媒体处理器。

```bash
# 只查看状态，不推理
./run.sh sparse-revisit --campaign output/quality-local-20260915

# 执行／续跑既定实验，包括 Astra 仅返回分数的独立评审
./run.sh sparse-revisit --campaign output/quality-local-20260915 --execute
```

最多三个候选，每个最多 90 分钟本地阶段时间，合计最多四小时。成功的请求直接
复用，失败请求最多一次技术恢复。所有识别、诊断及修改留在本机；外部评审不提供
问题位置或修正文案。两次独立评分均达到 4 分、且本地记录验证通过，才具备候选
资格；语境评分不等于音源或播放认证。详见[既定计划](design/quality-local-20260915.md)。

## 跨片段声音证据回顾实验

上一轮稀疏修复在相同声音证据池上的评分为 3、3、3，第三个结果回退到原译文，
直接继承基线分数。尚未达到 4，详见[实测记录](output/quality-local-20260915/RESULTS.md)。

后续 C1/C2 保留多种 ASR 读法，先生成仅含源语的全片语境图，再把跨片段的支持、
冲突及备选声音证据同时交给本地写作模型与检查模型。只生成精确的中文补丁，
原始日文仍标为未经音源认证。Astra 仅返回独立分数，不参与诊断或改写。

```bash
./run.sh context-revisit --campaign output/quality-context-20260915
./run.sh context-revisit --campaign output/quality-context-20260915 --execute
```

此实验仍绑定已保存的 66 个片段，不替换默认媒体流程。最多两个候选，每个最多
75 分钟，合计最多两小时本地阶段时间。详见[既定计划](design/quality-context-20260915.md)。

C1/C2 已结束，评分仍为 3/3。C1 留下四个修改片段，但最终本地回归检查未通过；
C2 的修改全部回退，继承基线分数。本地复访合计 68 分 20 秒，包含失败的回顾请求。
详见[实测结果](output/quality-context-20260915/RESULTS.md)。

## GLM 写作模型实验

G1 只把 C1 的补丁写作模型换为本机 GLM-4.7-Flash Q6_K，仍使用原基线、源语
语境图、既有诊断及完整定向证据；Qwen 检查和 Astra 仅评分的规则保持一致。
必须先用原生分词器确认所有完整提示能容纳，才开始生成。最多一个候选、一次
模型加载、75 分钟本地调度时间；到时仍须完成清理，并记录清理耗时。

```bash
.venv/bin/python -m src.glm_revisit --campaign output/quality-glm-20260915
.venv/bin/python -m src.glm_revisit --campaign output/quality-glm-20260915 --execute
.venv/bin/python -m src.glm_release output/quality-glm-20260915
```

首条命令仅准备并校验本地输入；已开始的候选不会重新启动。最后一条命令只有在
两次独立评分及本地检查均通过后才认可结果。实验仍绑定当前片段，不替换默认
媒体流程或已选字幕，详见[既定计划](design/quality-glm-20260915.md)。

G1 已完成，独立评分 **3**；本地检查通过，保留六个修改片段。本地阶段耗时
42 分 58 秒，其中 GLM 原生生成约 3 分 54 秒，重复权重校验占用了额外时间。
详见[结果](output/quality-glm-20260915/RESULTS.md)。

后续 M1 将 S1/S2/G1 中无冲突的既有补丁组合，再做全片回归检查；只试一个候选，
本地最多 45 分钟。若仍未达到两次确认的 4 分，N1 将在复访阶段重新识别约六秒的
原音频短窗，保留完整时间线、重叠及全部长窗证据，再回顾语境、局部修复并独立
评分。N1 最多一个候选、两小时本地处理；新证据池需先重新评分起始译文。
这些实验尚不替换默认处理器或已选字幕，参见 [M1](design/quality-assembly-20260915.md)
和 [N1](design/quality-short-audio-20260915.md) 的固定计划。


M1 已结束，独立评分 **3**，本地回归检查未通过。N1 在相同扩展声音证据池上，
从起始译文 **2** 分提高至 **3** 分，保留四个中文片段修复，并通过原生记录重放。
本地复访耗时 41 分 29 秒，其中短音频识别约 73 秒；初始字幕和早期证据为缓存，
并非全流程重新计时。详见 [N1 实测结果](output/quality-short-20260915/RESULTS.md)。

T1 保持 N1 声音证据及原始 G1 中文不变，让本地 Qwen 按实际裁剪时间顺序重新
解释全部源语观察，再构建一次全片源语回顾。短窗通常互补，某窗没包含某个词
不等于反证。随后仍由 Gemma 生成有限中文补丁，Qwen 逐项及全片核验；严格
接纳条件和两次独立评分保持不变。最多一个候选、两小时本地时间，不增加 ASR
或图像输入。实现分派前通过 43 项合成测试，测试通过不等于翻译达标。

```bash
.venv/bin/python -m src.temporal_revisit --campaign output/quality-temporal-20260915 --execute
.venv/bin/python -m src.temporal_release output/quality-temporal-20260915
```

详见 [T1 既定计划](design/quality-temporal-20260915.md)。
