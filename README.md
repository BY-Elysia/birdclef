# BirdCLEF+ 2026 ONNX Perch Dual SSMs Vectorized MLP 代码说明

本目录是 Kaggle kernel `yuriygreben/birdclef-26-onnx-perch-dual-ssms-vectorized-mlp` 的本地副本。原始 kernel 标题是 **BirdCLEF+ 26 | ONNX Perch Dual SSMs Vectorized-MLP**，版本号为 5，元数据见 `kernel-metadata.json`。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `birdclef-26-onnx-perch-dual-ssms-vectorized-mlp.ipynb` | Kaggle notebook 原始代码。 |
| `birdclef-26-onnx-perch-dual-ssms-vectorized-mlp.py` | 从 notebook 导出的脚本版，便于搜索和阅读。 |
| `kernel-metadata.json` | Kaggle kernel 元数据，包括数据源、模型源、作者、版本等。 |
| `kernel-api-response.json` | Kaggle public API 返回的原始响应，里面包含 notebook 源码。 |

## 这个项目要解决什么问题

这是 BirdCLEF+ 2026 比赛的音频多标签分类提交代码。输入是一段 60 秒的野外录音 soundscape，代码把每个录音切成 12 个连续的 5 秒窗口，然后对每个窗口预测所有目标物种/类群的出现概率。

最终输出是 `submission.csv`：

- 每一行对应一个 5 秒窗口。
- 第一列是 `row_id`，格式是 `音频文件名去掉.ogg_窗口结束秒数`，例如某文件第一个窗口对应 `_5`，第二个窗口对应 `_10`，直到 `_60`。
- 后面的列来自 `sample_submission.csv`，也就是比赛要求预测的目标标签。代码里这些列被读成 `PRIMARY_LABELS`。
- 这是多标签预测，不是单分类预测：同一个 5 秒窗口可以同时出现多个物种。

本 notebook 的本地验证指标是宏平均 ROC-AUC：`macro_auc()` 会跳过训练集中没有正样本的类别，然后对剩余类别计算 macro ROC-AUC。

## 原始数据是什么

代码默认在 Kaggle 环境中运行，主比赛数据路径是：

```text
/kaggle/input/competitions/birdclef-2026
```

它使用的比赛原始数据包括：

| 文件或目录 | 作用 |
| --- | --- |
| `train_soundscapes/*.ogg` | 训练音频，野外录音。代码按 60 秒处理，每个文件切成 12 个 5 秒窗口。 |
| `test_soundscapes/*.ogg` | 测试音频。正式提交时 Kaggle 会挂载隐藏测试集。 |
| `train_soundscapes_labels.csv` | 训练音频的窗口级标签。代码按 `filename/start/end` 聚合标签，得到每个 5 秒窗口的标签集合。 |
| `taxonomy.csv` | 目标标签的分类学信息，包括 `primary_label`、`scientific_name`、`class_name` 等。代码用它做 Perch 标签映射、属级代理标签、按类群调温。 |
| `sample_submission.csv` | 定义提交文件格式。除 `row_id` 外的所有列就是要预测的目标类别。代码从这里生成 `PRIMARY_LABELS` 和 `N_CLASSES`。 |

核心数据形状：

- 音频采样率：`SR = 32000`。
- 每个预测窗口：`WINDOW_SEC = 5` 秒，即 `160000` 个采样点。
- 每个文件：固定按 `60` 秒处理，不足补零，超过截断。
- 每个文件窗口数：`N_WINDOWS = 12`。
- Perch embedding 维度：`1536`。
- 目标类别数：代码从 `sample_submission.csv` 读取，当前 notebook 运行时是 `234` 类。

除了比赛数据，kernel 还挂载了几个外部输入：

| Kaggle 数据源 | 作用 |
| --- | --- |
| `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1` | Google Perch v2 TensorFlow SavedModel，作为声学 backbone。 |
| `rishikeshjani/perch-onnx-for-birdclef-2026` | Perch v2 的 ONNX 版本和 ONNX Runtime wheel，用于更快的 CPU 推理。 |
| `jaejohn/perch-meta` | 预计算的 Perch 训练缓存，包含训练集窗口的 Perch logits、embeddings 和元数据。 |
| `ashok205/tf-wheels` | Kaggle 离线环境使用的 TensorFlow/TensorBoard wheel。 |
| `needless090/birdclef2026-sed-ensemble` | 可选 SED 外部模型，包含 EfficientNet B0/B3 fold 权重。 |
| `needless090/birdclef2026-sed-v5-trio` | 可选 SED 外部模型，包含 v5、CE seed、pseudo、pseudo2 权重。 |

## 模型是什么

这个 notebook 不是训练一个单一模型，而是一个多阶段音频分类管线：

```text
OGG 音频
  -> 60 秒补齐/截断
  -> 切成 12 个 5 秒窗口
  -> Perch v2 backbone 提取 logits + 1536 维 embedding
  -> 标签映射/属级 proxy logits
  -> LightProtoSSM 时序模型
  -> site/hour 先验 + PCA-MLP probes
  -> ProtoSSM 和 MLP 结果 0.5/0.5 融合
  -> ResidualSSM 二次误差修正
  -> temperature scaling + sigmoid
  -> 文件级置信度缩放、rank-aware scaling、自适应时序平滑、阈值锐化
  -> 可选 SED ensemble rank average + sonotype mirroring
  -> submission.csv
```

### 1. Perch v2 声学 backbone

代码优先使用 ONNX 版 Perch：

```python
ONNX_PERCH_PATH = /kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx
```

如果 ONNX Runtime 不可用，就回退到 TensorFlow SavedModel：

```python
MODEL_DIR = /kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1
```

Perch 对每个 5 秒窗口输出两类信息：

- `label` logits：Perch 自己词表上的鸟鸣/声音类别分数。
- `embedding`：1536 维音频表征，后续 SSM 和 MLP 都依赖它。

比赛的目标标签和 Perch 的标签不完全一致，所以代码先用 `taxonomy.csv` 的 `scientific_name` 与 Perch 的 `assets/labels.csv` 做映射。能映射上的类别直接取对应 Perch logit；映射不上的类别，会尝试找同属 genus 的 Perch 标签，把这些同属标签的最大 logit 当作 proxy signal。

### 2. Perch 训练缓存

训练集上跑 Perch 比较耗时，所以代码优先读取缓存：

- `perch_meta.parquet`
- `perch_arrays.npz`

搜索顺序是：

1. 外部挂载缓存，例如 `jaejohn/perch-meta`。
2. 当前 notebook 工作目录 `/kaggle/working/cache`。
3. 如果都没有，就从 `train_soundscapes` 重新跑 `run_perch()` 构建缓存。

缓存里的主要数组是：

- `sc_tr`：训练窗口的 Perch 分数，形状约为 `(训练窗口数, N_CLASSES)`。
- `emb_tr`：训练窗口的 Perch embedding，形状约为 `(训练窗口数, 1536)`。
- `meta_tr`：每个窗口的 `row_id`、`filename`、`site`、`hour_utc`。
- `Y_FULL_aligned`：与缓存窗口顺序对齐的多标签真值矩阵。

### 3. LightProtoSSM

`LightProtoSSM` 是第一个主要时序模型。它把每个文件的 12 个窗口作为一个序列处理，输入是：

- `emb`: `(文件数, 12, 1536)` 的 Perch embeddings。
- `perch_logits`: `(文件数, 12, N_CLASSES)` 的原始 Perch/competition-label logits。
- `site_ids` 和 `hours`: 从文件名解析出的地点和 UTC 小时。

模型结构包括：

- 1536 维 embedding 投影到 `d_model=128`。
- 位置编码，表示 12 个 5 秒窗口的位置。
- site embedding 和 hour embedding，加入地点/时间先验。
- 两层双向 `SelectiveSSM`，前向和反向分别建模时间上下文。
- Cross-attention，用于在 SSM 层之后重新聚合 60 秒上下文。
- 每个类别一个 prototype，初始化为该类别正样本 embedding 的均值。
- class-wise `fusion_alpha`，把 prototype 相似度 logits 和原始 Perch logits 融合。

训练目标是：

- `BCEWithLogitsLoss`：拟合真实多标签。
- `0.15 * MSE(out, raw_perch_logits)`：蒸馏约束，避免模型偏离 Perch 原始信号太远。

### 4. MLP probes

`train_mlp_probes()` 会为每个正样本数足够的类别训练一个轻量 MLP 二分类器。它不是直接吃原始音频，而是吃 Perch 特征和时间特征：

- Perch embedding 先标准化，再 PCA 到 64 维。
- 当前类别的原始 Perch score。
- 同一文件内前一窗口、后一窗口的 score。
- 同一文件内该类别 score 的 mean、max、std。

每个类别的 MLP 结构是 `(128, 64)` 的隐藏层。为了加速测试集推理，`VectorizedMLPProbes` 会把所有 sklearn MLP 的权重堆叠起来，用 PyTorch `bmm` 一次性完成所有类别的前向传播。

MLP 输出会以 `alpha_blend=0.4` 和原始 score 融合：

```text
adjusted_score = 0.6 * base_score + 0.4 * mlp_logit
```

### 5. site/hour 先验

`build_prior_tables()` 从训练标签统计：

- 全局类别频率。
- 每个 site 的类别频率。
- 每个 UTC hour 的类别频率。

`apply_prior()` 把这些频率转成 logit，再以 `lambda_prior=0.4` 加到测试分数上。直觉是：某些物种更常在特定地点或时间出现，模型可以用这个软先验调节概率。

### 6. 第一阶段融合

测试集上先得到两路分数：

- `proto_scores_flat`：LightProtoSSM 的输出。
- `sc_te_adjusted`：经过 site/hour prior 和 MLP probes 修正后的 Perch 分数。

代码用固定权重融合：

```python
first_pass_flat = 0.5 * proto_scores_flat + 0.5 * sc_te_adjusted
```

### 7. ResidualSSM 二次修正

`ResidualSSM` 是第二个 SSM。它的输入是：

- 原始 Perch embedding。
- 第一阶段融合分数 `first_pass_flat`。
- site/hour 元信息。

它不直接预测最终类别，而是学习第一阶段的误差：

```text
residual = Y - sigmoid(first_pass)
```

输出是一个 additive correction，最后按权重加回第一阶段分数：

```python
final_scores = first_pass_flat + correction_weight * correction_flat
```

提交流程里 `correction_weight=0.30`。

### 8. 校准和后处理

最后的分数还会经过多步后处理：

1. **Temperature scaling**：按生物类群调整 logit 温度。`Amphibia` 和 `Insecta` 使用 `0.95`，其他主要按 `1.10`。
2. **Sigmoid**：把 logit 变成概率。
3. **file_confidence_scale**：用每个文件 top-2 窗口置信度缩放整段文件的概率。
4. **rank_aware_scaling**：用每个文件最大置信度进一步压制不确定文件。
5. **adaptive_delta_smooth**：对低置信度窗口做更多相邻窗口平滑，对高置信度窗口尽量少动。
6. **apply_per_class_thresholds**：根据训练预测估计的每类阈值，把阈值以上的概率推高、阈值以下的概率压低。

### 9. 可选 SED rank ensemble

当前版本还移植了另一个 LB 0.934 notebook 中最关键的外部 SED 集成模块。它由 `OPT["use_sed_rank_ensemble"]` 控制，默认开启，但只有在 Kaggle notebook 中挂载对应权重数据集时才会实际运行；如果找不到权重或缺少依赖，会自动打印原因并跳过。

SED 分支会加载两组外部模型：

- `birdclef2026-sed-ensemble`：`sed_fold0.pt`、`sed_fold1.pt`、`sed_b3_fold0.pt`。
- `birdclef2026-sed-v5-trio`：`best_model_v5_focal.pt`、`best_model_ce_s123.pt`、`best_model_ce_s456.pt`、`v5_pseudo.pt`、`v5_pseudo2.pt`。

这些模型直接对 5 秒窗口的 mel spectrogram 做预测，然后与 Perch/ProtoSSM 管线输出做按类别的 rank average：

```text
final_rank = 0.70 * Perch/ProtoSSM rank + 0.30 * SED rank
```

之后还会对若干容易混淆的 `sonotype` 标签组做 mirroring：同一组内取最大概率并同步到组内其它标签。这一步是为了利用相近声型标签之间的相关性。

## 代码执行流程

| notebook cell | 主要内容 |
| --- | --- |
| Cell 0 | 安装 ONNX Runtime 和 TensorFlow wheel。 |
| Cell 1 | 设置 `MODE = "submit"` 或 `"train"`。 |
| Cell 2 | 导入库，设置路径、采样率、窗口长度、训练参数。 |
| Cell 3 | 读取 `taxonomy.csv`、`sample_submission.csv`、`train_soundscapes_labels.csv`，构造标签矩阵。 |
| Cell 4 | 加载 Perch v2，优先 ONNX，失败则 TensorFlow。 |
| Cell 4b | 为无法直接映射到 Perch 标签的类别构造 genus proxy logits。 |
| Cell 5 | 定义音频读取和 Perch 推理函数 `run_perch()`。 |
| Cell 6 | 读取或构建训练集 Perch 缓存，并对齐标签。 |
| Cell 7-7h | 定义指标、先验、MLP probes、阈值校准、平滑等工具函数。 |
| Cell 7i-7j | 定义 LightProtoSSM、TTA、ResidualSSM。 |
| Cell 8-8b | `MODE="train"` 时运行 OOF 验证。 |
| Cell 9 | 对隐藏测试集或本地 dry-run 音频跑 Perch。 |
| Cell 10 | 训练/应用 SSM 和 MLP，融合、校准、后处理，写出 `submission.csv`。 |

## 运行方式

在 Kaggle Notebook 中，保持 `MODE = "submit"` 即可生成提交文件：

```python
MODE = "submit"
```

如果要做本地交叉验证，把它改成：

```python
MODE = "train"
```

注意：

- 这份代码依赖 Kaggle 的 `/kaggle/input/...` 路径，本地 Windows 直接运行通常会缺数据路径。
- Kaggle hidden test 只在提交时可见；本地没有 `test_soundscapes` 时，代码会退化为从 `train_soundscapes` 选若干文件 dry-run。
- 训练缓存必须和当前比赛数据、`sample_submission.csv` 的类别顺序一致；代码会检查 `row_id` 和 `primary_labels`，不一致时应重建缓存。

## 一句话总结

这份 kernel 的核心思想是：用 Google Perch v2 快速把音频转成强声学 logits 和 embeddings，再用两个轻量 SSM 建模 60 秒内的时序上下文，用 per-class MLP probes 和 site/hour 先验补强类别判断，最后通过校准和平滑把每个 5 秒窗口的 234 类出现概率写成 Kaggle 提交文件。
