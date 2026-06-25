# SpectraLM 纯文本 LLM 迁移设计

## 1. 目标

将 SpectraLM 从图像与峰表联合输入的 VLM 微调项目，收敛为只使用结构化 1H/13C 峰表和可选分子式的纯文本 LLM 微调项目。

唯一基座模型为：

```text
/mnt/data/kimariyb/models/Qwen3-8B
```

最终推理流程：

```text
1H peak table + 13C peak table + optional formula
                      ↓
              Qwen3-8B text LLM
                      ↓
          Direct SMILES / Top-k candidates
                      ↓
     canonicalization + formula hard filter
                      ↓
        NMR rules + candidate reranking
```

## 2. 研究问题

主研究问题：

> 在 random 和 scaffold-disjoint 测试条件下，使用结构化 1H/13C 峰表进行指令微调的纯文本 LLM，结合分子式硬约束和候选排序后，能否可靠地恢复未知分子的连接结构？

主任务与消融任务分开训练：

1. `Peak table + Formula -> connectivity SMILES`：主任务；
2. `Peak table only -> connectivity SMILES`：无分子式消融。

不在同一训练中随机隐藏分子式，避免模型学习不稳定的输入协议，也避免无法区分分子式贡献和训练增强效应。

## 3. 范围

### 3.1 保留

- 1H 和 13C 峰表解析；
- 分子清洗、元素过滤、中性单组分策略和同位素移除；
- JSONL offset 缓存、文件句柄复用和 lazy dataset；
- response-only supervision；
- 官能团识别、谱学区域分类、候选排序和直接结构预测；
- 两阶段训练和早停；
- formula-matched 困难负样本；
- SMILES 规范化、分子式硬过滤、规则验证和候选排序；
- Exact Match、Connectivity Exact Match、Valid SMILES、Formula Accuracy、Tanimoto、Scaffold Match、Functional Group F1、谱学一致性和模型行为指标；
- CUDA/Unsloth 训练路径。

### 3.2 删除

- 所有谱图图像生成、缓存、加载、resize 和图像消息构建；
- 所有 VLM、视觉 collator、视觉模型加载和图像 token 处理；
- `full`、`image_only`、`peak_table_only`、`formula_only` 等旧模态枚举；
- 输入模态混合权重；
- 图像 SNR、渲染种子、图像尺寸和预渲染目录配置；
- 视觉模态消融实验；
- 旧 VLM 训练输出、预测、预渲染图片、演示图和 TOC 图；
- 只描述旧 VLM 研究设计的文档、规格和计划。

### 3.3 不做

- 不开发 Set Transformer、图扩散或其他新模型架构；
- 不使用二维 NMR、IR、MS 或反应物上下文；
- 不要求恢复无法由 1D NMR 支持的立体化学；
- 不把规则文本默认注入 prompt；
- 不在本轮直接扩展到 50k 或 100k，先完成 10k 纯文本 pilot。

## 4. 数据接口

### 4.1 原始 JSONL

现有 `samples.jsonl` 和 split manifest 继续使用。每条样本的模型可见字段为：

```json
{
  "id": "sample-id",
  "molecular_formula": "C8H10O",
  "1H_NMR": {
    "frequency": "400 MHz",
    "solvent": "unknown",
    "peaks": [
      {
        "shift": 7.18,
        "multiplicity": "d",
        "J": [8.2],
        "integration": 2.0
      }
    ]
  },
  "13C_NMR": {
    "frequency": "100 MHz",
    "solvent": "unknown",
    "peaks": [{"shift": 158.4}]
  },
  "smiles": "target connectivity SMILES"
}
```

数据文件不保存图像路径，不要求 `rendered/` 目录存在。

### 4.2 文本序列化

峰表统一序列化为稳定、紧凑、可解析的纯文本：

```text
Molecular formula: C8H10O

1H NMR:
7.18 ppm | d | J=8.2 Hz | integration=2
6.82 ppm | d | J=8.2 Hz | integration=2
3.78 ppm | s | J=- | integration=3

13C NMR:
158.40, 129.60, 121.30, 113.80, 55.30, 21.10 ppm
```

规则：

- 1H 峰按化学位移降序排列；
- 13C 位移按降序排列；
- 1H 位移保留两位小数；
- J 保留一位小数；
- 积分使用紧凑十进制表示，不强制取整；
- 13C 位移保留两位小数；
- 缺失字段写为 `unknown` 或 `-`，不推断不存在的信息；
- 无分子式实验完全删除 `Molecular formula` 行，不写 `unknown formula`。

稳定序列化减少无意义文本长度，并使输入哈希、缓存和回归测试可复现。

## 5. Prompt 与输出

### 5.1 System prompt

所有任务共享一个简短 system prompt：

```text
You are a molecular structure elucidation model for one-dimensional NMR data. Follow the requested output format exactly and do not output reasoning.
```

system prompt 只定义角色和输出约束，不包含规则库全文，不提供目标结构线索。

### 5.2 结构预测 prompt

主任务：

```text
Infer the molecular connectivity from the 1H and 13C NMR peak tables and the supplied molecular formula.

{spectral_context}

Output exactly one canonical connectivity SMILES string and nothing else.
```

无分子式消融：

```text
Infer the molecular connectivity from the 1H and 13C NMR peak tables.

{spectral_context}

Output exactly one canonical connectivity SMILES string and nothing else.
```

辅助任务继续使用独立、确定的任务 prompt。候选排序 prompt 只包含文本峰表、可选分子式和候选 SMILES。

### 5.3 关闭思考模式

训练和推理必须统一调用 Qwen chat template，并显式传入：

```python
enable_thinking=False
```

验收要求：

- 训练后的完整文本不包含 `<think>` 或 `</think>`；
- assistant target 不包含思考前缀；
- 推理输出若包含思考标签，计为格式不合规；
- preflight 在训练开始前验证监督 token 解码后与目标完全一致。

## 6. 模型与 LoRA

### 6.1 模型加载

使用：

```python
import unsloth
from unsloth import FastLanguageModel
```

`import unsloth` 必须位于 `src/training/train.py` 的其他 Transformers、TRL 或 PEFT 导入之前。

加载参数沿用 CUDA 服务器路径：

- `max_seq_length: 8192`；
- `load_in_4bit: true`；
- `attn_implementation: sdpa`；
- BF16；
- QLoRA。

### 6.2 LoRA 目标

新 adapter 只挂载语言模块：

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

Stage 2 通过 `PeftModel.from_pretrained(..., is_trainable=True)` 继续 Stage 1 adapter。

### 6.3 训练模式

使用 `FastLanguageModel.for_training(model)` 和纯文本 collator。项目中不得再导入：

```text
FastVisionModel
UnslothVisionDataCollator
PIL
```

## 7. Response-only collator

新增项目内纯文本 collator，职责为：

1. 接收 system/user/assistant messages；
2. 使用 `enable_thinking=False` 渲染完整对话；
3. 单独渲染 system/user 加 assistant generation prompt；
4. tokenize、padding 和 truncation；
5. 将 prompt token 的 label 设为 `-100`；
6. 只保留 assistant target 和终止 token 的监督；
7. 检查至少存在一个监督 token；
8. preflight 解码监督 token，并与目标文本严格比较。

若模板的 prompt token 不是完整对话 token 的前缀，collator 必须报错，不允许近似寻找边界。

## 8. 训练实验

### 8.1 10k pilot

先重建 80/10/10 split：

```text
8,000 train / 1,000 validation / 1,000 test
```

同时生成：

- grouped random split；
- scaffold-disjoint split。

10k 指总样本数，不再表示 9k train 加额外 val/test。

### 8.2 Formula 主实验

Stage 1：

- structure prediction：0.40；
- functional-group recognition：0.20；
- candidate ranking：0.30；
- spectral-region classification：0.10。

Stage 2：

- structure prediction：1.00；
- 从 Stage 1 最佳 adapter 继续训练；
- 使用 connectivity SMILES；
- 早停依据 validation loss。

### 8.3 No-formula 消融

使用完全相同的 split、模型、batch、优化步数、prompt 数量和任务权重，仅从输入中移除分子式。单独保存 adapter 和日志。

### 8.4 训练后 Gate

完成 10k 后先运行：

1. 训练集固定 1,000 条回放；
2. validation 全量推理；
3. test 全量推理；
4. formula 与 no-formula 成对比较。

如果训练集 Connectivity Exact Match 仍接近零，不进入 50k，先检查目标学习、训练步数、prompt 长度和候选生成。

## 9. 推理

### 9.1 Direct inference

- greedy decoding；
- `temperature=0`；
- `top_p=1`；
- `enable_thinking=False`；
- 输出一个 SMILES；
- 保存原始输出、规范化输出和生成行为诊断。

### 9.2 Candidate inference

- `K=32`；
- `temperature=0.7`；
- `top_p=0.9`；
- 对候选执行 RDKit 解析、规范化和去重；
- formula 主实验执行严格分子式过滤；
- no-formula 消融跳过公式过滤，但仍执行元素、中性、单组分和自由基策略；
- 不足 K 个唯一候选时如实记录，不使用重复项填充。

### 9.3 候选排序

当前阶段使用确定的两步排序：

1. 规则引擎为每个合法、唯一候选计算一致性结果，并按硬冲突数量、强规则满足数量和稳定 canonical SMILES 顺序预排序；
2. 将预排序后的候选和相同峰表交给 candidate-ranking prompt，由同一个文本 LLM 选择一个候选。

公式一致性是硬过滤条件，不进入软分数。若 LLM 输出不属于候选集合，记录 ranking failure，并回退到规则预排序第一名，同时分别报告原始 LLM 排序合规率和回退后结果。前向 NMR 模型不在本次迁移中实现，待 10k pilot 证明文本任务可学习后再单独设计。

## 10. 配置

保留一套当前配置，不保留 VLM 版本：

```text
configs/train_smoke.yaml
configs/experiments/train_stage1_formula_10k.yaml
configs/experiments/train_stage2_formula_10k.yaml
configs/experiments/train_stage1_no_formula_10k.yaml
configs/experiments/train_stage2_no_formula_10k.yaml
configs/experiments/infer_direct_formula_10k.yaml
configs/experiments/infer_candidates_formula_10k.yaml
configs/experiments/infer_direct_no_formula_10k.yaml
configs/experiments/infer_candidates_no_formula_10k.yaml
```

所有配置使用 `/mnt/data/kimariyb/models/Qwen3-8B`，不得出现以下字段：

```text
image_backend
rendered_image_dir
missing_image_policy
image_size
h_snr
c_snr
render_seed
input_mode
input_mode_weights
eval_input_mode_weights
```

## 11. 删除清单

### 11.1 源码与脚本

删除：

```text
src/spectra/
src/data/modalities.py
script/pre_render_jsonl_images.py
```

重写图像依赖：

```text
src/data/dataset.py
src/data/tasks.py
src/evaluation/prompts.py
src/training/arguments.py
src/training/train.py
src/training/inference.py
src/training/constrained_inference.py
src/training/model_setup.py
script/run_experiment.sh
script/run_train_cuda_48g.sh
```

### 11.2 测试

删除：

```text
tests/test_pre_render_jsonl_images.py
```

重写或精简：

```text
tests/test_nmr_parsing.py
tests/test_dataset_transform.py
tests/test_auxiliary_tasks.py
tests/test_training_arguments.py
tests/test_model_setup.py
tests/test_response_masking.py
tests/test_constrained_inference.py
tests/test_experiment_design.py
```

新增纯文本覆盖：

- 峰表稳定序列化；
- formula 与 no-formula prompt；
- system prompt；
- chat template 明确关闭 thinking；
- response-only token 边界；
- FastLanguageModel 加载和语言 LoRA 目标；
- direct/candidate inference 不接受图像参数；
- YAML 中不存在视觉字段；
- 仓库源码、配置和活动文档中不存在 VLM/视觉流程引用。

### 11.3 文档和产物

删除旧 VLM 文档、旧 VLM specs/plans 和视觉研究图片。重写：

```text
README.md
docs/experiments.md
docs/research_design.md
```

保留：

```text
docs/nmr_1d_rulebook.md
rules/nmr_1d.yaml
references/
```

删除所有旧产物：

```text
outputs/
dataset/**/rendered/
img/
```

原始 CSV、JSONL、split manifest、candidate sidecar 和索引缓存不得删除。

## 12. 错误处理

- 配置中发现旧视觉字段时立即报错，不能静默忽略；
- `include_formula=true` 但样本无分子式时，数据加载报错；
- no-formula 配置中不得启用 formula hard filter；
- assistant target 为空、包含 thinking 标签或无法定位 response 边界时，训练前报错；
- candidate inference 没有合法候选时记录 hard failure；
- Stage 2 adapter 路径不存在时立即报错；
- 模型路径不存在由 Unsloth 输出明确加载错误，不回退到在线下载。

## 13. 测试策略

迁移采用测试驱动方式：

1. 先修改测试，使其定义纯文本目标行为并确认因旧实现失败；
2. 实现最小纯文本数据和 prompt 路径；
3. 实现 collator 与 response-only 验证；
4. 迁移 FastLanguageModel 训练与推理；
5. 迁移 constrained inference；
6. 删除视觉模块和视觉测试；
7. 更新配置、脚本和文档；
8. 运行 `conda activate ml && python -m compileall src tests`；
9. 运行 `conda activate ml && pytest`；
10. 运行 smoke dry-run，验证模型加载前的数据路径；真正 CUDA smoke 由服务器执行。

## 14. 验收标准

迁移完成必须同时满足：

1. 生产源码不导入 PIL、Matplotlib、`FastVisionModel` 或 `UnslothVisionDataCollator`；
2. 训练样本只含 system/user/assistant 文本消息；
3. prompt 包含 1H/13C 峰表，并按配置包含或删除分子式；
4. 训练和推理均关闭 thinking；
5. 只有 assistant target token 被监督；
6. 所有 YAML 使用 Qwen3-8B 且无视觉字段；
7. direct 和 candidate inference 均可运行纯文本输入；
8. formula 候选推理严格过滤错误分子式；
9. no-formula 候选推理不读取目标分子式；
10. `outputs/`、`dataset/**/rendered/` 和 `img/` 旧产物被删除；
11. README 和实验文档只描述纯文本研究；
12. 编译检查和完整测试通过；
13. 服务器 smoke 日志证明 `enable_thinking=False` 且 response-only preflight 目标匹配。

## 15. 实施顺序

1. 冻结本设计；
2. 编写逐文件 TDD 实施计划；
3. 迁移文本 prompt 与数据集；
4. 迁移 collator、训练和推理；
5. 迁移候选约束流程；
6. 更新配置和脚本；
7. 删除视觉代码与历史产物；
8. 更新研究设计和使用文档；
9. 本地完成编译与单元测试；
10. 用户在 CUDA 服务器运行 smoke；
11. 分析 smoke 后启动 10k formula 主实验；
12. 主实验通过 Gate 后运行 no-formula 消融。
