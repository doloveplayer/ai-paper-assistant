# VRAG RAG 闭环评估体系 技术文档

## 一、设计思路

### 1.1 为什么需要闭环评估

RAG 系统有多个独立环节（检索 → 压缩 → 生成），任一环节退化都会导致最终回答质量下降，但**仅看最终回答无法定位根因**。例如：

- 回答错误 → 是检索没找到相关页面？还是 VLM 看到了正确内容但生成了幻觉？
- 回答笼统 → 是 RRF 融合排序不准？还是 LLM 压缩丢失了关键细节？

闭环评估的核心思想：**分层度量每个环节的质量 → 根据指标组合诊断根因 → 给出可操作的改进建议**。

### 1.2 三层评估架构

```
Layer 1: 检索质量 (Retrieval Quality)
  └─ 度量：检索到的页面是否包含 ground-truth 相关信息
  └─ 指标：Recall@k / Precision@k / MRR / NDCG@k / Hit Rate
  └─ 数据源：Qdrant 返回的 ScoredPoint（vision_res, text_res, fused_results）

Layer 2: 生成质量 (Generation Quality)
  └─ 度量：LLM 生成的回答是否忠实、切题、事实正确
  └─ 指标：Faithfulness / Answer Relevance / Factual Correctness
  └─ 数据源：LLM-as-a-Judge（复用 vLLM :8000 Qwen2.5-7B）

Layer 3: 端到端效率 (End-to-End Efficiency)
  └─ 度量：系统运行的时间/Token 成本
  └─ 指标：Total Latency / Token Consumption / Throughput
  └─ 数据源：agent_core.py call_model 遥测拦截
```

### 1.3 数据捕获策略：线程局部 Side-Channel

**为什么不用返回值？** LangChain `@tool` 装饰器要求工具函数返回 `str`，改签名会破坏 Agent 的 tool-calling 协议。

**解决方案**：`EvalContext` — 基于 `threading.local()` 的单例 side-channel。

```
EvalRunner (主线程)                    Agent 线程
    │                                      │
    ├─ 创建 shared_state dict              │
    ├─ 启动 agent_thread ─────────────────→│
    │                                      ├─ EvalContext.enable()
    │                                      ├─ app.stream() 执行
    │                                      │   ├─ call_model() → ctx.capture_telemetry()
    │                                      │   └─ search_vision_knowledge() → ctx.capture_retrieval()
    │                                      ├─ 将 traces 写入 shared_state
    │                                      └─ EvalContext.disable()
    ├─ agent_thread.join()                 │
    ├─ 从 shared_state 读取数据            │
    └─ 计算指标 + 运行 Judge               │
```

关键点：
- 数据捕获代码在 `vision_rag.py` 和 `agent_core.py` 中各 ~15 行，用 `try/except` 包裹，评估关闭时零开销
- 每个用例使用唯一 `thread_id`（`eval_{case_id}_{uuid}`），避免对话记忆交叉污染

---

## 二、评估指标详解

### 2.1 检索指标 (Retrieval Metrics)

实现于 [eval/eval_metrics.py](eval/eval_metrics.py)。

#### Recall@k（召回率）
```
Recall@k = |retrieved_pages[:k] ∩ gt_relevant_pages| / |gt_relevant_pages|
```
- 回答"相关页面被找到了多少"
- 低 Recall → 检索遗漏，需检查索引覆盖、query expansion

#### Precision@k（精确率）
```
Precision@k = |retrieved_pages[:k] ∩ gt_relevant_pages| / |retrieved_pages[:k]|
```
- 回答"检索结果中有多少是真正相关的"
- 低 Precision → 检索噪声，需调整 RRF 融合权重

#### MRR (Mean Reciprocal Rank)
```
MRR = 1 / rank_of_first_relevant_hit
```
- 首个相关结果的倒数排名。1.0 = 第一名命中，0.5 = 第二名命中
- 低 MRR → 最相关页面排序靠后

#### NDCG@k (Normalized Discounted Cumulative Gain)
```
NDCG = DCG(actual) / DCG(ideal)
DCG = Σ relevance_i / log₂(i + 2)
```
- 考虑排序位置的累计收益，比 MRR 更精细
- relevance 简化为二元（相关=1，不相关=0）

#### Hit Rate
```
Hit Rate = 1 if 至少命中 1 页 else 0
```
- 最简单的二值指标，适合快速扫视

#### 分路对比指标

同时计算视觉路和文本路的独立 Recall@k，用于诊断哪一路是瓶颈：
- `vision_recall` > `text_recall` → 视觉信息丰富（CV 论文图表多）
- `text_recall` > `vision_recall` → 文本描述充分（综述类论文）
- 两者都低 → 索引覆盖不足或 query 与论文内容不匹配

### 2.2 生成指标 (Generation Metrics)

实现于 [eval/eval_judge.py](eval/eval_judge.py)。

#### Faithfulness（忠实度，1-5 分）

```
Prompt: 用户问题 + 检索上下文(最多5页×1000字) + AI回答 → 逐声明验证
输出: score + reasoning + claims_verified[{claim, supported, evidence}]
阈值: < 3.0 → 幻觉风险
```

核心检测逻辑：将回答拆解为原子声明，逐一在检索上下文中验证是否有依据。

#### Answer Relevance（回答相关性，1-5 分）

```
Prompt: 用户问题 + AI回答 → 判断是否直接回应问题
输出: score + reasoning + missing_aspects[] + irrelevant_parts[]
阈值: < 3.5 → 偏离主题
```

#### Factual Correctness（事实正确性，1-5 分）

```
Prompt: Ground-truth 事实列表 + AI回答 → 匹配/遗漏/矛盾分析
输出: score + reasoning + matched_facts[] + missed_facts[] + contradicted_facts[]
阈值: < 3.0 → 事实错误
```

注意：此指标需要标注数据（`gt_answer_facts`），仅对提供了 ground truth 的用例有效。

#### Judge 技术参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen2.5-7B-Instruct (vLLM :8000) | 复用已有推理服务 |
| Temperature | 0.0 | 确保确定性输出 |
| Max Tokens | 512 | 仅需要 JSON 结构化输出 |
| 输出格式 | JSON | 正则兜底提取 |
| 单次超时 | 60s | 超时则降级为 error |

### 2.3 效率指标 (Efficiency Metrics)

在 `agent_core.py` 的 `call_model` 节点中拦截，每次 LLM 调用记录：

| 指标 | 含义 | 用途 |
|------|------|------|
| E2E Latency | 单次 LLM 调用的端到端延时 | 定位慢查询 |
| Input Tokens | 输入 Token 数（含 system prompt + history） | 评估上下文膨胀 |
| Output Tokens | 输出 Token 数 | 评估回答长度 |
| TPOT | Time Per Output Token (ms) | 纯生成速度，排除 prompt 处理 |
| Throughput | Tokens per second | 整体推理效率 |

---

## 三、评估数据集设计

实现于 [eval/eval_dataset.py](eval/eval_dataset.py)，数据文件 [eval/data/eval_dataset.json](eval/data/eval_dataset.json)。

### 3.1 数据结构

```python
@dataclass
class EvalCase:
    id: str                        # 唯一标识 (q001-q010)
    query: str                     # 用户提问（中英文均可）
    paper_id: str | None           # 目标论文 ID（限定检索范围，可选）
    paper_name: str | None         # 目标论文名称（可读标识）
    gt_relevant_pages: list[int]   # Ground-truth 相关页码（用于检索评估）
    gt_answer_facts: list[str]     # Ground-truth 答案应包含的关键事实（用于事实正确性评估）
    gt_keywords: list[str]         # Ground-truth 答案应包含的关键词（辅助参考）
    category: str                  # 问题类别
    difficulty: str                # 难度
```

### 3.2 用例覆盖矩阵

10 个用例，覆盖 5 种类别 × 3 个难度：

| ID | 类别 | 难度 | 有 gt_pages | 说明 |
|----|------|------|-------------|------|
| q001 | architecture | medium | - | 雾天检测模型架构 |
| q002 | dataset | medium | - | 恶劣天气检测数据集 |
| q003 | methodology | medium | - | 数据增强/预处理技术 |
| q004 | methodology | hard | [3,4,5] | Domain Adaptation 方法 |
| q005 | methodology | medium | - | UAV 特征融合 |
| q006 | comparison | hard | - | YOLO 方法性能对比 |
| q007 | methodology | medium | - | 注意力机制的作用 |
| q008 | general | easy | [1,2,3] | 恶劣天气检测综述（挑战与方案） |
| q009 | methodology | hard | [3,4,5] | Multi-Task Learning |
| q010 | methodology | medium | - | 去雨/去雾预处理 |

### 3.3 两类评估模式

- **含 gt_pages 的用例（q004, q008, q009）**：可计算完整的检索+生成指标，用于精细诊断
- **不含 gt_pages 的用例（其余 7 个）**：仅计算生成+效率指标，用于监控整体回答质量

---

## 四、闭环诊断系统

实现于 [eval/eval_reporter.py](eval/eval_reporter.py) 的 `_analyze_results()` 方法。

### 4.1 阈值配置

定义于 [eval/eval_config.py](eval/eval_config.py) 的 `DiagnosticThresholds`：

```python
@dataclass
class DiagnosticThresholds:
    recall_at_3: float = 0.5
    precision_at_3: float = 0.4
    mrr: float = 0.3
    faithfulness: float = 3.0       # 1-5 分制
    answer_relevance: float = 3.5
    factual_correctness: float = 3.0
    token_efficiency_tps: float = 2.0
```

### 4.2 诊断映射表

| 触发条件 | 严重度 | 根因诊断 | 建议操作（含具体参数路径） |
|---------|--------|---------|--------------------------|
| Recall@3 < 0.5 | critical | 检索遗漏关键页面 | 检查 ColPali 索引完整性；增大 `pool_size` multiplier (当前 6→10)；尝试 query expansion |
| Precision@3 < 0.4 | warning | 检索噪声过多 | 调整 RRF k 值 (当前 60→40)；加强 paper_id 过滤；考虑加权 RRF |
| MRR < 0.3 | warning | 首个相关结果排序靠后 | 文本路优先权重；score normalization；增大 query token 数 |
| Faithfulness < 3.0 | critical | 生成存在幻觉 | 降低 VLM temperature (0.3→0.1)；保留更多原文 (增大 truncate)；强化护栏规则 |
| Answer Relevance < 3.5 | warning | 回答偏离主题 | 强化 VLM prompt 中的聚焦指令；检查 `_summarize_payloads` 是否引入偏差 |
| Factual Correctness < 3.0 | critical | 事实与标准答案矛盾 | 检查压缩是否丢失关键信息；VLM post-check 引用页码；检查检索覆盖 |
| Throughput < 2.0 t/s | info | 推理效率低 | 检查 vLLM continuous batching；增大 GPU memory utilization；排查 GPU 抢占 |

### 4.3 诊断报告格式

每次评估生成 Markdown 报告，包含四个部分：

1. **汇总指标** — 检索/生成/效率三维度的均值 + 通过率
2. **用例详情** — 每个用例的指标 + Judge 评语
3. **闭环诊断** — 低于阈值的指标 → 具体改进建议
4. **历史追踪** — 趋势对比（需多次运行）

报告自动保存到 `eval/reports/eval_report_{timestamp}.md`。

---

## 五、历史追踪与趋势对比

实现于 [eval/eval_storage.py](eval/eval_storage.py)。

### 5.1 存储结构

```
eval/history/
├── index.json                # 运行索引（最近 100 条）
│   [{"run_id": "2026-04-27_143052", "timestamp": "...", "summary": {...}}]
├── 2026-04-27_143052.json    # 单次完整结果
│   {run_id, timestamp, dataset_name, num_cases, summary, results: [...]}
└── ...
```

### 5.2 提供的能力

```python
from eval import EvalStorage

storage = EvalStorage()

# 保存
path = storage.save_run(results, dataset)

# 查询历史
runs = storage.list_runs()

# 加载指定运行
data = storage.load_run("2026-04-27_143052")

# 对比两次运行
comparison = storage.compare_runs("run1", "run2")
# → {recall: {before: 0.5, after: 0.67, delta: 0.17, trend: "up"}, ...}

# 趋势数据（供可视化）
trend = storage.latest_trend(num_runs=5)
# → {timestamps: [...], metrics: {recall: [...], faithfulness: [...]}}
```

---

## 六、实现架构总览

### 6.1 文件依赖关系

```
eval_runner.py (编排器)
    ├── eval_dataset.py   → 加载 test cases
    ├── eval_tracer.py    → 捕获 Agent 中间数据 (via vision_rag.py + agent_core.py)
    ├── eval_metrics.py   → 检索指标计算
    ├── eval_judge.py     → LLM-as-a-Judge (via vLLM :8000)
    ├── eval_reporter.py  → Markdown 报告 + 闭环诊断
    └── eval_storage.py   → JSON 持久化 + 趋势对比

修改点:
    tools/vision_rag.py   → search_vision_knowledge() 中插入 tracer 捕获 (~18 行)
    agent_core.py         → call_model() 中插入 telemetry 捕获 (~12 行)
```

### 6.2 关键设计决策回顾

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据捕获方式 | Thread-local side-channel | 不改 LangChain 工具签名，零侵入 |
| Judge 模型 | Qwen2.5-7B (vLLM :8000) | 复用已有推理，无需额外 GPU 显存 |
| Judge 模式 | 三个专项 Judge (非单一 Mega-Judge) | 2025 研究共识：专项 Judge 在忠实度上的对齐率 (~96%) 高于单一 Judge |
| 数据集格式 | JSON + Python dataclass | 轻量，无需安装额外依赖 (避免 pydantic) |
| Python 兼容 | `from __future__ import annotations` | 支持 Python 3.10 的 `X | None` 语法 |
| Agent 隔离 | 每用例独立 thread_id | 避免对话记忆交叉污染 |
| 线程安全 | Thread-local + shared_state dict | EvalContext 在 agent 线程内启用，数据通过共享容器传回 |

### 6.3 执行流程（完整）

```
run_evaluation()
  │
  ├─ EvalDataset.from_json()           # 加载测试用例
  │
  └─ for each case:
       │
       ├─ 启动 agent_thread
       │   ├─ EvalContext.enable()     # 在 agent 线程内启用追踪
       │   ├─ app.stream(inputs)       # 驱动 LangGraph 状态机
       │   │   ├─ call_model()  → ctx.capture_telemetry()
       │   │   ├─ ToolNode 执行
       │   │   │   └─ search_vision_knowledge() → ctx.capture_retrieval()
       │   │   └─ call_model()  → ctx.capture_telemetry() + 最终回答
       │   ├─ 将 traces 写入 shared_state
       │   └─ EvalContext.disable()
       │
       ├─ agent_thread.join()
       ├─ 从 shared_state 读取数据
       │
       ├─ compute_retrieval_metrics()   # Recall/Precision/MRR/NDCG/Hit Rate
       ├─ FaithfulnessJudge.evaluate()  # LLM-as-a-Judge (需要 vLLM)
       ├─ AnswerRelevanceJudge.evaluate()
       ├─ FactualCorrectnessJudge.evaluate()
       │
       └─ CaseResult
  │
  ├─ EvalReporter.generate_markdown()  # 汇总 + 诊断报告
  └─ EvalStorage.save_run()            # 持久化到 eval/history/
```

---

## 七、使用指南

### 7.1 前置条件

```bash
# 四个服务必须全部运行
sudo docker run -d -p 6333:6333 -p 6334:6334 ... qdrant/qdrant
python -m vllm.entrypoints.openai.api_server --port 8000 ...  # 文本 LLM
python -m vllm.entrypoints.openai.api_server --port 8001 ...  # 视觉 VLM
# Chainlit 可选（评估不需要）
```

### 7.2 运行评估

```bash
# 方式 1：命令行一键评估
cd /home/c2216-3090/disB/hyh/AI
/home/c2216-3090/ProgramFile/conda/miniconda3/envs/ai_agent/bin/python -m eval.eval_runner

# 方式 2：编程调用（按条件筛选）
python -c "
from eval import run_evaluation

# 仅评估困难用例
run_evaluation(difficulty='hard')

# 仅评估方法论类别
run_evaluation(categories=['methodology'])

# 全量评估
run_evaluation()
"

# 方式 3：使用 EvalRunner 进行更精细控制
python -c "
from eval import EvalRunner

runner = EvalRunner()
runner.load_dataset()
results = runner.run_all()
report = runner.generate_report()
runner.save_results()

# 查看历史趋势
storage = runner.storage
trend = storage.latest_trend(5)
print(trend)
"
```

### 7.3 查看结果

- **终端输出**：实时显示每个用例的评估结果
- **报告文件**：`eval/reports/eval_report_{timestamp}.md`
- **历史记录**：`eval/history/index.json` + 各次运行的 JSON 文件

### 7.4 自定义阈值

编辑 [eval/eval_config.py](eval/eval_config.py) 中的 `DiagnosticThresholds`：

```python
@dataclass
class DiagnosticThresholds:
    recall_at_3: float = 0.6        # 更严格的召回要求
    faithfulness: float = 3.5       # 更严格的忠实度要求
    # ...
```

### 7.5 添加新测试用例

编辑 [eval/data/eval_dataset.json](eval/data/eval_dataset.json)，在 `cases` 数组中添加：

```json
{
  "id": "q011",
  "query": "你的问题...",
  "paper_id": null,
  "paper_name": null,
  "gt_relevant_pages": [],
  "gt_answer_facts": ["期望的事实1", "期望的事实2"],
  "gt_keywords": ["关键词1"],
  "category": "methodology",
  "difficulty": "medium"
}
```

- `gt_relevant_pages` 非空时 → 激活检索指标计算
- `gt_answer_facts` 非空时 → 激活事实正确性 Judge
- 即使两者都为空，仍会运行 Faithfulness + Relevance + 效率指标
