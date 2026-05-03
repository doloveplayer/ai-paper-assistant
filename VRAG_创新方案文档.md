# VRAG：视觉增强型学术研究智能体 — 创新方案文档

---

## 一、场景选择与痛点分析

### 1.1 应用场景

**计算机视觉学术论文的智能检索与深度研读**

面向 CV 研究人员、研究生及 AI 算法工程师，覆盖论文发现→下载→精读→交叉比对的全流程。聚焦领域：**恶劣天气条件下目标检测（Adverse-Weather Object Detection）**——涉及去雨/去雾/低光照增强、域自适应、多模态融合等子方向。

### 1.2 核心痛点

| # | 痛点 | 传统方案局限 | 本项目解决方案 |
|---|------|-------------|---------------|
| 1 | **图表理解缺失** | 纯文本 RAG 无法解析论文中的网络架构图、实验曲线、检测结果可视化 | ColPali 视觉编码 + VLM 多模态分析 + **ROI 智能裁剪** |
| 2 | **公式/表格失真** | PDF 文本提取导致数学公式和表格严重乱码 | PyMuPDF 空间碰撞检测 + 表格→Markdown 还原 |
| 3 | **跨模态检索鸿沟** | "展示 YOLO 在雾天的检测对比图"类查询无法匹配纯文本索引 | 双路混合检索：ColPali 视觉 + BGE-M3 文本，RRF 融合排序 |
| 4 | **视觉 Token 爆炸** | 全页论文图片含大量无效背景，每张图约 2000 vision tokens | **ColPali 热力图 ROI 裁剪 + 25% 上下文扩边 + 拼图合成**，节省 ~60% Token |
| 5 | **知识碎片化** | 多篇论文相关信息分散，缺乏统一检索入口 | Qdrant 统一向量库，跨论文跨页码语义检索 |
| 6 | **GPU 资源受限** | 学术环境多为单卡消费级 GPU（如 RTX 3090 24GB） | 四位一体模型共享单卡：INT4 量化 + 懒加载 + CPU 卸载 |
| 7 | **论文安全风险** | 公开 PDF 可能嵌入恶意 JavaScript/触发器 | pikepdf 工业级安全扫描，拦截 JS/Launch/SubmitForm |
| 8 | **长对话记忆退化** | 传统摘要压缩导致记忆越来越模糊，丢失关键细节 | **四层分层记忆 + Qdrant 外部化长期记忆** (Mem0 思想) |
| 9 | **重复 Prompt 推理开销** | 每次请求重发完整 System Prompt，浪费算力 | **vLLM Prefix Caching** + Prompt 静态→动态倒序拼接，≥60% 缓存命中 |

---

## 二、系统架构设计

### 2.1 总体架构（三层体系）

```
+--------------------------------------------------------------+
|                    前端展示层 (Chainlit UI)                      |
|  . 异步队列 + 后台线程 -> 避免 WebSocket 超时                     |
|  . 自动补全引用来源 . 工具调用实时可视化 . 即时回答流式展示          |
+-----------------------------+--------------------------------+
                              | LangGraph State Machine
+-----------------------------v--------------------------------+
|                    智能体编排层 (agent_core.py)                   |
|  +------------+   +-----------+   +------------------------+ |
|  | call_model |<->|  tools    |   | 四层分层记忆管线         | |
|  | (LLM推理)   |   | (工具执行)  |   | init_core (Layer 1)    | |
|  +------------+   +-----------+   | retrieve_memory (L5)   | |
|       |               ^           | manage_context (L3)    | |
|       |               |           | cleanup_ephemeral (L4) | |
|       v               |           | update_profile (L2)    | |
|  should_continue      |           | extract_memory (L5)    | |
|  (tool_calls?)        |           +------------------------+ |
|       |               |                                      |
|       +--> tools -----+                                      |
|  SQLite Checkpoint + Qdrant user_memory -> 跨会话持久化记忆    |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                    数据与模型层                                  |
|  +------------------------+  +-----------------------------+ |
|  |   Qdrant 向量数据库      |  |       模型服务 (vLLM)         | |
|  |   . colpali_vision      |  |  . :8000 Qwen2.5-7B-AWQ     | |
|  |     128d 多向量 MAX_SIM  |  |    --enable-prefix-caching  | |
|  |   . text_dense          |  |  . :8001 Qwen2-VL-2B-AWQ    | |
|  |     1024d 稠密向量 COS   |  |    --enable-prefix-caching  | |
|  |   . user_memory (NEW)   |  |  . ColPali 8-bit (按需加载)  | |
|  |     1024d 长期记忆       |  |  . BGE-M3 (CPU 推理)        | |
|  +------------------------+  +-----------------------------+ |
+--------------------------------------------------------------+
```

### 2.2 数据流（端到端检索链路）

```
用户查询 "YOLO 在雾天场景的检测精度对比"
    |
    v
[1] LangGraph Agent (Qwen2.5-7B) 判断 -> 调用 search_vision_knowledge
    |
    v
[2] 长期记忆检索: Qdrant user_memory 检索 Top-3 相关偏好 → 注入上下文
    |
    v
[3] 双路 Query 编码
    +-- 视觉路: ColPali encode -> 128d x n_patches 多向量
    +-- 文本路: BGE-M3 encode -> 1024d 稠密向量
    |
    v
[4] Qdrant 独立检索 (pool_size = top_k x 6)
    +-- colpali_vision 通道 -> MAX_SIM 评分
    +-- text_dense 通道 -> COSINE 评分
    |
    v
[5] RRF 倒数排名融合 (k=60) -> Top-K 页面
    |
    v
[6] ColPali ROI 智能裁剪 (NEW)
    +-- 逐页计算 32×32 相似度热力图
    +-- Top-5 峰值 patch → 25% 边距 ROI → 原图裁剪
    +-- ROI 数 > 3? → 拼图合成 (Pillow)
    +-- 无显著热点? → 回退全图
    |
    v
[7] 文本压缩: 本地 LLM 将多页文本压缩为 <=500 字中文摘要
    |
    v
[8] VLM 综合分析: Qwen2-VL 接收 裁剪 ROI(Base64) + 文本摘要 -> 生成分析报告
    |
    v
[9] Agent 格式化输出 (Markdown + 引用来源)
    |
    v
[10] 后处理管线: cleanup_ephemeral → update_profile → extract_memory (后台线程)
```

### 2.3 LangGraph 分层记忆状态机设计

```
                         START
                           |
                           v
                    +---------------+
                    |  init_core    |  ← Layer 1: 注入不可变 SOP
                    +-------+-------+
                            |
                            v
                    +---------------+
                    |retrieve_memory|  ← Layer 5: Qdrant user_memory 检索 Top-3
                    +-------+-------+
                            |
                            v
                 +---------------------+
                 | token_check_router  |
                 | Token > 5000?        |
                 +------+------+-------+
                        | YES  | NO
                        v      v
              +-----------+  +------------+
              | manage_   |  |   agent    |<----------------------------+
              | context   |  |(call_model)|                              |
              | (Layer 3) |  +-----+------+                              |
              +-----+-----+        |                                     |
                    |              | has tool_calls?                     |
                    +------>       v              NO                     |
                            +----------+ --> cleanup_ephemeral (Layer 4) |
                            |  tools   |         ↓                       |
                            |(工具执行) |    update_profile (Layer 2)     |
                            +----------+         ↓                       |
                             执行完毕返回   extract_memory (Layer 5, bg)  |
                                                 ↓                       |
                                                END                      |
```

**State 定义 (四层记忆结构)**:
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 完整对话历史
    profile: dict              # Layer 2: {research_interests, mentioned_papers, preferences}
    working_context: list      # Layer 3: [{question, answer, timestamp}, ...]
    core_instructions: str     # Layer 1: immutable SOP
```

**State 各层说明**:

| 层 | 字段 | 生命周期 | 更新策略 |
|----|------|---------|---------|
| Layer 1 | `core_instructions` | 永久保留 | init_core 节点一次性写入，绝不修改 |
| Layer 2 | `profile` | 动态更新 | update_profile 节点 regex + 关键词探测 |
| Layer 3 | `working_context` | 滑动窗口 (5轮) | manage_context 节点超过窗口后移出旧轮次 |
| Layer 4 | (messages 中的 ToolMessage) | 每次回答后替换 | cleanup_ephemeral 节点替换为 ~100 字内部备忘 |
| Layer 5 | Qdrant `user_memory` | 永久 (外部化) | extract_memory_facts 后台线程 BGE-M3 编码入库 |

---

## 三、AI 模型与算法说明

### 3.1 模型矩阵

| 角色 | 模型 | 参数量 | 量化 | 显存占用 | 推理设备 |
|------|------|--------|------|----------|----------|
| **Agent 推理引擎** | Qwen2.5-7B-Instruct-AWQ | 7B | INT4 (AWQ) | ~8.4 GB | GPU (vLLM :8000) |
| **视觉编码器** | ColPali (PaliGemma-3B + LoRA) | 3B | INT8 (BitsAndBytes) | ~4 GB | GPU (on-demand) |
| **文本编码器** | BGE-M3 (XLM-RoBERTa) | 568M | FP32 | 0 GB GPU | CPU |
| **视觉分析模型** | Qwen2-VL-2B-Instruct-AWQ | 2B | INT4 (AWQ) | ~6 GB | GPU (vLLM :8001) |

### 3.2 核心算法

#### 3.2.1 ColPali 视觉-语言后期交互检索

ColPali 基于 PaliGemma-3B（SigLIP ViT-448px + Gemma-2B 文本解码器），通过对比学习微调生成**多向量表征**：

- 每张论文页面生成 N 个 128 维 patch 向量（N ≈ 32-128，取决于图片分辨率）
- 检索时使用 **MAX_SIM 比较器**：查询向量与文档的每个 patch 向量计算 COSINE 相似度，取最大值求和

```
MAX_SIM(q, d) = Σ_i max_j COSINE(q_i, d_j)
```

- 优势：保留细粒度视觉信息，支持"图片中的文字"检索（如架构图中的标注、实验曲线上的数字）

#### 3.2.2 BGE-M3 文本稠密检索

BGE-M3 基于 XLM-RoBERTa，输出 1024 维 L2 归一化向量：

- 支持中英双语，8192 Token 上下文窗口
- 使用 COSINE 距离进行稠密向量相似度计算
- 部署在 CPU 上，单页文本编码仅需 10-50ms

#### 3.2.3 RRF 倒数排名融合算法

将异构检索结果（视觉 + 文本）统一排序：

```
RRF_score(d) = Σ_{r ∈ R} 1 / (k + rank_r(d))

其中: k = 60 (平滑参数), R = {vision_ranking, text_ranking}
```

- 优点：无需训练，不依赖 score 校准，天然适合异构检索通道融合
- 候选池扩大至 `top_k × 6`，RRF 排序后截断至 Top-K，提升召回率

#### 3.2.4 两阶段文本压缩

为解决 VLM 分析时多页文本 Token 爆炸问题：

**阶段一**：本地 LLM（Qwen2.5-7B）将多页文本压缩为 ≤500 字中文摘要，输入包含用户查询意图
**阶段二**：VLM 接收压缩摘要 + 原始图片，进行视觉-文本综合分析

效果：文本 Token 节省约 **90%**（~3000 字/页 → ~500 字总计），同时保留与查询相关的关键信息。

#### 3.2.5 表格空间碰撞检测

PyMuPDF 原生 `find_tables()` 在非标准 PDF 上召回率低。本方案使用：

1. `page.find_tables()` 检测表格 → DataFrame → Markdown
2. `page.get_text("blocks")` 获取按坐标排序的文本块
3. **空间碰撞检测**：文本块 Rect 与表格 Rect 有交集 → 插入 `[TABLE_i_PLACEHOLDER]`
4. 非表格文本进行正则清洗（滤除 10 位+十六进制乱码、控制字符）

#### 3.2.6 ColPali ROI 视觉 Token 剪枝（核心创新）

**动机**：全页论文图片送入 VLM 时，每张图约 2000 vision tokens，其中大量无效背景（白边、无关段落）浪费显存和推理时间。

**算法流程**：

1. **热力图计算**：对检索到的每个页面，复用 ColPali query embedding（GPU 端保留），逐页前向得到 image_embeddings，调用 `get_similarity_maps_from_embeddings()` 生成 32×32 patch 级相似度热力图
2. **峰值定位与上下文扩边**：在热力图上定位 Top-5 峰值 patch，映射回原图像素坐标；每个 ROI 向外扩展 25%（`context_padding=0.25`），确保公式符号定义、表格列标题等周边上下文不丢失
3. **均匀性回退**：若热力图标准差 < 0.3 × 均值（无显著热点），自动回退至全图模式
4. **拼图合成**：若所有页面的 ROI 总数超过 VLM 的 `--limit-mm-per-prompt` 限制（3 张），使用 Pillow 将所有 ROI 拼接为一张合成图（网格布局），规避 API 报错
5. **显存管理**：每页计算后 `torch.cuda.empty_cache()` 释放中间张量

**数学表达**：

```
给定 query_embeddings Q ∈ ℝ^(T_q × 128) 和 image_embeddings I ∈ ℝ^(1024 × 128)

相似度热力图 S ∈ ℝ^(32×32)：
S[i,j] = max_{t ∈ [0, T_q)} (Q_t · I_{(i×32+j)} / (||Q_t|| × ||I_{(i×32+j)}||))

ROI 选择：
peaks = TopK(argmax(S), k=5)
roi = { (cx ± patch_w×(0.5+padding), cy ± patch_h×(0.5+padding)) for each peak }
```

**效果**：节省约 60% 视觉 Token，同时保留核心图表/公式/表格的完整上下文。

#### 3.2.7 四层分层记忆 + 外部化长期记忆

**动机**：传统摘要式压缩 (summary string) 在多次触发后越来越模糊，无法区分不同信息的生命周期。用户聊了 50 篇论文后，单一 summary 字段无法精确检索历史偏好。

**四层架构**：

| 层 | 存储位置 | 内容 | 生命周期 | 更新策略 |
|----|---------|------|---------|---------|
| Layer 1: 核心指令 | `state.core_instructions` | SOP 规则、输出格式 | 永久 | 一次性写入 |
| Layer 2: 用户画像 | `state.profile` (JSON) | 研究兴趣、论文 ID、偏好 | 动态 | regex + 关键词探测 |
| Layer 3: 工作上下文 | `state.working_context` (list) | 最近 5 轮 Q&A | 滑动窗口 | 移出旧轮次 |
| Layer 4: 临时备忘 | messages (替换后) | 工具结果的 ~100 字摘要 | 每次回答后 | RemoveMessage + AIMessage memo |
| Layer 5: 长期记忆 | Qdrant `user_memory` | 所有历史关键事实 | 永久 (外部化) | BGE-M3 编码 + Qdrant upsert |

**Layer 5 (外部化长期记忆) 详细流程**：

1. **提取** (`extract_memory_facts`)：每次 Agent 给出最终回答后，后台 daemon 线程调用 LLM 提取记忆事实（格式：`「论文关注:xxx」` / `「用户偏好:xxx」`），20s 超时静默失败
2. **编码**：BGE-M3 将每条事实编码为 1024 维 L2 归一化向量
3. **存储**：upsert 到 Qdrant 新集合 `user_memory`（单路 dense 向量，COSINE 距离）
4. **检索** (`retrieve_long_term_memory`)：每次用户提问时，用 query 编码 → Qdrant 搜索 Top-3 → 注入 Prompt 动态尾部
5. **非阻塞设计**：记忆提取全程在独立线程中执行，图状态不等待其结果（fire-and-forget），UI 无需等待即可展示回答

---

## 四、工具函数设计与调用逻辑

### 4.1 工具清单

| 工具名 | 输入参数 | 功能 | 调用策略 |
|--------|---------|------|---------|
| `search_academic_papers` | primary_concept, secondary_concept, max_results | Arxiv 高级语法检索 + Semantic Scholar 兜底 | Agent 判断需要寻找外部新论文时调用 |
| `download_and_ingest_vision_paper` | paper_id, pdf_url | 下载→安全扫描→双路编码→Qdrant 入库 | Agent 判断需要精读某篇新论文时自动触发 |
| `search_vision_knowledge` | query, paper_id (optional) | 双路检索→RRF 融合→文本压缩→VLM 分析 | Agent 回答论文细节问题的**首选**工具 |

### 4.2 Agent SOP（标准操作流程）

```
用户提问
    |
    +-- 询问已入库论文细节？
    |   +---> search_vision_knowledge (paper_id=指定ID)
    |
    +-- 询问某个领域的前沿进展？
    |   +-- 1st: search_vision_knowledge (全局检索本地库)
    |   +-- 2nd: 若本地无结果 -> search_academic_papers
    |   +-- 3rd: 若发现相关新论文 -> download_and_ingest_vision_paper
    |         +-- 4th: 入库成功后自动 -> search_vision_knowledge
    |
    +-- 直接要求下载某篇论文？
        +---> download_and_ingest_vision_paper
```

### 4.3 download_and_ingest_vision_paper 详细流程

```
输入: paper_id, pdf_url
    |
    +--[1] 查重: Qdrant 中已存在该 paper_id? -> 返回拦截
    +--[2] 下载: requests.get(stream=True) + 3次重试 + 8KB chunk
    +--[3] 安全扫描: pikepdf 检测
    |   +-- 文件 > 50MB? -> 拒绝
    |   +-- 全局 JS 注入 (/Names/JavaScript)? -> 拒绝
    |   +-- 恶意触发器 (/OpenAction)? -> 拒绝
    |   +-- 页面脚本 (/AA)? -> 拒绝
    +--[4] 逐页双路编码:
    |   for page_num in 1..total_pages:
    |   +-- pdf2image 渲染 (单页, 防 OOM)
    |   +-- ColPali encode -> vision_embedding (GPU)
    |   +-- PyMuPDF 文本提取 + 表格还原
    |   +-- BGE-M3 encode -> text_embedding (CPU)
    |   +-- 保存页面图片 JPEG
    |   +-- Qdrant upsert (双向量 + payload)
    |   +-- torch.cuda.empty_cache()
    +--[5] 返回: 入库成功 (X 页)
```

### 4.4 search_vision_knowledge 详细流程 (含 ROI 剪枝)

```
输入: query, paper_id (optional)
    |
    +--[1] 双路 Query 编码
    |   +-- ColPali process_queries -> query_embeddings (GPU, 保留用于后续热力图)
    |   +-- BGE-M3 encode -> text_query_vector (CPU)
    |
    +--[2] Qdrant 双路检索 (pool_size = top_k x 6)
    |   +-- query_points(using="colpali_vision")
    |   +-- query_points(using="text_dense")
    |
    +--[3] RRF 融合 (k=60) -> Top-K 页面
    |
    +--[4] ROI 智能裁剪 (ColPali 热力图) ← NEW
    |   对每个检索到的页面:
    |   +-- 加载缓存 JPEG + ColPali 前向 → image_embeddings
    |   +-- get_similarity_maps_from_embeddings(query, image) → (32,32) 热力图
    |   +-- 热力图 std < 0.3×mean? → 回退全图
    |   +-- Top-5 峰值 patch → 25% 上下文扩边 → 原图像素坐标映射
    |   +-- 所有页面 ROI 总数 > 3? → Pillow 拼图为一张合成图
    |   +-- ≤ 3? → 各 ROI 独立编码为 Base64
    |   +-- torch.cuda.empty_cache() 每页释放显存
    |
    +--[5] Payload 组装
    |   +-- 表格占位符替换为 Markdown 表格
    |   +-- 裁剪 ROI Base64 编码 → vision_messages
    |   +-- 文本摘要压缩 (_summarize_payloads)
    |
    +--[6] VLM 多模态分析
    |   +-- POST :8001/v1/chat/completions
    |   +-- {model: qwen2-vl-7b-instruct, temperature: 0.3, max_tokens: 1024}
    |
    +--[7] 返回: VLM 分析报告 + 引用来源
```

---

## 五、性能优化措施

### 5.1 显存优化（核心瓶颈）

| 措施 | 技术细节 | 效果 |
|------|---------|------|
| **AWQ INT4 量化** | Qwen2.5-7B 和 Qwen2-VL-2B 均使用 AWQ 4-bit，权重仅 0.5 bytes/param | 7B 模型从 14GB→3.5GB，2B 模型从 4GB→1GB |
| **ColPali INT8 量化** | BitsAndBytes `load_in_8bit=True`，权重 1 byte/param | 3B 模型从 6GB→3GB |
| **BGE-M3 CPU 卸载** | `device="cpu"`，完全不占用 GPU 显存 | 释放 ~4.3GB GPU 显存 |
| **懒加载模式** | `_get_vision_model()` / `_get_text_model()` 仅在工具被调用时才初始化 | 空闲时不占用额外 GPU 显存 |
| **逐页渲染** | `convert_from_path(first_page=p, last_page=p)` 而非一次性渲染全部 | 杜绝长 PDF（50+ 页）的 OOM |
| **显存碎片清理** | 每页编码后执行 `torch.cuda.empty_cache()` | 防止显存碎片累积导致分配失败 |
| **vLLM 内存配额** | 文本 LLM: 0.35, 视觉 VLM: 0.30 | 预留 ~8.4GB 给 ColPali + 系统开销 |

### 5.2 推理加速

| 措施 | 技术细节 | 效果 |
|------|---------|------|
| **vLLM Prefix Caching** | `--enable-prefix-caching` + Prompt 静态→半静态→动态倒序拼接 | ≥60% KV Cache 命中率，大幅降低 Time-To-First-Token |
| **vLLM 连续批处理** | PagedAttention + continuous batching | 吞吐量提升 3-5× vs 原生 transformers |
| **ColPali ROI 视觉 Token 剪枝** | 32×32 热力图 → ROI 裁剪 + 25% 边距 + 拼图合成 | 视觉 Token 节省 ~60% |
| **文本预压缩** | VLM 前先经本地 LLM 将多页文本压缩至 ≤500 字 | VLM 文本 Token 减少 ~90% |
| **临时执行层替换** | ToolMessage 替换为 ~100 字内部备忘 | 上下文 Token 节省 ~90%，保留溯源能力 |
| **候选人池扩大** | RRF 前检索 pool_size = top_k × 6 | 召回率提升，避免异构排序截断丢失关键结果 |
| **Arxiv 高级语法** | `abs:"term1" AND abs:"term2" AND cat:cs.CV` | 精准匹配，减少无效下载 |
| **断点续传入库** | Qdrant `count()` 查询已入库页数，跳过已处理页面 | 中断后无需重新处理整个 PDF |

### 5.3 吞吐量优化

| 措施 | 技术细节 |
|------|---------|
| **批量入库** | mix_ingest.py 每批 4 页并行 ColPali 编码 |
| **流式下载** | `requests.get(stream=True, chunk_size=8192)` 避免大文件占满内存 |
| **SQLite WAL 模式** | LangGraph checkpoint 使用 WAL，支持并发读写 |
| **后台线程 + 队列** | Chainlit UI 使用 daemon thread + queue.Queue，避免 WebSocket 超时 |

### 5.4 安全加固

| 措施 | 技术细节 |
|------|---------|
| **PDF 恶意代码扫描** | pikepdf 检测 JS 注入、Launch/SubmitForm 动作、页面触发器 |
| **文件大小限制** | >50MB 直接拒绝 |
| **下载重试机制** | 3 次重试，3s 间隔 |
| **User-Agent 伪装** | 模拟浏览器请求头，避免被反爬 |

---

## 六、可扩展性与实用性评估

### 6.1 算力需求评估

#### 6.1.1 硬件配置

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| VRAM | 24 GB GDDR6X |
| CUDA Core | 10752 |
| Tensor Core | 336 (第3代) |
| Memory Bandwidth | 1008 GB/s |
| FP32 算力 | ~40 TFLOPS |
| FP16/BF16 算力 | ~80 TFLOPS |
| INT8 TOPS | ~160 (dense) / 320 (sparse) |
| INT4 TOPS | ~320 (dense) / 640 (sparse) |

#### 6.1.2 各模型算力需求分解

| 模型 | 量化精度 | 权重体积 | 每 Token 运算量 | 稳态 TOPS 需求 | 突发 TOPS (Prefill) |
|------|---------|---------|----------------|---------------|-------------------|
| Qwen2.5-7B-AWQ | INT4 | 3.5 GB | ~14 GFLOPs | **~0.3** (20 t/s) | **~28** (2000 tok) |
| Qwen2-VL-2B-AWQ | INT4 | 1.0 GB | ~4 GFLOPs | **~0.08** (20 t/s) | **~24** (3 图预填充) |
| ColPali (PaliGemma-3B) | INT8 | 3.0 GB | ~6 GFLOPs/页 | **~2-5** (批量4页) | — |
| BGE-M3 | FP32 (CPU) | 0 GB GPU | — | 0 | — |

#### 6.1.3 综合分析结论

| 指标 | 需求值 | 3090 Ti 能力值 | 利用率 |
|------|--------|---------------|--------|
| **稳态 INT4 TOPS** | < 1 | 320-640 | < 0.3% |
| **突发 INT4 TOPS** | ~30 (文本 Prefill) | 320-640 | ~5-10% |
| **稳态 INT8 TOPS** | ~2-5 (ColPali 编码) | 160-320 | ~1-3% |
| **显存占用** | ~18.4 GB | 24 GB | **~77%** ⚠️ |
| **显存带宽** | ~200-400 GB/s (稳态) | 1008 GB/s | **~20-40%** |

#### 6.1.4 关键结论

> **本项目的算力瓶颈是显存容量（77% 占用），而非 TOPS 算力（峰值利用率 < 10%）。**
>
> RTX 3090 Ti 的 Tensor Core INT4/INT8 算力远超此推理工作负载的实际需求。
> 这是因为 LLM 推理（特别是 Batch Size = 1 的自回归解码）是典型的 **Memory-Bandwidth Bound** 场景：
> 每生成 1 个 Token 需要从显存读取全部 3.5GB 权重，但每个权重仅参与 1 次矩阵乘加运算。
>
> 在 1008 GB/s 带宽下，Qwen2.5-7B-AWQ 的理论最大吞吐量为 1008 / 3.5 ≈ 288 tokens/s，
> 实际受 KV Cache 访问和计算效率影响，观测值约为 20-30 tokens/s。

**部署建议**：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| vLLM 文本 LLM GPU 利用率 | 0.35 | 保留充足显存给 ColPali |
| vLLM 视觉 VLM GPU 利用率 | 0.20-0.25 | Qwen2-VL-2B 可降低至 0.20 |
| 建议使用 2B VLM | ✅ 推荐 | Qwen2-VL-2B 比 7B 版节省 ~4GB，精度损失可接受 |
| ColPali 量化 | INT8 最低 | 可尝试 INT4 量化进一步节省 ~1.5GB |
| 批量入库批次大小 | 4 | 当前最优，增加批次有 OOM 风险 |

### 6.2 可扩展性

| 维度 | 现状 | 扩展方案 |
|------|------|---------|
| **论文领域** | 恶劣天气目标检测 | 只需修改 System Prompt 中的领域描述，无代码改动 |
| **多 GPU** | 单卡 3090 Ti | vLLM 原生支持张量并行，ColPali 可 device_map="auto" |
| **多用户** | 单用户 CLI/Web | LangGraph thread_id 隔离 + SQLite → PostgreSQL 迁移 |
| **更大模型** | 7B + 2B | vLLM 支持 pipeline parallelism，可换 14B/32B |
| **云端部署** | 本地工作站 | 所有组件均容器化（Qdrant Docker + vLLM Docker） |
| **增量更新** | 手动入库 | 可增加定时 Arxiv 轮询 + 自动入库后台任务 |
| **多语言** | 中英文论文 | BGE-M3 原生支持 100+ 语言，VLM 支持中英文图片文字 |

### 6.3 实用性评估

#### 优势

1. **全自动闭环**：搜索 → 下载 → 入库 → 研读，Agent 自主决策工具调用链路，无需人工干预
2. **视觉原生**：ColPali + VLM 解决了传统 RAG 对图表/公式/表格的"盲区"，这是学术论文的核心信息载体
3. **单卡可运行**：整个系统（4 个模型 + 向量数据库）在 24GB 消费级 GPU 上即可完整运行
4. **安全可靠**：pikepdf 恶意代码检测、断点续传、自动记忆压缩等工业级工程实践
5. **可评估**：内置 10 个测试用例的评估框架，含 Recall/Precision/MRR/NDCG + LLM-as-Judge 三重评判

#### 局限性

1. **实时性受限**：VLM 单次分析需要 5-15 秒，不适合即时交互场景
2. **ColPali 生态依赖**：依赖 colpali-engine 库，transformers 升级可能触发兼容性问题（当前需 Monkey Patch）
3. **语言模型能力边界**：Qwen 系列在英文论文理解上弱于 GPT-4/Llama-3，可能影响分析深度
4. **单线程 Agent**：同一 thread_id 只能串行处理，不支持同一用户并发提问

#### 改进方向

1. **ColPali INT4 量化**：进一步降低视觉编码的显存占用（预计节省 1.5GB），为更大批次入库腾出空间
2. ~~**缓存机制**~~：已实现 — vLLM Prefix Caching + 缓存友好 Prompt 排序，≥60% KV Cache 命中率
3. ~~**记忆系统重构**~~：已实现 — 四层分层记忆 + Qdrant `user_memory` 外部化长期记忆
4. ~~**视觉 Token 优化**~~：已实现 — ColPali ROI 智能裁剪 + 拼图合成，节省 ~60% 视觉 Token
5. **混合模型策略**：文本推理可换用 API 服务（如 DeepSeek），本地仅保留视觉管道
6. **流式 VLM 输出**：支持 VLM 结果的 Token 级流式展示，改善用户体验
7. **多模态引用溯源**：VLM 回答中标注具体引用的 ROI 区域坐标，进一步提升可信度

---

## 附录 A：完整显存预算表

```
3090 Ti 24 GB GDDR6X
+-- vLLM Text (Qwen2.5-7B-AWQ, port 8000) ........ 8.4 GB  (gpu-memory-util 0.35)
|   +-- 模型权重 (INT4) ............................ 3.5 GB
|   +-- KV Cache (max-model-len 8192) .............. 2.5 GB
|   +-- vLLM 运行时开销 ............................ 2.4 GB
|
+-- vLLM Vision (Qwen2-VL-2B-AWQ, port 8001) ...... 6.0 GB  (gpu-memory-util 0.25)
|   +-- 模型权重 (INT4) ............................ 1.0 GB
|   +-- ViT 视觉编码器 ............................. 0.5 GB
|   +-- KV Cache (max-model-len 12288) ............. 2.0 GB
|   +-- vLLM 运行时开销 ............................ 2.5 GB
|
+-- ColPali (PaliGemma-3B, INT8, 按需加载) ......... 4.0 GB
|   +-- ViT (SigLIP 448px) ......................... 0.8 GB
|   +-- Gemma-2B 文本解码器 ........................ 2.2 GB
|   +-- 激活值/中间张量 ............................ 1.0 GB
|
+-- 系统预留 ........................................ 1.0 GB
|
+-- 已使用 .......................................... 19.4 GB
+-- 剩余 ............................................ 4.6 GB  (安全余量)
```

## 附录 B：磁盘存储统计

| 类别 | 占用 | 说明 |
|------|------|------|
| 模型文件 | ~39 GB | 含冗余 bf16 模型（约 15GB 可清理） |
| Qdrant 向量数据 | ~340 MB | 21 篇论文，约 200 页 |
| PDF 原始文件 | ~172 MB | data/ + data/downloads/ |
| 图片缓存 | ~121 MB | image_cache/ (页面渲染 JPEG) |
| 对话历史 | ~1.5 MB | SQLite checkpoint |
| **合计** | **~40 GB** | |

---

*文档生成日期：2026-05-03*
*项目路径：/home/c2216-3090/disB/hyh/AI*
