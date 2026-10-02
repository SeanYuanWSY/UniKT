# LLM×KT 可解释性下游：实验设计 v2（2026-10-03）

v1 经三视角并行评审（KT 领域 / LLM 评测工程 / 统计方法）+ 跨领域调研修订。变更标记【v2】。

## 0. 定位与术语（修订）

- 架构：**"先审计、后翻译"两段式**（Credit Risk 2026 实证支持：LLM 转述给定归因近乎完美 Overlap≈1，自主推断归因 0.14-0.44）。程序化提取冻结模型的信号并算成可审计排名/统计 → LLM 只负责翻译成自然语言诊断与建议。
- 术语：程序对账指标称 **verifiability/grounding**；"faithfulness"仅用于扰动敏感性实验的结论。
- 与竞品区隔：RAG-KT/CLST = LLM 参与预测；IKT-EIA = 训练对齐；Okechukwu (IEEE) 需全文精读后逐点对比（P0 待办）；本工作 = post-hoc 解读冻结模型内部信号 + 可校验证据协议 + 留出段有效性。【v2】

## 1. 任务定义

冻结模型 M、学生 s、交互序列按时间切分：**前 80% 为 evidence 段（进证据包），后 20% 为 holdout 段（绝不进 prompt）**。【v2 核心】

LLM 输入 = evidence 段的证据包；输出 = 结构化 JSON：
- `claims[]`: {id, type∈{掌握,薄弱,预测,趋势,建议}, statement, citations:[{evidence_id, row, field, value}]}
- `holdout_predictions[]`: 对留出段将出现的每个已学 KC，预测其正确率（0-1）——主评测锚点
- `weakest_kcs[]`: top-3 薄弱 KC（限定 evidence 段出现过的 KC）

## 2. 信号层（四模型统一 + DKVMN 专属）

1. 通用：evidence 段逐步预测 p_t、KC 统计（尝试/正确率/模型预测均值/最近5次）。
2. **readiness 扫描（v2 新增，四模型统一、零 hook）**：用 evidence 段历史做条件，对每个候选 KC 的虚拟下一题前向，`readiness(c) = M 的预测概率`。DKT/DKVMN/SAKT/AKT 各自前向接口不同，由 restore 层适配。
3. DKVMN 专属（次要分析）：Mv[-1] 读出 mastery_proxy，声明为复用模型权重的自制探针；校准 = 与该 KC **留出段**表现的相关（非历史正确率），逐生计算聚合并报 CI，ρ<0.3 弃用该列。

## 3. 证据包层

- E1 KC 表（markdown 表格 + 显式行 id + 每格可寻址）：KC 名 | 尝试 | 正确率 | 模型预测均值 | readiness | (DKVMN mastery)。**行序固定按 KC id**（防 LLM 照抄表序）。【v2】
- E2 异常事件 top-3（|p_t − y_t| 最大步）。
- E3 全局统计（序列长度、正确率、模型在该生上的 AUC/Brier——均在 evidence 段内计算）。
- 数值统一四舍五入 2 位小数；claim 校验要求**规范化后精确相等**（60%→0.6 归一化），不设 ±0.01 容差。【v2】

## 4. LLM 层（工程修订）

- glm-5.3-flash 主 / kimi-for-coding 对照。温度 0；max_tokens 充足（kimi 为 reasoning-only：只解析 `content`，`finish_reason!=stop` 判截断计失败；剥离 markdown 围栏取首个配平 JSON，jsonrepair 兜底）。【v2】
- few-shot 2 例（合成学生，非抽样池内）：数字-证据绑定示例 + "证据不足"拒答示例。
- 合法出口："证据不足以判断 X"；统计拒答率。
- **重试只救 schema/解析失败（不附带忠实性错误详情）；忠实性违规按首试记录为最终结果**；两套指标并报。【v2】
- 每次调用归档 JSONL：request/response 原文、model 版本串、request_id、时间戳、token 计数、全部尝试。【v2】
- 证据包含"禁止外部世界知识（如'研究表明'）"指令；建议类 claim 的 factual preface 可校验、主观部分单独统计。

## 5. 评估体系（预注册）

**主终点（单一）**：GLM×含信号条件下，holdout_predictions 与留出段真实 KC 正确率的逐生 Spearman ρ 的 Fisher-z 加权平均（w_i=k_i−3），单侧 95% CI 下界 > 抄表基线的同指标 + 0.05。k_i≥10 的学生进主分析。【v2】

条件矩阵（其余全部次要/描述性）：
- **抄表基线（copy-baseline）**：程序直接用 E1 排序/外推生成报告——上界参照，证明 LLM 增量（报告质量维度人工评）与 extraction loss。
- **模板基线**：同证据包确定性模板报告（零幻觉 ρ=1）——LLM 价值 = 模板做不到的交叉推理/建议具体性。
- **消融 A**：含 readiness/内部信号 vs 仅行为统计（同生配对，Δz Wilcoxon 符号秩 + cluster bootstrap CI；McNemar 检通过率）。
- **置换对照**：readiness 列生内随机置换 KC 后填回（格式长度不变）——若指标不掉，说明 LLM 没读信号（同时是最强证伪测试）。
- **扰动探针**（各 5-10 包）：换生证据包（敏感度）、删 KC 行（comprehensiveness）、矛盾行、缺失值（编造检测）。
- **接地性**：单元格级绑定校验（evidence_id+row+field+value 精确匹配）→ 违规率（claim 级、首试口径，学生级 cluster bootstrap CI）；KC 名 ∈ 该生 E1 行集合；NLI 交叉裁判（Kimi 判 GLM 报告、GLM 判 Kimi，supports/contradicts/unverifiable）+ 人工抽 50-100 条校准。
- **稳定性**：10% 格子重复调用 3 次（温度 0），claim Jaccard + weakest τ。
- 跨 LLM 一致性：top-3 Jaccard + Cohen's κ（给机会基线），描述性。
- 跨模型一致性：先量化证据包相似度再解释报告一致率；高分歧学生做案例分析。

**样本**：n=150（按序列长度×正确率分层配额），**dev=30 / holdout=120 拆分**：dev 上迭代证据包与 prompt；冻结后在 holdout 单次确认运行，标准不可再改；迭代轮次全披露。极端案例 3 人单独案例分析、不入推断。【v2】

**多重比较**：主终点单一；次要 ≤6 项 Holm；其余 BH-FDR 或只报 CI。【v2】

## 6. 成本

- 训练：4060 本地 0 元（已完成 4/4 checkpoint）。
- LLM：150 生 × (2 LLM × ~3 条件 + 探针) ≈ 1000 次调用 ≈ 4-6M token，Coding Plan 内，边际 0 元。

## 7. 代码结构（research/llm_explain/）

```
restore.py        # run_dir → (model, test_loader, id_mappings)，四模型适配
signals.py        # 通用信号 + readiness 扫描 + DKVMN mastery
evidence.py       # 证据包构建（80/20 切分、行 id、markdown 表）
llm_client.py     # OpenAI 兼容、JSONL 归档、kimi 解析规则、重试策略
prompts.py        # 系统提示 + few-shot（合成）
faithfulness.py   # 单元格校验、NLI 裁判调用、扰动探针构造
run_explain.py    # CLI：sample → evidence → llm → verify → 报告/指标
analysis.py       # Fisher-z 聚合、bootstrap、Wilcoxon/McNemar、图表
```

## 8. 里程碑

1. restore + signals（四模型 readiness 扫描跑通，AKT 训完即入组）
2. evidence + prompts（dev 30 生迭代 ≤3 轮）
3. holdout 120 单次确认 + 探针 + 基线
4. 分析与报告样例 3-5 份 + 指标表
5. 失败或未达标 → 全部轮次披露，定位瓶颈（信号质量 vs LLM 翻译 vs 证据包设计）
