# LLM×KT 可解释性下游：实验设计 v1（2026-10-03）

支线：LLM+KT 即插即用（冻结 KT 模型、纯烧 token、基本不花钱）。分支 `feat/llm-kt-plugin`。

## 1. 任务定义

给定**已训练并冻结**的知识追踪模型 M（checkpoint 只读）和学生 s 的交互序列，用 LLM（glm-5.3-flash / kimi-for-coding）生成自然语言诊断报告 D(s,M)，要求：

- **忠实（Faithful）**：报告中每个事实性陈述可追溯到 M 的内部信号或计算结果，不允许幻觉数字/KC。
- **与模型一致（Consistent）**：LLM 对学生状态的判断（"掌握/薄弱"）与模型内部信号方向一致。
- **可操作（Useful）**：诊断附带基于证据的学习建议。

**与竞品的区隔**：RAG-KT/CLST 让 LLM 当预测器（替代 KT 模型）；IKT-EIA 做解释但对齐机制是训练出来的。我们：预测仍由冻结深度模型完成，LLM 只做**下游解读层 + 硬校验的忠实性**——即插即用、零训练、只花 API token。

## 2. 系统设计（四层）

### 2.1 信号层 `signals.py`（SignalExtractor）
从 run_dir 恢复模型（复用各模型 trainer 的 `build_components` + `CheckpointManager.load_weights`），对测试集前向，按用户导出：

- 通用（四个模型都有）：逐步预测 p_t、交互序列（question_id/skill_id/correct）、KC 聚合统计（尝试数、正确率、模型预测均值）。
- DKVMN 专属：最终记忆矩阵 Mv[-1]（[size_m, dim_s]）+ 各 KC 嵌入的读出 → `mastery_proxy(c) = σ(W·read(Mv[-1], kc=c))`，即模型视角"该生此刻对 KC c 的准备度"（DKVMN 原论文的 memory profile 思路）。
- SAKT/AKT 注意力 hook：v1 不做（层结构各异，hook 脆弱），v2 视需要加。

### 2.2 证据包层 `evidence.py`（EvidencePack）
把信号压缩成 LLM 可引用的结构化证据，每条带 id：

- E1 KC 统计表：每行 {KC 名, 尝试数, 正确率, 模型预测均值, DKVMN mastery(若有), 最近5次正确率}
- E2 异常事件：|p_t − y_t| 最大的 top-3 步（"模型意外判错/意外判对"的时刻，含题目/KC）
- E3 全局：序列长度、总体正确率、模型在该生上的 AUC/Brier
- 数值全部四舍五入到固定小数位，id 用 E1.3 / E2.1 格式。

### 2.3 LLM 层 `llm_client.py` + `prompts.py`
- OpenAI 兼容接口；base_url/key 从 `/root/.llm-keys.env`（绝不入库）。
- 主模型 glm-5.3-flash，对照 kimi-for-coding（thinking budget 低档）。温度 0。
- 输出强制 JSON：`{summary, claims: [{id, type∈{掌握,薄弱,预测,趋势,建议}, statement, evidence_ids[], numbers[]}], weakest_kcs[]（排序）}`。
- 解析失败/校验失败自动重试 ≤2 次。

### 2.4 忠实性校验层 `faithfulness.py`
- **硬校验**（不通过即违规）：claim 引用的 evidence id 存在；statement 中的数字 ∈ 该证据的数值集合（容差 ±0.01）；提到的 KC 名 ∈ 数据集 KC 表；weakest_kcs ⊆ E1 的 KC。
- **一致性统计**（汇总指标）：
  - 排序一致：LLM 的 weakest_kcs 排序 vs 模型 mastery 升序 / 正确率升序的 Spearman ρ。
  - 方向准确：LLM 标"薄弱"的 KC 平均 mastery 是否低于标"掌握"的。
  - 幻觉率：硬校验失败的 claim 占比。

## 3. 实验计划

- 数据：assistments09 fold0 测试集（新版流水线重跑产物，3852 用户 / 123 KC）。
- 模型：DKT / DKVMN / SAKT / AKT 四个 checkpoint（同数据同折）。
- 学生抽样：按（序列长度 × 正确率）分层抽 30 人 + 3 个极端案例（模型高 AUC 生 / 低 AUC 生 / 短序列生）。
- 对比条件：
  1. **消融 A**：证据包含模型内部信号（p_t 轨迹 + mastery） vs 仅原始做题记录 → 量化"内部信号的价值"。
  2. **消融 B**：GLM flash vs Kimi → 结论对 LLM 的稳健性。
  3. **跨模型一致性**：同一学生四个模型的证据包 → 诊断方向是否一致（模型分歧本身是发现）。
- 汇总指标：硬校验通过率、Spearman ρ、方向准确率、幻觉率、（消融 A 的）两组指标差。
- 成功标准（预注册）：硬校验 ≥95%、平均 ρ ≥0.6、两 LLM 方向一致率 ≥80%。达不到则回到证据包设计迭代。

## 4. 成本核算（"基本不花钱"约束）

- 训练：本地 4060，assistments09 全程 <30 分钟，0 元。
- LLM：每生证据包约 1.5-3K token 输入 + 1K 输出；30 生 × (2 LLM × 2 消融) ≈ 120 次调用 ≈ 50 万 token 级别——GLM/Kimi 均 Coding Plan 包含，边际 0 元。

## 5. 交付物

1. `research/llm_explain/` 五个模块 + `run_explain.py` CLI。
2. 3-5 份学生诊断报告样例（markdown，含证据引用）。
3. 汇总指标表 + 简短分析（research/ 下 markdown）。
4. 全部 checkpoint 留在 WSL `runs/normal/`。

## 6. 风险与备选

- AKT 前向含 Rasch 嵌入与多层注意力，v1 只取通用信号（不 hook）。
- LLM 不守 JSON：重试 + jsonschema 校验；仍失败则记为解析失败样例（本身是指标）。
- DKVMN mastery_proxy 的合理性：校准检查——mastery 与该 KC 历史正确率的点二列相关，若 ρ<0.2 则弃用该信号并在报告中说明。
