# LLM + KT 相关论文清单（支线调研底稿）

收集日期：2026-10-03。用途：LLM+KT 即插即用支线（冻结 KT 模型、LLM 前端语义、纯烧 token）的切入点讨论底稿。
来源：网络检索结果整理，标注【转述】= 摘要级信息未逐篇读全文；【已核实】= 已读摘要页确认。竞品风险分三档：⚠️高（正面撞车）/ ◐中（部分重叠）/ ○低（可借用组件或背景）。

## A. 直接竞品 / 最近邻（必读，决定切入点位）

1. **RAG-KT: Cross-platform Explainable Knowledge Tracing** ⚠️【已核实摘要】
   - ACL 2026 Findings（Zhiyi Duan, Hongyu Yuan, Rui Liu）。https://aclanthology.org/2026.findings-acl.257/
   - 做法：跨平台 KT 视为"可靠上下文约束下的 LLM 推理"，Question Group 抽象统一多源数据，检索增强 + 冻结 LLM 直接出预测与可解释诊断。三个公开基准，宣称跨平台精度与鲁棒性提升。
   - 与本支线的关系：**最接近的已发表工作**（frozen LLM、可解释、免微调）。区隔点：它是"LLM 当 KT 器"（替代深度 KT 模型），我们是"深度 KT 模型保留、LLM 做前端语义增强"。写 related work 必须正面对比：保留深度 KT 的序列建模能力 vs LLM 直接预测的精度上限。

2. **CLST: Cold-Start Mitigation in KT by Aligning a Language Model** ◐【已核实摘要】
   - arXiv:2406.10296，JEDM 2025 发表。https://arxiv.org/abs/2406.10296
   - 做法：KT 转自然语言任务，微调生成式 LLM 当知识追踪器；数学/社会/科学三科，<100 学生小数据场景显著提升。
   - 关系：冷启动卖点重合，但它要微调（烧卡），我们免训练。可作为"为什么要免训练"的对照。

3. **IKT-EIA: Interpretable Knowledge Tracing via Explicit–Implicit Alignment** ◐【转述】
   - Knowledge-Based Systems 2026。https://www.sciencedirect.com/science/article/abs/pii/S0950705126007975
   - 做法：双模型框架，深度 KT 预测 + LLM 解释，显式-隐式对齐连接两者。
   - 关系："深度 KT + LLM 解释"的组合重合；需读全文确认是否免训练、解释是否 grounded。

4. **A Framework for Interpretable Knowledge Tracing and Explanation** ◐【转述】
   - arXiv:2506.16982。https://arxiv.org/html/2506.16982v1
   - 要点：指出 LLM 直接预测/总结会幻觉、无精度保证——可引作"LLM 不该当预测头"的论据。

## B. 语义增强线（朋友说的"语义增强"已有谱系）

5. **SEEP: Semantic-Enhanced Question Embeddings Pre-training** ◐【转述】
   - Information Sciences 2022（Wang et al.，引用 33+）。https://www.sciencedirect.com/science/article/abs/pii/S0020025522011355
   - 语义预训练题目嵌入注入 KT。用的是早期 PLM 预训练嵌入——**现代 LLM API embedding + 冻结 KT 的组合仍是空白**。

6. **Attentive Pre-training Question Embeddings for KT** ○【转述】
   - Neural Networks 2026。https://dl.acm.org/doi/10.1016/j.neunet.2026.109194
   - SEEP 后续，语义检索增强 + 概念标签引导异构图。

7. **Enhancing Deep KT with Semantic Embeddings**（UCalgary 学位论文 2026）○【转述】
   - https://ucalgary.scholaris.ca/items/26bfd93b-63f2-4e80-a798-b9a527d0907c

## C. 冷启动线（LLM 语义最自然的用武之地）

8. **LOKT: Mitigating Cold Start in KT Using LLMs** ◐【转述】
   - arXiv:2410.12872。LLM option-weighted KT，新题/新生冷启动。

9. **Mitigating Cold-Start Problems in KT with LLMs**（CIKM 2024）◐【转述】
   - https://dl.acm.org/doi/10.1145/3627673.3679664

10. **Enhancing KT Robustness for New Questions** ○【转述】
    - arXiv:2512.07179。题目冷启动鲁棒性的实证分析，可借其 setting。

11. **MAML-KT**（arXiv:2603.00137）○【转述】少样本元学习冷启动，非 LLM。

## D. Q-matrix / 知识点标注线（LLM 标注的已有工作）

12. **A Framework for Human-AI Q-Matrix Refinement** ◐【转述】
    - LLM 生成候选 Q-matrix + 人工校验。https://www.semanticscholar.org（检索命中，需定位具体论文）

13. **Research on Question Bank Construction Based on LLM**（ACM 2025）○【转述】LLM 题库构建与自动标注。

14. **PrereqGen**（agentic 多阶段框架）○【转述】AQG + 概念间先修关系建模。

## E. 综述与直接预测线

15. **A Systematic Review of KT and LLMs in Education** ○【转述】
    - arXiv:2412.09248。LLM 可为 KT 生成题目特征、解冷启动。
16. **Explainable Few-Shot KT**（Li et al.，引用 47+）◐【转述】LLM 少样本追踪知识状态。
17. **Exploring Interpretable Deep KT Through LLM Prompts**（IEEE，Okechukwu 2026）◐【转述】
    - 用教育 AI 助手 prompt 解释 DKT 预测——与我们"解释冻结 KT 模型"的想法直接重合，需读全文确认深度。
18. **Multi-Agent LLM Stealth Assessment**（arXiv 2026-06）○【转述】
19. **Enhancing KT with LLMs**（HAL，Baig 2025）○【转述】
20. **Learning Path Recommendation Enhanced by KT and LLM**（MDPI Electronics 2025）○【转述】下游导学应用，与工作区"智能导学"主题呼应。

## 空白点观察（讨论用初步判断）

- 【判断】"training-free / 即插即用 + 冻结深度 KT backbone + LLM 语义前端"的组合在检索中没有直接命中：语义增强线（SEEP 系）都是"预训练嵌入+重训 KT"；LLM 直接预测线（RAG-KT/CLST）都绕开了深度 KT 模型；解释线（IKT-EIA/IEEE prompt）不改进预测精度。
- 【判断】候选切入方向（按与聊天设定的贴合度）：
  1. **冷启动题目映射**：LLM 给新题标注 KC/语义近邻 → 免训练映射到冻结 KT 已见的 embedding 空间（新题不重训即可推理）。
  2. **语义图结构修正**：LLM embedding 重建题目/KC 关系图 → 图类 KT（GKT 等）推理时换图不换权重。
  3. **LLM 解释器**：对冻结 KT 的预测轨迹生成 grounded 诊断报告（与 IKT-EIA/RAG-KT 正面竞争）。
- 【判断】风险：方向 3 竞品密度最高；方向 1/2 的"免训练"卖点最硬（有 CLST 微调作对照），且完全符合"烧 token 不烧卡"约束。
