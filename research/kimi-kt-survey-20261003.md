# 调研报告：冻结 KT 模型 + LLM 跨数据集知识点语义对齐 → 零训练迁移

> 调研日期：2026-10-03。标注规则：【已核实】= 调研过程中实际打开论文/文档页面核读过；【转述】= 仅有搜索摘要或二手引用，未核到原文。
> TL;DR：**"把一个训练好的深度 KT 模型完全冻结、仅靠 KC 语义对齐零训练迁移到新数据集"在 2020–2026 文献中没有先例**。最接近的是 RAG-KT（ACL 2026 Findings，做了显式 KC 对齐 + 零训练跨平台，但它用检索+冻结 LLM 绕开了深度 KT 模型本身）和 LRCD/LCST（认知诊断侧的语义对齐零样本迁移，但冻结的是文本编码器/通用 LLM 而非教育预训练模型）。这个组合是真实空白。

---

## 1. 跨数据集/跨平台 KT 迁移文献（2020–2026）

按年份排序。维度：方法一句话 / 有无 KC 对齐环节 / 代码 / 迁移设置（zero-shot 还是需要目标域数据）。

### 2020–2021：奠基期（尚无真正跨数据集迁移）

1. **Pre-training Question Embeddings for KT**（IJCAI 2020）[arXiv:2012.05031](https://arxiv.org/abs/2012.05031)
   用题目难度+题目-KC 二部图预训练题目嵌入再喂给 DKT。KC 对齐：无。代码：未找到。需目标域 fine-tune。【已核实】

2. **LM-KT**（ACL-IJCNLP 2021）[arXiv:2106.04262](https://arxiv.org/abs/2106.04262)
   fine-tune GPT-2 做 KT，输入题目文本，可泛化到未见题目（题目级 zero-shot，非跨数据集）。KC 对齐：无。【已核实】

3. **ATKT**（ACM MM 2021）[arXiv:2108.04430](https://arxiv.org/abs/2108.04430)
   交互嵌入上加对抗扰动提升单数据集鲁棒性。注意：是对抗训练但**不是** domain adaptation。KC 对齐：无。【已核实】

### 2022：Domain Adaptation 正式引入 KT

4. **AdaptKT**（WSDM 2022）[ACM](https://dl.acm.org/doi/10.1145/3488560.3498379) / [作者 PDF](https://base.ustc.edu.cn/pdf/2021/SongCheng-WSDM2022.pdf)
   三阶段迁移：题目文本相似度选源域实例 → MMD 对齐两域知识状态分布 → **因两域 KC 数不同必须换掉输出层**，冻结主干只 fine-tune 输出层。KC 对齐：部分（题目文本嵌入相似，无显式 KC 名称映射）。代码：未找到。需目标域交互数据。**最经典的"冻结主干跨数据集 KT 迁移"，其"换头"方案正是我们想法要绕开的痛点**。【已核实】

5. **CrossLing**（AAAI 2022）[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/20735)
   对抗 DA 学域不变+域私有表征，大编程数据集→小编程语言数据集。教育领域少见的成熟 adversarial DA 实例，但非标准 KT、无 KC 对齐。代码：未找到。需目标域小数据。【已核实】

### 2024–2025：跨域/冷启动 KT 爆发

6. **Explainable Few-shot KT**（arXiv 2024 / Frontiers of Digital Education 2025）[arXiv:2405.14391](https://arxiv.org/abs/2405.14391)
   认知理论引导的 LLM prompting，从少量学生记录做 KT+解释。KC 对齐：无。代码：[GitHub](https://github.com/LeavesLi1015/Explainable-Few-shot-Knowledge-Tracing)。学生级 few-shot prompting，无训练，但无显式跨数据集协议。【已核实】

7. **AEGOT-CDKT**（World Wide Web 期刊 2025）[DOI](https://repository.eduhk.hk/en/publications/a-cross-domain-knowledge-tracing-model-based-on-graph-optimal-tra/)
   自编码器学两域嵌入 → Graph Optimal Transport 跨域嵌入对齐 → 源域嵌入训练 LSTM 预测目标域。KC 对齐：部分（嵌入分布级 GOT 对齐，非语义映射）。代码：未找到。需目标域无标签数据。【已核实】

8. **TransKT**（IJCAI 2025 Oral）[arXiv:2505.13489](https://arxiv.org/abs/2505.13489) / [IJCAI](https://www.ijcai.org/proceedings/2025/823)
   **zero-shot LLM 提问构建跨课程概念图**（显式链接不同课程相关概念）+ GCN 迁移 + 单/跨课程知识状态对比对齐。KC 对齐：**有**（LLM 建跨课程概念关联图，显式对齐代表）。代码：[GitHub](https://github.com/DQYZHWK/TransKT)。需两课程联合训练，非目标域零训练。【已核实】

9. **OCLCKT**（Knowledge-Based Systems 2025）[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125016557)
   用源/目标学科的重叠学生做知识状态映射+学科级对比学习。KC 对齐：无（学生状态层面）。需重叠学生。【已核实】

10. **csKT**（ESWA 2025）[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417424028550)
    kernel bias + cone attention 缓解新学生冷启动。KC 对齐：无；代码已并入 [pyKT](https://pykt.org)。单数据集内，非跨数据集。【转述】

11. **CLST**（JEDM 2025）[JEDM PDF](https://jedm.educationaldatamining.org/index.php/JEDM/article/download/854/264)
    把 KT 写成语言任务（KC 名称文本序列→yes/no），LoRA 微调生成式 LLM，<100 学生即可。KC 对齐：部分（KC 以自然语言名称进模型，语义天然可迁移，但仍逐数据集微调）。【已核实】

12. **LMM 抽取 KC**（EDM 2025）[EDM](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.long-papers.170/) / [arXiv:2409.20167](https://arxiv.org/html/2409.20167v2)
    zero-shot 用多模态大模型从题目文本/图片自动抽 KC。KC 对齐：有（上游环节——自动 KC 标签可作为统一不同数据集 KC 体系的工具）。【已核实】

13. **Adaptive Knowledge Transfer for Cross-Disciplinary Cold-Start KT**（arXiv 2025-11）[arXiv:2511.20009](https://arxiv.org/abs/2511.20009)
    源学科预训练+MoE 门控跨域映射+对抗判别器，20 个极端跨学科冷启动场景。KC 对齐：无。仍需少量重叠实体信号。【已核实】

14. **Cold Start Problem 评估研究**（arXiv 2025-05）[arXiv:2505.21517](https://arxiv.org/abs/2505.21517)
    严格"往届学生训练、全新学生测试"协议下 DKT/DKVMN/SAKT 性能大幅下滑。跨学生而非跨数据集，但**是跨数据集迁移研究的最佳动机证据**。【已核实】

### 2026：跨平台/zero-shot 成为明确目标

15. **MAML-KT**（arXiv 2026-02）[arXiv:2603.00137](https://arxiv.org/abs/2603.00137)
    MAML 元学习，新学生 1–2 步梯度适配。KC 对齐：无（论文明确指出未见 skill 是性能下跌主因）。代码：[GitHub](https://github.com/Indronil-Prince/MAML-KT)。【已核实】

16. **MCKT**（Electronics 2026）[DOI](https://www.mdpi.com/2079-9292/15/3/642)
    跨课程 KT：属性关系图+注意力融合跨课程学生表征。KC 对齐：部分（图表征层）。需目标课程数据联合训练。【已核实】

17. **DTransKT**（WWW 2026）[DOI](https://doi.org/10.1145/3774904.3792501)
    即插即用框架：跨学科图匹配抽 meta-skill 表征对齐学生特征 + PLM 文本 meta-语义。**direct transfer 设置下**使 7 个主流 KT 模型平均 ACC +14.2%、AUC +4.5%。KC 对齐：有（图节点匹配+PLM 双通道）。代码：[GitHub](https://github.com/Dual-KT/DTransKT)。最接近"低目标域成本"的可训练方案，但框架本身需源域训练。【已核实】

18. **RAG-KT**（ACL 2026 Findings）[ACL Anthology](https://aclanthology.org/2026.findings-acl.257/) / [arXiv:2604.10960](https://arxiv.org/abs/2604.10960)
    检索增强范式：跨平台异构知识图 + **Question Group 抽象层统一不同平台题目/KC** + 多视图检索 + 冻结 LLM zero-shot 预测。KC 对齐：**有且最完整**（词汇匹配+LLM 语义对齐两阶段，专家标注 ground truth 评估对齐准确率 92.7%–96.5%）。代码：未找到。全冷启动实验（Eedi，知识库零目标平台信息）仍最优。**离本想法最近，详见第 3 节**。【已核实，细读正文】

19. **LT-MKT**（arXiv 2026-08）[arXiv:2608.24005](https://arxiv.org/abs/2608.24005)
    LLM 融合题目/概念文本构建多域层级图+认知负载建模。KC 对齐：部分。多域联合训练，需所有域数据。【已核实】

### 本方向小结：与本想法的距离排序

| 排名 | 工作 | 差距 |
|---|---|---|
| 1 | RAG-KT (ACL 2026 F.) | 唯一做到目标平台零训练+显式 KC 对齐；但绕开了深度 KT 模型，不是迁移已训练模型权重 |
| 2 | DTransKT (WWW 2026) | direct transfer 增强现有 KT 模型，但框架需源域训练 |
| 3 | AdaptKT (WSDM 2022) | 冻结主干迁移，但 KC 数不同被迫换输出层并用目标域数据 fine-tune |
| 4 | AEGOT-CDKT (2025) | 嵌入级 GOT 对齐，仍需目标域无标签数据 |

**确认空白点**：① 无人在"冻结深度 KT checkpoint + 仅 KC 语义对齐 + 目标域零样本"组合上工作；② KC 对齐方法分裂（RAG-KT 显式/TransKT·DTransKT 隐式/AdaptKT·AEGOT 嵌入级），**无公认跨数据集 KC 对齐基准与指标**；③ 无统一"源域训练→目标域 zero-shot 评测"协议，跨论文数字不可比；④ DANN 式 domain-adversarial KT 基本空缺。

---

## 2. 基准 AUC 数字表（DKT / DKVMN / SAKT / AKT / GKT）

**口径警告（重要）**：同一模型在同一数据集上，不同论文因预处理（采样、序列截断、预测粒度 question-level vs KC-level）差异可达 0.02–0.09 AUC。**只能在同一来源行内横比，不能跨来源横比**。

### 2.1 pyKT 官方 benchmark（来源：pyKT 论文 [arXiv:2206.11460v5](https://arxiv.org/html/2206.11460v5)，NeurIPS 2022 Datasets & Benchmarks）

pyKT 口径：question-level all-in-one 预测、maxlen=200、5-fold 交叉验证（AUC±std）。**pyKT 官方表只含 AS2009/AS2015，不含 AS2012/AS2017/EdNet**。【已核实，核读原文表格】

| 模型 | ASSIST09 AUC | ASSIST15 AUC |
|---|---|---|
| DKT | 0.7541±0.0011 | 0.7271±0.0005 |
| DKVMN | 0.7473±0.0006 | 0.7227±0.0004 |
| SAKT | 0.7246±0.0017 | 0.7114±0.0003 |
| GKT | 0.7424±0.0021 | 0.7258±0.0012 |
| AKT | **0.7853±0.0017** | **0.7281±0.0004** |

（pyKT 另报 KC-level all-in-one：AS2009 DKT 0.7419 / DKVMN 0.7330 / SAKT 0.7085 / GKT 0.7227 / AKT 0.7650。）

### 2.2 ASSISTments2012（GKT 无经典权威出处）

| 模型 | DIMKT (SIGIR'22) 表3 | LPKT (TKDE'22 扩展版) 表4 | GIKT (ECML'20) 表2 |
|---|---|---|---|
| DKT | 0.7304 (ACC 0.7375) | 0.7276 (ACC 0.7335) | 0.7286 |
| DKVMN | 0.7255 (ACC 0.7353) | 0.7188 (ACC 0.7285) | 0.7283 |
| SAKT | 0.7279 (ACC 0.7364) | 0.7238 (ACC 0.7320) | — |
| AKT | 0.7841 (ACC 0.7638) | 0.7706 (ACC 0.7515) | — |
| GKT | — | — | — |

来源：[DIMKT PDF](http://staff.ustc.edu.cn/~qiliuql/files/Publications/Shuanghong-Sheng-SIGIR22.pdf)、[LPKT TKDE PDF](http://staff.ustc.edu.cn/~qiliuql/files/Publications/Shuanghong-Shen-TKDE22.pdf)、[GIKT arXiv:2009.05991](https://arxiv.org/abs/2009.05991)。【已核实，核读原文表格】

### 2.3 ASSISTments2017

| 模型 | AKT 原文 (KDD'20) 表2 | LPKT (TKDE'22) 表4 | DGAKT (arXiv:2507.18668) 表2 |
|---|---|---|---|
| DKT | 0.7263±0.0054 | 0.7136 (ACC 0.6895) | 0.7005 (ACC 0.6789) |
| DKVMN | 0.7073±0.0044 | 0.6978 (ACC 0.6826) | 0.6952 (ACC 0.6795) |
| SAKT | 0.6569±0.0027 | 0.6733 (ACC 0.6759) | 0.6742 (ACC 0.6575) |
| AKT | 0.7702±0.0026 ⚠️ | 0.7501 (ACC 0.7080) | 0.7263 (ACC 0.6891) |
| GKT | — | — | 0.7125 (ACC 0.6858) |

⚠️ AKT 原文的 0.7702 是带 Rasch embedding 的 **AKT-R**，且为原文自有预处理；LPKT 复跑 AKT 仅 0.7501，DGAKT 复跑仅 0.7263。引用必须注明版本。
来源：[AKT arXiv:2007.12324](https://arxiv.org/abs/2007.12324)（注意：常见误引的 2006.05491 是别的论文）、LPKT 同上、[DGAKT arXiv:2507.18668](https://arxiv.org/pdf/2507.18668)。【已核实】

### 2.4 EdNet-KT1（采样差异最大的数据集）

| 模型 | LPKT (TKDE'22) 表4 | GIKT (ECML'20) 表2 | DGAKT (2025) 表2 | SAINT (L@S'20) |
|---|---|---|---|---|
| DKT | 0.6663 (ACC 0.6812) | 0.6822 | 0.6771 (ACC 0.6868) | — |
| DKVMN | 0.6625 (ACC 0.6798) | 0.6967 | 0.6732 (ACC 0.6859) | — |
| SAKT | 0.6658 (ACC 0.6808) | — | 0.6407 (ACC 0.6355) | **0.7671** (ACC 0.7271) |
| AKT | 0.7557 (ACC 0.7204) | — | 0.7145 (ACC 0.6938) | — |
| GKT | — | — | 0.6697 (ACC 0.7118) | — |

采样口径：SAINT 用全量；GIKT 只抽 5000 学生；LPKT 用 10% 数据（78,431 学生）；DGAKT 用 117,345 学生/30M 条交互。**EdNet 原论文（[arXiv:1912.03072](https://arxiv.org/abs/1912.03072)）本身不含 KT 模型 AUC 表**。SAINT 来源：[arXiv:2002.07033](https://arxiv.org/abs/2002.07033)。【已核实】

### 2.5 补充：RAG-KT 论文自报的基线（口径：子序列长 25、学生级不重叠划分、1000 条测试序列、5 次平均）

ASSIST09 AUC：DKT 81.82 / DKT+ 82.47 / AT-DKT 82.96 / DKVMN 81.28 / Deep-IRT 80.98 / AKT 84.11 / GKT 81.53 / HISE-KT 84.31 / EFKT 61.10（百分制，详见第 3 节）。【已核实】

---

## 3. RAG-KT（ACL 2026 Findings, [arXiv:2604.10960](https://arxiv.org/abs/2604.10960)）完整实验表

已核实 arXiv 存在（v1 2026-04-13, v2 2026-04-22）且 ACL Anthology 收录（[2026.findings-acl.257](https://aclanthology.org/2026.findings-acl.257/)）。作者：Duan, Hongyu Yuan, Rui Liu（内蒙古大学）。以下全部表格数字直接核读 arXiv HTML 原文（v2）。【已核实】

实验设置：三个基准 ASSIST09 / ASSIST12 / DBE-KT22（澳国立大学在线课程数据集，[Abdelrahman et al. 2022]），另用 Eedi（NeurIPS 2020 挑战赛）做全冷启动。子序列长 25，学生级不重叠划分，1000 条测试序列，5 次随机平均。知识图谱规模：317 概念、593 先修关系、932 关联边、34,171 学生、951 个 Question Group、330 万+交互。

### 3.1 表1 主结果（ACC / AUC / F1，百分制；粗体为原文最优，下划线次优）

| 类别 | 方法 | A09 ACC | A09 AUC | A09 F1 | A12 ACC | A12 AUC | A12 F1 | DBE ACC | DBE AUC | DBE F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| DL | DKT | 72.18 | 81.82 | 78.12 | 70.86 | 69.12 | 74.98 | 71.08 | 72.26 | 74.52 |
| DL | DKT+ | 72.64 | 82.47 | 78.65 | 71.34 | 70.16 | 75.42 | 71.76 | 73.42 | 75.06 |
| DL | AT-DKT | 72.62 | 82.96 | 79.16 | 71.58 | 71.28 | 75.84 | 72.32 | 74.18 | 75.48 |
| DL | DKVMN | 71.32 | 81.28 | 77.92 | 70.51 | 68.73 | 74.76 | 71.47 | 72.03 | 74.65 |
| DL | Deep-IRT | 71.96 | 80.98 | 77.84 | 71.02 | 69.19 | 75.03 | 72.04 | 72.81 | 75.09 |
| DL | AKT | 73.76 | 84.11 | 80.28 | 73.09 | 72.79 | 77.82 | 73.61 | 75.29 | 77.48 |
| DL | GKT | 72.23 | 81.53 | 78.23 | 70.96 | 69.07 | 75.11 | 72.06 | 73.11 | 75.03 |
| LLM-prompt | HISE-KT | 79.12 | 84.31 | 82.03 | 73.77 | 72.33 | 79.51 | 70.82 | 72.40 | 79.94 |
| LLM-prompt | EFKT | 64.85 | 61.10 | 73.59 | 63.03 | 66.79 | 66.83 | 63.50 | 68.28 | 66.44 |
| LLM-finetune | EPLF | 70.13 | 81.27 | 71.60 | 70.30 | 69.64 | 75.82 | 74.63 | 72.72 | 72.38 |
| LLM-finetune | LLM-KT | 78.68 | 83.55 | 81.03 | 72.23 | 72.57 | 78.73 | 77.48 | 75.27 | 78.25 |
| LLM-finetune | 2T-KT | 74.55 | 81.32 | 80.65 | 72.80 | 71.60 | 75.45 | 75.43 | 74.42 | 78.86 |
| LLM-finetune | CIKT | 75.17 | 82.27 | 80.27 | 72.63 | 71.89 | 77.67 | 76.38 | 74.73 | 77.14 |
| Ours | RAG-KT GPT-4o | 78.34 | 82.57 | 83.20 | 72.50 | 72.87 | 79.37 | 76.50 | 74.53 | 85.74 |
| Ours | RAG-KT Qwen-Plus | 79.20 | 83.97 | 85.06 | 72.60 | 73.35 | 80.68 | 78.40 | 73.69 | 86.60 |
| Ours | **RAG-KT DeepSeek-R1** | **80.00** | **85.74** | **85.75** | **74.80** | **73.89** | **82.60** | **78.89** | **76.32** | **86.80** |

注：RAG-KT 在 ASSIST12 上 AUC 73.89，仅比 AKT（72.79）高 1.1 个点；在 DBE-KT22 上 AUC 76.32 对 AKT 75.29 也只高 1 个点——其相对传统 DL 的优势没有宣传语那么大，主要增益在 F1（DBE 86.80 vs AKT 77.48，疑似阈值/类别不平衡所致，值得警惕）与冷启动场景。

### 3.2 表2 消融（DeepSeek-R1 骨干，百分制）

| 变体 | A09 ACC/AUC/F1 | A12 ACC/AUC/F1 | DBE ACC/AUC/F1 |
|---|---|---|---|
| Full | 80.00 / 85.74 / 85.75 | 74.80 / 73.89 / 82.60 | 78.89 / 76.32 / 86.80 |
| w/o Similar Students | 77.28 / 81.07 / 83.60 | 69.53 / 71.47 / 78.13 | 77.40 / 73.98 / 84.17 |
| w/o Question Groups | 76.73 / 84.25 / 83.32 | 70.01 / 72.33 / 78.67 | 77.94 / 75.50 / 85.16 |
| w/o KC Interest Subgraph | 77.46 / 85.49 / 84.23 | 70.01 / 73.06 / 78.54 | 77.40 / 75.23 / 85.08 |
| w/o Structural Path Subgraph | 78.50 / 84.16 / 84.89 | 72.40 / 71.54 / 80.67 | 76.71 / 75.98 / 85.32 |
| Random Similar Students | 78.54 / 83.62 / 84.03 | 73.36 / 73.14 / 81.56 | 78.38 / 75.72 / 86.43 |
| **w/o Retrieval** | **68.50 / 62.37 / 74.35** | **64.20 / 67.41 / 71.35** | **64.40 / 68.58 / 72.65** |

要点：去掉检索模块后性能崩塌到接近纯 prompting 基线水平——说明增益几乎全部来自检索到的结构化上下文，而非 LLM 本身。

### 3.3 表3 全冷启动（目标平台 Eedi，知识库/图中完全不含 Eedi 信息，唯一跨平台信号是 Question Group 检索）

| 类别 | 方法 | Eedi ACC / AUC / F1 |
|---|---|---|
| DL | DKT | 49.54 / 50.23 / 49.61 |
| DL | DKT+ | 49.98 / 50.62 / 49.84 |
| DL | DKVMN | 51.58 / 51.20 / 52.64 |
| DL | AKT | 53.55 / 49.69 / 47.68 |
| LLM cold-start | LOKT | 65.88 / 65.52 / 57.90 |
| LLM cold-start | CLST | 61.45 / 63.60 / 42.23 |
| Ours | RAG-KT DeepSeek-R1 | **68.00 / 74.40 / 70.43** |

**这张表是本想法最核心的弹药**：传统 DL KT 模型在未见平台上 AUC≈50（接近随机），直接证明"平台绑定的 ID 嵌入无法跨平台"；而带 KC 语义对齐（Question Group）的零训练方法 AUC 74.4，比最强 LLM 冷启动基线高 8.9 个点。

### 3.4 表6 KC Match 对齐准确率（专家人工标注 ground truth）

| 源 → 目标 | 对齐准确率(%) |
|---|---|
| ASSIST09 → ASSIST12 | 96.33 |
| ASSIST09 → DBE-KT22 | 92.67 |
| ASSIST12 → DBE-KT22 | 96.50 |

方法：两阶段混合——直接词汇/嵌入匹配处理简单情形，剩余歧义概念由 LLM（GPT-4，四条严格判据的 prompt）映射。**这是目前唯一对"LLM 跨数据集 KC 对齐"给出量化准确率评估的工作**，可直接引用作为可行性证据。

### 3.5 表4 报告质量 / 表5 成本（摘要）

- 报告质量（0.7 人评+0.3 LLM 评，四维各 1–5 分）：RAG-KT 总分 19.24 > HISE-KT 18.08 > CIKT 17.23 > EFKT 10.05。
- 成本：RAG-KT 训练 0 小时、推理约 8 s/样本、AUC 85.74；CIKT 训练 24 h、3 s/样本、82.27；HISE-KT 训练 0、19 s/样本、84.31。
- 超参：检索权重 λ1:λ2:λ3=4:3:3（行为:结构:能力），top-k=2，A09 最佳 seq_len=25，A12/DBE 在 100 仍上升。
- 代码：未找到官方仓库（截至 2026-10-03）。

---

## 4. 各数据集 KC 数量与命名体系

| 数据集 | KC 数 | 核实来源 | 标注 |
|---|---|---|---|
| ASSISTments2009 | **123** | [pyKT 论文表1](https://arxiv.org/html/2206.11460v5)、EdNet 论文对比表、simpleKT §4.1.1 | 【已核实】✓ 与预期一致 |
| ASSISTments2012 | **265** | EdNet 论文对比表、DIMKT 表2、GIKT 表1、LPKT 表3 四处一致（前提：with-predictions 文件+删无 KC 记录） | 【已核实】 |
| ASSISTments2015 | **100** | [pyKT 文档](https://pykt-toolkit.readthedocs.io/en/latest/datasets.html)、simpleKT | 【已核实】 |
| ASSISTments2017 | **102** | AKT 论文表1、LPKT（ASSISTchall）、DGAKT | 【已核实】（pyKT 文档页面写"102 questions"是笔误，实为 102 KC） |
| EdNet-KT1 | **188**（官方口径） | EdNet 论文：KT1/KT2=188 tags，KT3/KT4=293（含 lecture tags）；DGAKT 也用 188。**189 出自 GIKT 的 5000 学生子样本**，非主流口径 | 【已核实】 |

命名体系（决定 LLM 对齐可行性）：

- **ASSISTments**：`skill_id` + `skill_name` 双字段，skill_name 是**有语义的自然语言名称**（已核实例："Ordering Integers"（AKT 论文表7）；ASSISTments 官方数据页：[skill builder data](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)）。【已核实】→ 可直接用名称做 LLM 语义对齐。
- **EdNet**：tags 是**纯数字 ID**（如 `179;53;183;184`），专家标注、有层级，但**官方未公开语义名称**（EdNet GitHub README 已核实）。LPKT 只用最细粒度 tag，其口径下仅 141 个概念。【已核实】→ **EdNet 无法直接做名称级 LLM 对齐，需先经题目内容反推 tag 语义**，这对实验设计是个硬约束。
- DBE-KT22：有 KC 名称与体系（DBE-KT 数据集自带知识组件标注），RAG-KT 用其做对齐目标。【转述】

---

## 5. 零样本/免训练跨域迁移在教育任务的先例

### A. 认知诊断（CD）——已有成形的 DZCD（domain-level zero-shot cognitive diagnosis）文献线

1. **TechCD**（SIGIR 2023）[ACM](https://dl.acm.org/doi/10.1145/3539618.3591774) / [代码](https://github.com/bigdata-ustc/TechCD)：人工教学知识概念图（KCG）为中介连接不同域，GCN 学可迁移认知状态。目标域零作答数据，但需人工 KCG。【已核实】
2. **Zero-1-to-3**（AAAI 2024）[arXiv:2312.13434](https://arxiv.org/abs/2312.13434)：域共享/域特有状态解耦 + 一小批"early-bird"学生生成虚拟日志。**需少量目标域数据**，非严格零训练。【已核实】
3. **ICDM**（WWW 2024）[arXiv:2404.11290](https://arxiv.org/abs/2404.11290)：学生中心图邻居聚合替代学生 ID 嵌入，新学生免重训。同域 inductive，非跨域。【已核实】
4. **LRCD**（SIGIR 2025）[arXiv:2501.13943](https://arxiv.org/abs/2501.13943)：行为模式写成文本画像→统一语言空间→轻量"语言-认知映射器"接入任意 CDM。**目标域零训练零数据**，但映射器需源域训练。跨学科（数↔英↔物）与跨平台（ASSISTments↔Junyi↔Khan）部分场景媲美目标域全量训练。**与本想法最接近的先例**。【已核实 abstract】
5. **LCST**（Frontiers of Digital Education 2025）[期刊页](https://journal.hep.com.cn/fde/EN/10.1007/s44366-025-0054-y)：源域 CDM 预诊断→自然语言→**完全冻结的 LLM**（Llama 3.2/Gemma）推理概念关系直接给目标域诊断，目标域零作答数据。但冻结的是通用 LLM 而非教育预训练模型。【已核实】
6. **LMCD**（arXiv 2025-05）[arXiv:2505.21239](https://arxiv.org/html/2505.21239v1)：LLM 知识扩散+语义-认知融合，exercise-cold 与 domain-cold 均超 SOTA；仍需源域训练。【已核实】
7. **PromptCD**（IEEE TCSS 2025）[arXiv:2412.05004](https://arxiv.org/pdf/2412.05004v1)：软提示迁移做双方面跨域 CD 适配。【转述】
8. **PLCD**（arXiv 2026-09）[arXiv:2609.12403](https://arxiv.org/abs/2609.12403)：完全抛弃 ID 嵌入，LLM 构建概念图式+认知过程图作为语言先验；仍需作答记录校准。【已核实】

### B. 自适应测试（CAT）——基本空白

- **DCSR**（arXiv 2024-11）[arXiv:2411.12182](https://arxiv.org/abs/2411.12182)：用考生在其他课程的作答记录，扩散模型构建域间认知状态迁移桥，生成目标域初始能力接入现有选题算法。迁移的是"初始能力先验"，选题策略不迁移。【已核实】
- 未找到"冻结整套 CAT 系统搬到新题库"的工作。【已核实（否定结论）】

### C. LLM zero-shot 教育评估的证据（偏负面 → 恰好支持"语义对齐是关键瓶颈"）

- **Neshaei et al.（EDM 2024）**[EDM 页](https://educationaldatamining.org/edm2024/proceedings/2024.EDM-posters.84/index.html) / [arXiv:2403.14661](https://arxiv.org/html/2403.14661v1)：首次系统检验 LLM 直接做 KT——**zero-shot 无法捕捉知识状态随时间变化**，微调后才超基线。【转述】
- **MATHCOG**（arXiv 2025-04）[arXiv:2504.00843](https://arxiv.org/abs/2504.00843)：3036 条教师标注诊断上评 18 个 LLM，**全部 F1<0.5**，证据模糊时骤降，系统性幻觉。【已核实】
- **LLMKT**（LAK 2025）[PDF](https://learninganalytics.upenn.edu/ryanbaker/Dialogue_KT_LAK_25-2.pdf)：辅导对话上 LLM KT 显著超基线——但对话有丰富语义，不同于稀疏答题矩阵。【转述】

### D. "冻结预训练模型+语义对齐零训练迁移"的直接先例

- **Schmucker & Mitchell**（arXiv 2022）[arXiv:2202.03980](https://arxiv.org/abs/2202.03980)：课程无关性能模型，新课程**零学生交互**，仅靠专家提供的题目难度等语义特征即达数千学生训练的 BKT/PFA 精度。KT 侧最早的"特征对齐零训练迁移"先例之一。【已核实】
- 领域外同构：推荐系统 TIGER（CIKM 2022）/ TMCDR（SIGIR 2021）的"可迁移嵌入+语义中介"路线，教育侧 TechCD/LRCD 即其移植。【转述】

### 本方向小结

CD 侧已有任务定义（DZCD/ZCCD）与三代方法（图中介→少量目标域数据→语义对齐零样本），但**没有一例冻结的是"预训练教育模型"本身**：LRCD 冻结文本编码器但映射器要源域训练；LCST 冻结的是通用 LLM；Zero-1-to-3 还要 early-bird 学生。CAT 侧几乎空白。KT 侧见第 1 节：RAG-KT 冻结 LLM 但绕开深度 KT 模型。**"冻结深度 KT checkpoint + LLM 做 KC 语义对齐 + 目标域零训练"的严格形态在 KT/CD/CAT 三个相邻文献中均未见先例**。

---

## 6. 给研究想法的定位建议（判断性意见）

1. **novelty 主张**：组合空白真实存在——AdaptKT 证明"换头"之痛，RAG-KT 证明 KC 对齐可行且给量化准确率（92.7%–96.5%），RAG-KT 表3 证明 DL KT 跨平台崩塌到 AUC≈50。三者拼起来就是 motivation 链。
2. **直接对手**：RAG-KT（同思路、零训练、已发 ACL 2026 Findings）。差异化点应放在"迁移已训练的深度 KT 模型的知识状态建模能力"（RAG-KT 没有知识状态递推，只有检索上下文）和"推理成本"（8 s/样本 vs 冻结模型毫秒级）。
3. **实验注意**：EdNet 无公开 skill 名称，若选 EdNet 做目标域需先解决 tag 语义化；ASSISTments 系（09/12/17 同平台但 KC 体系不同）是最自然的源-目标对；DBE-KT22 可作为跨平台对。
4. **评测口径**：引用基线数字时锁定单一来源（建议 pyKT 口径复跑或 RAG-KT 表1 口径），跨来源数字不可混比（见 §2 警告）。
5. **需要自建的基准**：跨数据集 KC 对齐准确率没有公认基准，RAG-KT 的专家标注协议（源→目标对齐 ground truth）是可复用的评测范式，可作为论文贡献之一。

### 未决缺口（诚实声明）

- GKT 在 AS2012/AS2017/EdNet 上无经典权威数字，仅有 2025 年新论文（DGAKT）复跑值。
- RAG-KT、AdaptKT、AEGOT-CDKT、DTransKT 以外的大部分迁移工作代码不可得，复现需谨慎。
- LRCD 的 venue（SIGIR 2025）与 LLMKT（LAK 2025）标注来自引用清单转述，未在会议官网逐一核对。
- ASSISTments2012 的 265 KC 依赖特定预处理口径（with-predictions 文件）；换口径数字会变。
