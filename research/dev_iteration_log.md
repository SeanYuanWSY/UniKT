# dev 迭代日志（30 生，全部披露）

按设计 v2 预注册的 dev/holdout 协议：dev=30 生（seed=7 分层）用于迭代，holdout=120 生单次确认。
本文件披露 dev 上的全部迭代轮次、每次发现与修改，holdout 阶段配置以此为准、不再改动。

## 轮 1（2026-10-03 03:20–03:35）：管线打通 + full 条件基线

- 修复三个工程问题：GLM-5.3-flash 为 reasoning 模型（`thinking: disabled` 参数，否则 4096 token 全耗在思维链、正文为空）；`import model` 需仓库根入 sys.path；holdout 目标 KC 空交集越界。
- **dev full 条件（DKT×GLM，30 生）**：解析 30/30，接地违规率 2.2%，holdout 覆盖 100%。
- Fisher-z 加权 ρ=0.375 [0.013, 0.662]；readiness 直排基线 0.180；模型在线基线 0.490。
- **发现 1（关键约束）**：每生 holdout 目标 KC 中位数仅 4（min 0 / max 8），逐生 ρ 噪声大。加权（k−3）已缓解；holdout n=120 时总体 CI 收窄。
- **发现 2**：LLM 预测分布健康（0.1–1.0 全谱段，无中位塌缩）。

## 轮 2（03:40–04:10）：探针 + 消融 + 一个评分 bug

- NLI 交叉裁判（Kimi 判 GLM，5 生 71 条 claim）：supports 31 / unverifiable 52 / contradicts 15。contradicts 的 claim 待人工抽查归因（裁判严格 vs 真矛盾）。
- 修复：Kimi 仅接受 temperature=1（传 0 全部 400）；NaN 进 spearman 产生垃圾基线（behavior 条件下 readiness 基线 0.427 为 bug 产物，NaN 过滤后重算）。
- **dev permuted（readiness 生内置乱）**：加权 ρ=0.363，违规率 4.4%（vs full 0.375 / 2.2%）。ρ 微降 + 引用违规上升 ⇒ LLM 确实在读取并引用 readiness，但对预测的边际贡献小。
- **dev behavior-only（删 readiness 列）**：加权 ρ=0.497（配对符号检验 vs full：中位 Δz=0，p=0.078，方向不利）。违规率 9.8%（含引用已删列的 claim，符合预期的纪律违规）。
- **发现 3（负结果候选）**：对 DKT，readiness 内部信号列不增加 LLM 的 holdout 预测力；行为统计（正确率/最近5次）主导。是否成立于 AKT/DKVMN 由 holdout 矩阵回答——"内部信号价值是否模型依赖"本身成为 holdout 阶段的研究问题之一。

## 冻结的 holdout 配置（不再改动）

- 样本：kfold fold0 验证用户、evidence KC≥10、80/20 切分、dev 30 生排除后分层抽 120（seed=42）。
- 条件：4 模型 × GLM × full（并行）→ DKT×Kimi×full、DKT×GLM×behavior-only。
- 主终点（单一）：DKT×GLM×full 的 Fisher-z 加权 ρ 的 95% CI 下界 > readiness 直排基线（同 run 内计算）。
- 次要（Holm 校正，≤6）：各模型 ρ、Kimi 一致性、behavior 配对差、违规率、NLI contradicts 率、解析失败率。
- 其余（模型间对比、探针在 holdout 上的复验）为描述性。
