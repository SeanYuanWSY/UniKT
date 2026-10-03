# 题级对账设计 v1（零 API 成本，复用已归档报告）

背景：KC 级对账功效天花板（每生 holdout KC 中位 4 → 只能排除 |Δρ|>0.2）。题级把统计单位从 40 个学生级 ρ 变成数千条作答级 (prediction, label) 对，检测限预期压到 0.05-0.08。

## 数据（全部已有，无需新调用）

- 9 条件的 `report_U{uid}.json`：每生的 `holdout_predictions` [{kc, pred_acc}]（LLM 在只看 evidence 证据包时对将出现的 KC 的正确率预测）。
- `pack_U{uid}.md` / 离线重建：evidence 段每 KC 的 acc（抄表基线源）。
- 冻结模型 + window 样本：离线重跑 `holdout_actuals` 得到每条 holdout 作答 (student, kc, question, label, model_online_pred)。

## 配对单位

每条 holdout 作答 (s, kc, q, y)。四个预测来源在**同一批作答**上对账：

| 来源 | 值 | 信息面 |
|---|---|---|
| LLM | 该生该 KC 的 pred_acc | evidence 段模型信号+行为统计（KC 级常数，同 KC 并列） |
| 抄表 acc | evidence 该 KC 正确率 | 仅行为统计 |
| readiness 直排 | 该 KC 的 readiness(c) | 仅模型内部信号 |
| 模型在线 | 该步前向预测 | evidence+holdout 前序真实作答（信息面最大，参照上界） |

## 指标

- 主：AUC（全局，学生级 cluster bootstrap 95% CI；并列处理：LLM/抄表是 KC 级常数 → 同 KC 内作答天然并列，AUC 按标准并列 0.5 记分）。
- 次：学生内 AUC（每生内算再聚合，与 KC 级分析衔接）、Brier。
- 关键检验（confirmatory，本题级重分析的唯一预注册比较）：**full vs behavior-only 的 LLM 预测在题级 AUC 上的配对差**（同生同作答，cluster bootstrap 符号化 CI）。四模型各做，Holm 校正。
- 描述性：LLM vs 抄表 vs readiness 直排 vs 模型在线的绝对位置；GLM vs Kimi（DKT）。

## 并发症与预案

1. LLM 的 KC 级常数在同 KC 并列 → AUC 保守（低估 LLM 上限）；若需要题级变化值，升级版让 LLM 输出每条作答的预测（新 API 调用，~113×4 生次）。
2. holdout 段内同一 KC 多次出现的作答共享 evidence 统计 → 无泄漏（预测只来自 evidence 段）。
3. 学生权重：长序列生贡献更多作答对 → 主指标同时报"每生等权的学生内 AUC 平均"。
4. 模型在线基线在 holdout 前序有真实作答（信息面不对等）→ 只作参照上界，不进主检验。

## 判定标准（预注册）

- 若任一模型上 full > behavior 的题级 AUC 配对差 95% CI 下界 > 0（Holm 后仍显著）→ **readiness 内部信号有真实题级增量**，motivation 成立，研究升级为完整论文（扩数据集/模型/LLM）。
- 若全部不显著但 LLM 题级 AUC > 抄表基线显著 → motivation 改为"LLM 综合研判优于照抄统计"（弱一档但可写）。
- 若两者都不成立 → 本接入位点关闭，文献驱动的备选位点（Q-matrix 语义重建/跨数据集 KC 对齐/题级 LLM 直接预测）接棒。
