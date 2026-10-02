# Holdout 结果汇总（2026-10-03，v2 修复版矩阵）

执行：冻结补充协议（freeze_addendum）后的单次确认运行。全部 8 个条件 × 113 生（用户去重后池），主分析入组 k_i≥4（40 生）。LLM=glm-5.3-flash（thinking disabled，temp=0）/ kimi-for-coding（temp=1 强制）。全部调用归档于 research/llm_explain/archive/（含 model 版本串、request_id、token 计数、全部重试）。

## 1. 主表：Fisher-z 加权 ρ（w=k−3，ρ 截 ±0.99）

| 条件 | LLM ρ | 单侧95%下界 | 基线: acc抄表 | 基线: readiness直排 | 基线: 模型在线 | 接地违规率 |
|---|---|---|---|---|---|---|
| DKT×GLM×full | 0.120 | −0.031 | 0.159 | 0.019 | 0.423 | 0.28% |
| DKVMN×GLM×full | 0.283 | 0.138 | 0.159 | 0.227 | 0.573 | 0.25% |
| SAKT×GLM×full | 0.220 | 0.072 | 0.159 | 0.195 | 0.215 | 0.87% |
| AKT×GLM×full | 0.267 | 0.121 | 0.159 | 0.236 | 0.536 | 4.62% |
| DKT×GLM×behavior | 0.188 | 0.038 | — | — | — | 2.59% |
| DKVMN×GLM×behavior | 0.257 | 0.110 | — | — | — | 0.98% |
| SAKT×GLM×behavior | 0.193 | 0.043 | — | — | — | 1.85% |
| AKT×GLM×behavior | 0.214 | 0.065 | — | — | — | 2.84% |

解析成功率 113/113 全条件（Kimi 条件完成后补）。敏感性（不加权/中位/k≥5）方向一致，见 finalize 输出。

## 2. 主终点（confirmatory）

DKT×GLM×full：ρ̂=0.120，单侧 95% 下界 −0.031，**未超过 acc 抄表基线 0.159+0.05 → 未通过**。

## 3. 预注册次要结论

- **readiness 消融（full vs behavior，逐生配对符号检验，n=40）**：DKT p=0.430 / DKVMN p=0.430 / SAKT p=0.875 / AKT p=0.875——任何模型均**未检测到** readiness 列的显著增量。组间加权差方向与各模型 readiness 信号质量一致（DKVMN +0.026 / SAKT +0.027 / AKT +0.053 / DKT −0.068；readiness 直排基线 0.236/0.195/0.227/0.019），【探索性】提示"内部信号价值取决于信号质量；对信号无预测力的模型（DKT）引导使用反而有害"。
- **接地违规率**：全条件 <5%（0.25%–4.6%），首试口径、单元格级绑定（kc+field+value 精确匹配、%归一化）。
- **SAKT 上 LLM 追平模型在线预测**（0.220 vs 0.215）：LLM 仅凭 evidence 段信息达到模型在线（可看 holdout 前序真实作答）的水平，【探索性】。

## 4. 功效限制（预注册披露）

每生 holdout 目标 KC 中位 4（max 8）→ k≥4 入组仅 40/113；逐生 ρ 噪声大，消融配对 CI 半宽 ~0.15-0.2（z 空间），**只能排除 |Δρ|≳0.2 的增量**，无法排除小幅增量。"内部信号无增量"的等效性结论**不成立也不宣称**；结构性改进方向：更高 KC 密度的 holdout 设计（更长 holdout 段/题级预测对账）。

## 5. NLI 交叉裁判

dev 阶段（5 生，Kimi 判 GLM）：supports 31 / unverifiable 52 / contradicts 15（98 条，此前日志 71 为笔误）。holdout 全量裁判与人工校准列入待办（未完成项，如实披露）。

## 6. 结论措辞（按 freeze addendum 约束）

1. 【confirmatory·未通过】主终点未通过。
2. 【confirmatory】冻结 KT 模型 + 便宜 LLM（flash 档）的"先审计后翻译"诊断流水线**工程可行**：解析 100%、接地违规 <5%、holdout 覆盖 100%、诊断样例引用全部可溯源（样例见 results/holdout_dkvmn_glm_full/report_U1817.json）。
3. 【exploratory】LLM 预测与三个程序基线的相对位置随 KT 模型变化；DKT 的 readiness 信号本身接近无预测力（0.019），其消融差为负。
4. 【功效受限】本数据结构的 holdout KC 密度不支持 <0.2 的增量检测——这是该实验范式在 assistments09 上的结构性边界，不是 LLM 能力结论。
