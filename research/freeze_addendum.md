# 冻结补充协议（揭盲前书面修正）

时间戳：2026-10-03 04:2x（holdout 矩阵运行中，**任何 holdout 汇总指标尚未读取**——各 run 目录 summary.json/records.jsonl 尚未生成完毕）。本补充回应协议终审（外部审查）发现的 dev 后偏离，全部落痕。

## 1. 主终点恢复预注册口径（修 P0-1）

- 主终点（confirmatory）：DKT×GLM×full、n=120、k_i≥4 学生、Fisher-z 加权 ρ（w_i=k_i−3）、ρ_i 截断至 ±0.99，**单侧 95% CI 下界 > E1 行为外推抄表基线同指标 + 0.05**。
- 抄表基线（程序化，零 LLM 成本，离线重算）：evidence 段每 KC 正确率直排（acc-copy 基线）。readiness 直排与模型预测均值直排一并报告。
- readiness 直排版本降级为"偏离预注册的次要版本"，两版都报。

## 2. 入组标准修正（修 P0-2）

- 预注册 k_i≥10 在该数据不可行（dev 实测 holdout KC 中位 4、max 8）。主分析入组改为 **k_i≥4**（k=3 及以下排除并报告排除数）；`evidence KC≥10` 声明为抽样筛选项，非主分析入组。
- ρ_i=±1 截断至 ±0.99（Fisher-z 有限性）；dev 表用同一规则重算。
- 敏感性分析（预注册）：不加权均值、逐生中位 ρ、k_i≥5 子集、Kendall τ-b 佐证。

## 3. 范围与混杂披露（修 P1-3/4/5/6/7）

- 四模型同一划分【已核实：四 run_config 均 data.fold=0、seed=42、同一份 data/assistments09 产物，fold 划分由预处理 seed 决定】。
- Kimi 温度被迫 1.0（API 限制）登记为对"温度 0"条款的强制偏离；Kimi 全部结论限描述性；稳定性改为重复 3 次离散度。
- **补 3 条 behavior 臂**（DKVMN/SAKT/AKT × behavior-only × n=120）——"内部信号价值是否模型依赖"由配对增量检验回答，不再是描述性代理。
- permuted 未进 holdout：范围收缩披露，相关结论限 dev 探索性。
- 失败处理：首试解析失败→剔除+计数+最坏情形敏感性；按条件统计重试/截断/429；归档含 model 版本串（llm_client 自始记录 `model_returned`）。
- NLI 计数以归档文件为准（dev 日志 71 为笔误，正确总数以 nli_judge_kimi.json 计）；unverifiable 抽样纳入人工校准。
- 冻结决策与 dev 结果的相关性声明：主终点钉死 DKT×GLM×full 发生在 dev 结果之后（dev 只跑过 DKT），"模型依赖性"升级为研究问题由 dev 发现 3 驱动——报告相应措辞降级为模板（见 dev_iteration_log 附录）。

## 4. Claims 措辞约束

confirmatory 仅限：主终点通过/未通过 + 消融配对差（Holm）。负结果只能写"未检测到（CI 半宽 ~0.15，不足以排除小幅增量）"；禁止普适化表述（"内部信号无价值"/"LLM 增量得到证明"/"模型依赖性已回答"除非补臂检验完成）。
