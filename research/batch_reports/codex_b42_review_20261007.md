**总体判定：FAIL。** 数字引用、端点分支和 v2 复现锚成立，但仍存在两条标签泄漏路径，不能认可“无泄漏修复完成”及“干净排除”的升级。**未发现 P0。**

以下判断基于实际打开全部指定文件、JSON 算术复核，以及从源码提取函数运行的只读合成反例；未从原始学生轨迹重跑三域实验。

**P1：必须修复并重跑**

1. **多 KC 展开的调参作答标签仍进入 `prior_tuning`。**

   [v3.py:67–84](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:67) 只排除调参索引；同次作答的 A1 副行不进入 `tset`，因此仍被纳入先验。同一作答展开并携带相同标签的依据是 [data_source.py:988–997](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/utils/data_process/data_source.py:988) 和 [skill_model_data.py:285–292](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/utils/model_data/skill_model_data.py:285)。

   合成反例：调参主行 `i=5,q=100,kc=1` 被排除，副行 `i=6,q=100,kc=2` 留下。修改这次作答的标签，`prior_tuning[2]` 随之改变，影响另一学生的 `kc=2` 调参预测。

   **准确边界：主行自身索引的标签已排除；副 KC 标签不会直接进入当前主行的 KC=1 先验。** 问题是调参作答标签仍进入训练表，破坏整个调参集的标签隔离。需按作答交互整体排除所有展开行。本批实际受影响数量及是否翻转 α，尚未核实。

2. **真实 holdout 标签仍通过“建表学生成员选择”影响 `prior_final`。**

   [baseline_eval.py:64–90](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/baseline_eval.py:64) 在第 80 行根据学生真实 holdout 是否包含正负两类，决定其 evidence 是否进入建表集合。[v3.py:118](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:118) 仍调用这一构造。

   源码函数反例：两位学生的 evidence 完全不变，仅将第二人的一个 holdout 标签从 0 改成 1，使其 holdout 从单类变双类，先验从 `prior[0]=0` 变为 `0.5`。

   因而 [报告:7、13](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch42.md:7) 的“eval 不受影响／holdout 未动”不成立。需从全部学生的非 holdout evidence 独立建表；双类过滤只用于评分。旧构造可保留为复现锚。

**P2：裁决与证据解释需要修正**

3. **G1 的 delta 子门在 α*=1 时自动满足；“干净排除”超出当前证据。**

   [v3.py:149–172](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:149) 决定 α=1 时两臂逐行预测完全相同，因此 `delta≡0`。`delta≤+0.005` 对这些域没有额外检验力。

   **报告如实披露了这一点**：[报告:27](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch42.md:27) 明写“delta 平凡为 0”；正文也说明它是构造恒等式。

   五点均值严格单调、α=0 端点明显更差，支持“该代理在当前网格、学生内 AUC 准则下未被选中”。它们不能单独证明该代理没有增量价值：未检查更细 α 网格，也没有选择稳定性或增量上界证据；最终端点还受上述 P1 影响。[报告:32–38](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch42.md:32) 的“干净排除”应撤回。

4. **`leaky−leakfree` 不是隔离后的“泄漏量级”。**

   `prior_tuning` 使用全部 samples；`prior_leaky` 使用经 holdout 双类筛选的学生集合。证据：[v3.py:56–84](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:56)、`baseline_eval.py:80–81`。

   因此 +0.0143/−0.0056/−0.0005 同时可能包含建表学生集合变化。[报告:36](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch42.md:36) 将其归因为泄漏影响，未隔离混杂。当前五点排序未变属实，但不能据此认定泄漏本身影响极小。

5. **选择层样本是否充足，现有结果无法判断。**

   [v3.py:99–112](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:99) 只保存调参均值，没有调参行数、有效学生数、正负类数或选择稳定性。[transfer_eval.py:36–49](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/transfer_eval.py:36) 会排除调参标签只有一类的学生。

   `eval n=261/245/213` 是评估层覆盖，不能代替调参层覆盖。没有证据证明样本不足，也没有足够记录证明选择稳定。

**P3：描述与自检精度**

6. **“最近≤20题”“纯能力”需要限定。**

   [v3.py:40–42](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:40) 实际取最近 100 个展开行，排除当前同题行，再取最多 20 个响应行；没有按作答交互去重。多 KC 作答可被重复计权，因此不严格等于“20题”。

   同时 α=0 在历史不足时仍 fallback 到 `p1`（第 149–154 行），应称“能力端点，历史不足时回退先验”。

7. **B1 自检是容差检查，并非严格相等或失败即阻断。**

   [v3.py:176–179](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:176) 允许误差 ≤0.002，只记录布尔值。不过本次 JSON 的四位小数确实与 v2 完全相等，因此没有实际锚数值错误。

**其余要求核对通过**

- **A，索引与历史时序：** `hold_set | tset_set` 正确排除被选中索引；调参预测后才 append，评估也是预测后 append。历史中此前已揭示的标签符合声明的前缀评估协议。缺口是上述交互展开与成员选择。
- **B，参数与等价性：** `build_rows(samples,None)` 第二参数是 **KC alignment 名称**；`None` 表示不做 alignment 映射筛选，见 `baseline_eval.py:36–55`。`prior_final == prior_leaky` 注释成立，两者与 [v2.py:44–45](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v2.py:44) 同构，也共同继承成员选择泄漏。
- **C，调参规则：** v2 第 61–70 行与 v3 第 67–70、97–105 行一致。严格说是“原始展开索引满足 `i%5==0`”，而非重新编号后的每第五个合格 evidence 行；复现比较公平。
- **F，端点：** 当前 `eval_alpha1=p1`、`eval_alpha0=ab`，分支没有反转；历史不足时共同回退 `p1`。
- **G，学生数：** v3 与 [v2 JSON:6、13、20](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/results/a17_ability_decompose_v2.json:6) 均为 **261/245/213，没有差异**。评估只纳入 A1 排除后仍有正负两类的学生，v3 再取两臂共同学生。

**E：全部指定数字对数通过。** 域顺序为 A17／EdNet／Junyi，以下行号均指 [a17_ability_decompose_v3.json](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/results/a17_ability_decompose_v3.json:1)。

| 数字项 | 核对值 | JSON 行号 |
|---|---|---|
| leakfree α=1 调参均值 | 0.5512 / 0.5980 / 0.6093 | 9 / 53 / 97 |
| leakfree α=0 调参均值 | 0.4797 / 0.4838 / 0.4542 | 5 / 49 / 93 |
| α=0 endpoint delta | −0.1536 / −0.1875 / −0.1698 | 36 / 80 / 124 |
| endpoint 配对 n | 261 / 245 / 213 | 35 / 79 / 123 |
| α=1 leaky−leakfree | +0.0143 / −0.0056 / −0.0005 | 17−9 / 61−53 / 105−97 |
| B1 复现锚 | 0.5481 / 0.5252 / 0.5624 | 21 / 65 / 109 |

A17、EdNet 用已舍入的 `B1ab−B1` 相减会与 stored delta 差 0.0001；代码先平均未舍入配对差再舍入，这是正常精度差异。

**门的最终判断：G1 按数字机械满足，G2 未触发，G3 及附加 B1 锚成立；“无泄漏”前提未成立，因此裁决升级不通过。** 本地 Git 中预注册门已出现在结果提交之前的 `f9a4d78`，无需怀疑门本身是结果后补写。修复两条 P1 并重跑前，可保留数字和旧路径复现事实，应撤回“干净排除”。
