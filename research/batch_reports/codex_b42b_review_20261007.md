总体判定：**PASS-with-fixes**。原 **P1-1、P1-2 已关闭**，未发现新的 P0/P1；但报告仍有 **3 项 P2、2 项 P3**，主要涉及统计措辞和效应归因。

已实际读取指定代码、JSON、两份报告及必要调用路径，并执行内存反例验证；未修改文件或重跑完整模型。当前工作树没有原始数据，因此 **8537/5124/3653 的计数未独立重算**，其结构逻辑已核验。

**问题清单**

| 等级 | 问题与证据 | 必要修正 |
|---|---|---|
| P0 | 无 | — |
| P1 | 无残留；原两项关闭，证据见下文 | — |
| **P2-1** | [batch42.md:38](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch42.md:38) 写“显著为负”，但 [v3.py:249](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:249) 和 JSON 只提供配对均值差、n，没有 CI/p 值 | 改为“三域配对平均 delta 均为负”；现有证据不能支持统计显著 |
| **P2-2** | [batch42.md:35](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch42.md:35) 称“仅调参行回代”。实际 [v3.py:149](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:149) 恢复整个非 holdout 的调参所在 run，包含孪生行，也可能包含独立重答 | 改为“同成员与 holdout 规则下，恢复调参所在同题 run 全部证据的影响”。这些差值不能精确归因为调参标签自身的泄漏量 |
| **P2-3** | [batch43.md:33](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch43.md:33) 将多行同题 run 解释为一次交互/孪生展开。`run_ids` 只识别连续相同 question；尤其 [junyi2015.py:126](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/utils/data_process/junyi2015.py:126) 的处理每题只有一个 KC，同题多行可以是独立重答 | 写“同题连续 run 整体 holdout”；多行 run 计数不能当成多 KC 孪生行计数，也不能单独排除“全是单 KC” |
| **P3-1** | [batch43.md:39](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/batch_reports/20261007_batch43.md:39) 写“18 格全同” | 实际 **12 格**；B1/delta/CI 共 **36 对字段**全部相同 |
| **P3-2** | [v3.py:232](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:232) 在能力历史不足时，即使 α=0 也回退先验；报告称其“纯能力端点” | 写“α=0 端点，历史不足时先验回退”。当前没有保存回退次数 |

**A/B/C/E/F 核验**

- **A：P1-1 关闭。** [v3.py:113](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:113)–128 的唯一建表来源排除 `hold_runs ∪ tuning_runs`，选择入口第188行使用该 `prior_tuning`。内存验证中，调参索引 `[5,10]` 所在 run 行为 `[5,6,10]`；翻转这三行全部标签，`prior_tuning` 完全不变。没有发现自身或孪生标签绕回路径。

- **B：P1-2 关闭。** [v3.py:119](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v3.py:119)–136 直接从全部学生的非 holdout-run 证据构建 `prior_clean`；第209行 `eval_final` 明确使用它。旧 `build_rows` 双类过滤仅影响 leaky/v2repro 及诊断。将学生 holdout 从双类改为全0，三个新先验均不变。**评分时筛选有效 AUC 学生，不再反向决定建表成员。**

- **C：JSON 零漂移不能推出结构，但 loader 提供了独立证明。** 漂移在 [t1.py:125](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/t1_fair_rescore.py:125)–127 被舍入；即使未舍入漂移为0，也可能只是删除行没有改变 KC 均值。我已构造 `partial_runs=1`、两个先验仍精确相等的反例。另一方面，[restore.py:184](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/llm_explain/restore.py:184)–198 将边界推进至同题 run 末尾，再取整个后缀作为 holdout，严格保证 run 不被切开。**同一完整样本集上的 `partial_runs=0` 足以闭合 run 级排除证明；但不证明 run 就是一次作答或多 KC 题。**

- **E：限定结论成立，精确泄漏归因不成立。** 两臂确实保持相同学生成员和 holdout 排除规则，成员选择混杂已消除；但差集是整个调参 run。可保留：**“该标量代理在当前网格与学生内平均 AUC 准则下未被选中，不外推。”** “被拒绝”只能指选择程序没有选它；α*=1 时最终 delta=0 是构造恒等，不是额外拒绝证据。

- **F：未发现新增代码回归。** 空/单行/分离同题块的 `run_ids` 正确；连续同题重答会保守合并、可能多排除证据，但不会漏排目标孪生标签。loader 每用户只保留一条轨迹，排除了 `user_id` 字典覆盖路径。t1 的 `hold_runs` 只改变 `prior_fixed2`，没有改变 `jmap`、keep-set 或 deep 支持集。新增建表处理线性；网格同分选择和 bootstrap seed 固定。未验证完整运行时及 GPU 位级确定性。

**D：数字对账**

顺序均为 **A17 / EdNet / Junyi**。J42＝[a17_ability_decompose_v3.json](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/results/a17_ability_decompose_v3.json)，J43＝[t1_fair_rescore.json](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/results/t1_fair_rescore.json)。

| 字段 | 核验值 | JSON 行号 |
|---|---|---|
| leakfree α=1 调参均值 | 0.5370 / 0.5974 / 0.6098 | J42:9、72、135 |
| leakfree α=0 调参均值 | 0.4797 / 0.4838 / 0.4542 | J42:5、68、131 |
| α=0 配对 delta | −0.1541 / −0.1887 / −0.1742 | J42:55、118、181 |
| 配对 n | 261 / 245 / 213 | J42:36、99、162 |
| isolated−leakfree，α=1 | +0.0284 / −0.0088 / +0.0006 | J42:21−9、84−72、147−135 |
| 调参评分行数 | 4136 / 2086 / 4121 | J42:12、75、138 |
| 调参有效学生数 | 269 / 212 / 173 | J42:13、76、139 |
| clean/fixed2 B1 均值 | 0.5485 / 0.5263 / 0.5668 | J42:34、97、160；J43:10、106、202 |
| sibling 最大漂移 | 0.0 / 0.0 / 0.0 | J43:7、103、199 |

其余裁决数字也全部吻合：v2 B1 锚 **0.5481/0.5252/0.5624**；KC 最大漂移 **0.0087/0.1667/0.1231**；B1 构造漂移 **+0.0004/+0.0011/+0.0044**；KC 数 **96/96/96、175/175/174、38/38/38**；对称闭包最大值 **0.5391/0.5570/0.5141**；全部 selfcheck 为 true。

batch43 的五个 transfer 显著格及 EdNet DKT 跨零格，其 delta/CI 均与 JSON 一致；12 格的 rc 与原字段共36对全部相等。历史 A17 **0.5512→0.5370** 的准确差是 **−0.0142**，报告写约−0.014合理。

**终审意见：修复逻辑与限定的选择结论通过；上述报告措辞修正后可通过终稿核验。**


