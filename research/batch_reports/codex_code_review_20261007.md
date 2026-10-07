**总 verdict：FAIL。** P0 针对“残余泄漏上界、因此可排除泄漏解释”这一主张；另有 B1 建表的信息泄漏、旧脚本错位和迁移评分支持集问题。AUC、置换公式及学生级 bootstrap 的核心运算大部分正确。

审计对象为当前工作树 `feat/llm-kt-plugin`，HEAD `ea8e12d`。全程只读，未修改文件。已执行源码函数与 JSON 的内存复算；本地缺原始数据、检查点、torch/polars，未重跑真实模型预测或完整历史 CI。

| 脚本 | Verdict | 主要依据 |
|---|---|---|
| `t1_paired_ci.py` | PASS-with-fixes | 学生配对及 bootstrap 正确；建表选择泄漏、迁移行集不一致、首位 dummy 路径 |
| `table_a1_fix.py` | PASS-with-fixes | 独立遍历修复了 zip；仍继承建表选择泄漏 |
| `a_bucket_kc_split.py` | PASS-with-fixes | A1/A2 判据正确；迁移首位评分路径需修正 |
| `a1_leftovers.py` | FAIL | Junyi 查表块仍有已发生的 zip 错位；DKT 分支无此问题 |
| `a1_copy_leakage.py` | FAIL〔上界用途〕 | copy 数字计算正确，但不能构成所声称的上界 |
| `llm_diff_formal.py` | PASS-with-fixes | 置换正确；空评分集可产生伪显著 p |
| `llm_pooled_visibility.py` | PASS | 当前有限值输入下统计实现正确 |
| `llm_pooled_pairshare.py` | PASS | 对分解恒等式正确 |
| `llm_pooled_power.py` | PASS | 符合固定置换族的条件 bootstrap 设计；解释需修正 |
| `llm_perstu_power.py` | PASS | 同上，且正确复用 batch28 的置换映射 |
| `a17_factorial_noise2.py` | PASS-with-fixes | 四格正确；异常集合和复现断言缺失 |

**逐条发现**

1. **P0｜“copy 攻击是泄漏上界”已被反向攻击否证。**  
   位置：[a1_copy_leakage.py:50](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a1_copy_leakage.py:50)、[PAPER_SECTIONS_V1.md:21](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/PAPER_SECTIONS_V1.md:21)。

   当前 tie=0.5 的 AUC 严格满足 `AUC(1−p)=1−AUC(p)`。JSON 中取反后，`copy_prev` 的 A17/EdNet AUC 为 **0.5296/0.5443**；反向 `copy_prev2` 为 **0.5391/0.5570**。因此，“最强确定性利用 ≤0.514”“残余贡献 <12%”“不可能解释主张”均没有成立的上界证明。反向收益也可能来自合法历史相关性，不能据此认定实际模型存在不可得标签泄漏。

2. **P1｜B1 建表成员取决于未来 holdout 标签。**  
   位置：[t1_paired_ci.py:53](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/t1_paired_ci.py:53)、[table_a1_fix.py:29](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/table_a1_fix.py:29)，根因：[baseline_eval.py:80](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/baseline_eval.py:80)。

   `build_rows` 先按 holdout 是否包含正负两类筛学生，随后只用这些学生的 evidence 拟合 prior。执行原函数的反例：两学生 evidence 分别全 1、全 0，prior 为 **0.5**；只将第一人的 holdout 改成单类，全部 evidence 保持不变，prior 变为 **0.0**。这是未来标签参与建表成员选择。应先固定 evidence 贡献队列，再进行 AUC 可评估性筛选；相关 B1、Δ 和 CI 需要重算，影响方向目前未知。

3. **P1｜`a1_leftovers` 的等长检查不能阻止学生错配。**  
   位置：[a1_leftovers.py:40](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a1_leftovers.py:40)。

   `students` 已过滤，`surviving` 未采用相同条件。原函数反例配成 `(ws.user_id, st.user_id)=(10,20),(20,30)`，两次长度检查都通过。实际旧 Junyi 结果为 **0.513、n=15**，独立遍历修复结果为 **0.5624、n=213**。当前底稿已废弃旧数字，但该脚本的 Junyi 块仍然错误。

4. **P1｜迁移分支可能把压缩序列首位 dummy 纳入评分。**  
   位置：[t1_paired_ci.py:100](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/t1_paired_ci.py:100)、[a_bucket_kc_split.py:85](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a_bucket_kc_split.py:85)。

   代码只排除原始 `i==0`，未排除映射后的 `jmap[i]==0`。若首个映射成功行落在 holdout，原始 `i>0`，压缩位置却为 0；[DKT_model.py:68](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/model/DKT/DKT_model.py:68) 此列明确是人为补的 **0**。两个脚本也未要求至少三个映射 evidence 行。代码路径已确认；当前数据中触发人数未验证。

5. **P1｜T1 配对了学生，没有配对评估行。**  
   位置：[t1_paired_ci.py:57](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/t1_paired_ci.py:57)、[t1_paired_ci.py:86](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/t1_paired_ci.py:86)。

   B1 使用每生全部 A1-excluded holdout；迁移模型使用映射成功且截尾覆盖的子集。bootstrap 正确估计了这两个不同支持集 AUC 的学生平均差，但不能直接解释为同一评估集上的模型优势。公平比较需要逐格在深度模型实际评分行上重算 B1。另从 JSON 可推 EdNet 迁移配对子集 B1 约 **0.5475**，展示的 **0.5252** 是全子集均值。

6. **P1｜空集可以伪造显著 p；NaN 会静默改变比较集合。**  
   位置：[llm_diff_formal.py:70](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/llm_diff_formal.py:70)、[llm_diff_formal.py:82](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/llm_diff_formal.py:82)、[transfer_eval.py:40](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/transfer_eval.py:40)。

   无有效学生时，真实量和全部置换量都是 NaN，比较全部为 False，得到 **p=1/201≈0.004975**，并可写入 JSON。公共 AUC 静默删除 NaN、保留 Inf；noise2 各格独立剔除无效学生，却只报告 TT 的 n，异常时四格可能比较不同学生。当前缓存全部 finite，11 个目标 JSON 无非有限值；本项是已复现的代码异常路径，未证明现有结果因此受污染。零分母经 `max(n,1)` 输出 0 的路径也应改为显式空值并报告人数。

7. **P1｜辅助机制链的 ability 调参把 pseudo-holdout 标签回代到预测。**  
   位置：[a17_ability_decompose_v2.py:44](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v2.py:44)、[a17_ability_decompose_v2.py:66](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_ability_decompose_v2.py:66)。

   先用全部 evidence 建 prior，再将其中每五行当调参留出集；该行自身标签已进入 prior。原函数构造复算得到 `alpha=1` 的选择集 AUC=1、`alpha=0` 为 0。该问题限于调参和机制负结论；不能依据当前 `alpha_star=1、delta=0` 可靠排除动态 ability。

8. **P1〔解释〕｜“功效趋于 α/1”的勘误与代码不符。**  
   位置：[llm_perstu_power.py:87](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/llm_perstu_power.py:87)、[llm_pooled_power.py:96](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/llm_pooled_power.py:96)，对应转述：[PAPER_SECTIONS_V1.md:63](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/PAPER_SECTIONS_V1.md:63)。

   两脚本固定 200 个置换，只重抽学生，得到的是**条件于现有学生、缓存和固定置换族的 bootstrap 拒绝频率**。无均值并列时，N 增大后 p 趋于固定值，拒绝率趋于 0 或 1。只读反例设置 12 个置换始终优于真实、188 个始终劣于真实，任何抽样均为 `p=13/201`，拒绝率恒为 0。代码符合其设计；α/1 解释需撤回。

9. **P2｜“逐位复现”没有自动断言，当前只能核到输入生成及聚合锚。**  
   位置：[a17_factorial_noise2.py:35](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_factorial_noise2.py:35)、[a17_factorial_noise2.py:42](/Users/wsy/项目/智能导学与知识追踪/UniKT/代码工作区/unikt-llm-kt/research/transfer/a17_factorial_noise2.py:42)。

   参考常量仅写入输出，未比较或 assert。已核 base1000 的 TT/FT/TF/FF：AKT **.5841/.5707/.5665/.5529**，DKT **.6331/.5958/.5497/.5398**，与被引批次的四位均值一致；两组合成输入也核实 b29 与 noise2 的噪声标签逐位相等。没有逐行预测产物，无法证明真实模型输出逐位复现。另 `:12` 注释称 synergy 等于 b33 residual，实际 `:98` 是其负值；批报已经说明该符号更正。

**通过的核查与边界**

- **A1 语义**：独立遍历脚本均按原始索引实现 `q[i]==q[i−1] && kc[i]!=kc[i−1]`；先跳过 0，同 KC 相邻保留。`a1_leftovers` 的错误来自学生错配。
- **AUC／置换**：正例分数较大计胜、tie=.5；穷举 **8,604** 个有限值样例，逐对法与 average-rank 法最大差为 **0**。置换对象是 KC→LLM 值指派，标签固定，真实与置换统计量使用同一函数，`(ge+1)/(B+1)` 正确。复算学生内 p 为 **.0995/.0448/.0647**，pooled 为 **.1244/.0249/.0149**。种子复用支持精确配对复算，不能当独立新实验。
- **Bootstrap**：学生交集、`B1−deep` 方向、学生差分重采样、5000 次及 2.5/97.5 百分位均正确；原函数与独立向量化复算一致。
- **预测信息集**：DKT 移位、AKT 因果掩码静态检查未见直接读取当前或未来响应。holdout 前缀用于后续预测属于 one-step online 协议，不能称整窗盲预测。LLM fallback 为固定表值或 .5，源码生成提示未输入作答标签。本域 AKT 也传 `question=None`；启用 Rasch 的检查点会省去题目分支，当前缺配置验证其影响。
- **2×2／pooled 分解**：loader 保证连续尾段 holdout，因此 `i<first_hold` 与 `i>=first_hold` 严格互补；四格始终对原始标签评分。pooled 对分解恒等式正确，within 项是按正负对数加权的 AUC；四位 JSON 的舍入精度不足以重验源码的 `1e−9` 门禁。


