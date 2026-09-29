# 数据预处理

UniKT 目前支持 11 个公开数据集的下载和统一预处理，数据集的下载链接可以在附录中找到。

## 预处理流水线

```{mermaid}
flowchart LR
 A[原始数据] --> B[下载]
 B --> C[清洗]
 C --> D[序列化]
 D --> E[K 折划分]
```

## 快速上手

### 下载数据

```bash
# 从源下载
python data_process.py download -d assistments09

# 强制重新下载
python data_process.py download -d assistments09 --force
python data_process.py download -d assistments09 -f
```

下载参数：

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--force`` | False | 即使文件已存在也强制重新下载 |
| ``--max_retries`` | 3 | 最大下载重试次数 |
| ``--num_threads`` | 4 | 并行下载线程数 |
| ``--data_url`` | None | 覆盖数据下载 URL |


### 处理数据

:::{tip}
非特殊情况下，推荐使用默认参数进行预处理。默认参数已经被设置为通用参数。
:::

```bash
# 使用默认设置处理
python data_process.py process -d assistments09

# 自定义参数
python data_process.py process \
 -d assistments09 \
 --min_seq_len 3 \
 --max_seq_len 200 \
 --kfold 5 \
 --test_ratio 0.2 \
 --seed 42
```

**处理参数：**

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--min_seq_len`` | 3 | 最小序列长度 |
| ``--max_seq_len`` | 200 | 最大序列长度 |
| ``--kfold`` | 5 | K 折交叉验证的折数 |
| ``--test_ratio`` | 0.2 | 测试集比例 |
| ``--seed`` | 42 | 随机种子（用于可复现性） |


**采样参数：**

| 参数 | 默认值 | 描述 |
| --- | --- | --- |
| ``--sample_size`` | None | 绝对采样数量。random/stratified 模式为用户数，time 模式为交互数（None 表示禁用） |
| ``--sample_ratio`` | None | 采样比例（0.0-1.0）。设置后将覆盖 ``sample_size``。random/stratified 模式为用户的比值，time 模式为交互的比值 |
| ``--sample_strategy`` | random | 采样策略（random、stratified、time） |
| ``--sample_attempts_bins`` | 20 100 | 分层采样的答题次数分箱边界 |
| ``--sample_correct_bins`` | 0.4 0.8 | 分层采样的正确率分箱边界 |


:::{warning}
``--sample_size`` 和 ``--sample_ratio`` 互斥，只能使用其中一个。
:::

**采样策略说明：**

- ``random``：随机采样 N 个用户
- ``stratified``：基于用户答题次数和正确率进行分层采样
- ``time``：按时间戳排序，取最早的 N 条记录

```bash
# 随机采样 1000 个用户
python data_process.py process -d assistments09 --sample_size 1000

# 按时间采样 10% 的交互（最早的优先）
python data_process.py process -d assistments09 --sample_ratio 0.1 --sample_strategy time

# 使用自定义分箱进行分层采样
python data_process.py process -d assistments09 --sample_size 500 --sample_strategy stratified
```

**额外处理：**

| 参数 | 描述 |
| --- | --- |
| ``--extra+ windowlate`` | 为 KC 模型的滑动窗口测试构建 windowlate 数据 |

``+`` 是 jsonargparse 列表参数的官方追加语法，可重复出现（``--extra+ a --extra+ b``）；也接受 JSON 数组形式（``--extra '["windowlate"]'``）。


## 输出结构

处理完成后，在 ``data/{dataset}/`` 下生成以下文件：

```
data/{dataset}/
├── {dataset}_sequence.parquet # 用户交互序列
├── {dataset}_split_question_sequence.parquet # 按折划分的题目交互序列
├── {dataset}_split_skill_sequence.parquet # 按折划分的技能交互序列
├── {dataset}_relation_question_skill.parquet # 题目-技能关系表
├── {dataset}_relation_question_assignment.parquet # 题目-作业关系表（可选）
├── {dataset}_relation_question_template.parquet # 题目-模板关系表（可选）
├── {dataset}_windowlate.parquet # 滑动窗口数据（可选）
└── metadata.json # 处理元数据
```

### 文件说明

| 文件 | 描述 |
| --- | --- |
| ``*_sequence.parquet`` | 用户交互序列 |
| ``*_split_question_sequence.parquet`` | 按 K 折划分的题目交互序列 |
| ``*_split_skill_sequence.parquet`` | 按 K 折划分的技能交互序列 |
| ``*_relation_question_skill.parquet`` | 题目-技能映射关系表 |
| ``*_relation_question_assignment.parquet`` | 题目-作业映射关系表（如有） |
| ``*_relation_question_template.parquet`` | 题目-模板映射关系表（如有） |
| ``*_windowlate.parquet`` | 为滑动窗口训练预构建的窗口数据 |
| ``metadata.json`` | 处理参数和文件校验和 |


:::{note}
题目序列 vs 技能序列的差异：

+--------------+-----------------------+----------------------------------------+
| 类型         | 粒度                  | 用途                                   |
+==============+=======================+========================================+
| **题目序列** | 一次交互 = 一个时间步 | 预测特定题目的表现                     |
+--------------+-----------------------+----------------------------------------+
| **技能序列** | 一个技能 = 一个时间步 | 预测技能掌握度（多技能题目展开为多行） |
+--------------+-----------------------+----------------------------------------+

对于多技能题目（例如标记了技能 ``2``、``37``、``70`` 的题目）：

- **题目序列**：单条目，``question=123``
- **技能序列**：三条记录，``skill=[2, 37, 70]``，``question=123``，``label`` 和其他字段相同

技能序列保留了 ``question`` 列，因此需要题目 ID（如 AKT Rasch）的模型可以直接使用。

   当存在多技能题目时，技能序列比题目序列更长。
:::

### 序列数据字段

**划分后的题目序列**（``*_split_question_sequence.parquet``）：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| ``user`` | ``Int32`` | 用户标识符（已重映射，划分为子序列） |
| ``question`` | ``Int32`` | 题目标识符 |
| ``label`` | ``Int8`` | 回答正确性（0 或 1） |
| ``timestamp`` | ``Int64`` | Unix 时间戳（毫秒） |
| ``fold`` | ``Int32`` | K 折标签（-1 = 测试集，0..n_splits-1 = 训练/验证集） |
| ``seq_pos`` | ``Int32`` | 子序列中的位置 |
| ``attempt_count`` | ``Int32`` | 该题目的答题次数（如有） |
| ``hint_count`` | ``Int32`` | 使用的提示次数（如有） |


**划分后的技能序列**（``*_split_skill_sequence.parquet``）：

包含上述题目序列的所有列，外加：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| ``skill`` | ``Int32`` | 技能/概念标识符（多技能题目已展开） |
| ``question`` | ``Int32`` | 原始题目标识符（展开后保留） |


**滑动窗口数据**（``*_windowlate.parquet``）：

长格式数据，每行对应一个（样本，位置）对：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| ``sample_id`` | ``Int64`` | 唯一样本标识符 |
| ``position`` | ``Int32`` | 窗口内的位置 |
| ``skill`` | ``Int32`` | 技能/概念标识符 |
| ``question`` | ``Int32`` | 题目标识符 |
| ``response`` | ``Int8`` | 正确性（0 = 错误，1 = 正确，目标位置为 0） |
| ``mask`` | ``Int8`` | 预测掩码（1 = 预测该位置） |
| ``user_id`` | ``Int32`` | 原始用户标识符 |
| ``group_id`` | ``Int64`` | 题目级分组 ID |
| ``true_label`` | ``Int8`` | 用于评估的真实标签 |
| ``fold`` | ``Int32`` | K 折分配 |


## 列映射

每个数据集的原始列名不同。下表展示了处理过程中如何将原始列映射为标准列名。

### 标准输出列

| 列名 | 描述 |
| --- | --- |
| ``user`` | 用户标识符（重映射为连续整数） |
| ``question`` | 题目标识符（重映射为连续整数） |
| ``skill`` | 技能/概念标识符（重映射为连续整数） |
| ``assignment`` | 作业/任务标识符 |
| ``template`` | 题目模板标识符 |
| ``label`` | 回答正确性（0 或 1） |
| ``attempt_count`` | 该题目的答题次数 |
| ``hint_count`` | 使用的提示次数 |
| ``timestamp`` | Unix 时间戳（毫秒）或顺序序号 |
| ``fold`` | K 折标签（-1 为测试集，0 到 n_splits-1 为训练/验证集） |


### ASSISTments 2009

| 原始列 | 标准列 |
| --- | --- |
| ``user_id`` | ``user`` |
| ``problem_id`` | ``question`` |
| ``correct`` | ``label`` |
| ``skill_id`` | ``skill`` |
| ``assignment_id`` | ``assignment`` |
| ``template_id`` | ``template`` |
| ``order_id`` | ``timestamp`` |


**说明：**

- 技能按 ``_`` 分割（如 ``"2_37_70"`` → ``["2", "37", "70"]``）
- 多技能题目在题目数据中展开为多行。

### ASSISTments 2012

| 原始列 | 标准列 |
| --- | --- |
| ``user_id`` | ``user`` |
| ``problem_id`` | ``question`` |
| ``correct`` | ``label`` |
| ``skill_id`` | ``skill`` |
| ``assignment_id`` | ``assignment`` |
| ``template_id`` | ``template`` |
| ``start_time`` | ``timestamp`` |


**说明：**

- ``start_time`` 解析为日期时间并转换为 Unix 毫秒
- 技能为 null 的行会被过滤

### ASSISTments 2017

| 原始列 | 标准列 |
| --- | --- |
| ``studentId`` | ``user`` |
| ``problemId`` | ``question`` |
| ``correct`` | ``label`` |
| ``skill`` | ``skill`` |
| ``assignmentId`` | ``assignment`` |
| ``hintCount`` | ``hint_count`` |
| ``attemptCount`` | ``attempt_count`` |
| ``startTime`` | ``timestamp`` |


**说明：**

- 该数据集没有模板列
- 技能为 null 的行会被过滤

### EdNet-KT1

| 原始列 | 标准列 |
| --- | --- |
| ``user_id``（来自文件名） | ``user`` |
| ``question_id`` | ``question`` |
| ``user_answer`` + ``correct_answer`` | ``label``（计算得到） |
| ``timestamp`` | ``timestamp`` |
| ``tags`` | ``skill`` 和 ``template`` |
| ``bundle_id`` | ``assignment`` |


**说明：**

- ``label`` 通过比较 ``user_answer`` 和 ``correct_answer`` 计算得到
- ``attempt_count`` 和 ``hint_count`` 为占位值（分别设为 1 和 0）
- 标签按 ``;`` 分割（如 ``"algebra;geometry"`` → ``["algebra", "geometry"]``）
- 分割前会对标签进行去重和排序

## 注意事项

- **预测位提取**：K 折划分保留序列内的时间顺序
- **内存**：大数据集（EdNet）处理时需要较大内存
- **幂等性**：重新运行 ``process`` 会覆盖之前的结果

## 附录

### 支持的数据集

| 数据集 | 来源 |
| --- | --- |
| ``algebra2005`` | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/) |
| ``algebra2006`` | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/) |
| ``assistments09`` | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ``assistments12`` | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ``assistments15`` | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ``assistments17`` | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ``bridge2006`` | [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/) |
| ``ednet_kt1`` | [GitHub](https://github.com/riiid/ednet) |
| ``junyi2015`` | [Junyi Academy](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset) |
| ``nips2020_t34`` | [NeurIPS 2020 Education Challenge](https://eedi.com/projects/neurips-education-challenge) |
| ``slepemapy`` | [SLEP](https://www.fi.muni.cz/adaptivelearning/) |
