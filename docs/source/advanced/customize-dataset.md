# 自定义数据集

在 UniKT 中添加自定义数据集，遵循"定义 → 注册 → 使用"三步法。数据集通过继承 ``DataSource`` 基类实现，使用 Polars 作为数据处理引擎。

## 数据处理管线

UniKT 的数据处理由 ``DataSource`` 基类（``utils/data_process/data_source.py``）统一管理，遵循标准化的六步管线：

```{mermaid}
flowchart LR
 A[下载原始数据] --> B[加载原始数据]
 B --> C[清洗]
 C --> D[转换为标准格式]
 D --> E[K 折划分]
 E --> F[序列化保存]
```

每一步都有明确的职责划分：

| 步骤 | 方法 | 职责 |
| --- | --- | --- |
| 下载 | ``fetch_data`` | 从 URL 下载压缩包，解压到 ``data/{dataset}/raw/`` |
| 加载 | ``load_src_data``（抽象） | 读取原始文件（CSV、JSON 等）到 Polars DataFrame |
| 清洗 | ``clean_raw_data``（抽象） | 过滤无效行、处理缺失值、标准化列名 |
| 转换 | ``transform_data``（抽象） | 构建 ``sequence_data``（用户交互序列）+ ``relation_data``（关系表） |
| 划分 | ``add_kfold_labels`` | 按用户划分 K 折标签 |
| 保存 | ``save_data`` | 写入 Parquet 文件 + MD5 校验 + metadata.json |


## 第一步：定义数据源

创建数据源类，继承 ``DataSource``，实现三个核心抽象方法。它们按固定顺序被调用：``load_src_data`` → ``clean_raw_data`` → ``transform_data``。

```python
# utils/data_process/my_dataset.py
from typing_extensions import override
import os
import polars as pl
from utils.core import get_logger, register_data_source
from utils.data_process.data_source import (
 DataSource,
 exclude_short_sequences,
)

logger = get_logger(__name__)


@register_data_source("my_dataset")
class MyDatasetProcessor(DataSource):
 """自定义数据集处理器。"""

 def __init__(self, args):
 super.__init__(
 dataset="my_dataset",
 data_base_path=args.data_base_path,
 seed=args.seed,
 )
 self.args = args
 self.raw_data_path = f"{self.data_folder}/raw/my_dataset.csv"

 @override
 def load_src_data(self):
 if not os.path.exists(self.raw_data_path):
 raise FileNotFoundError(f"找不到数据文件: {self.raw_data_path}")
 logger.info(f"加载原始数据: {self.raw_data_path}")
 self.raw_data = pl.read_csv(self.raw_data_path, ignore_errors=True).lazy

 @override
 def clean_raw_data(self):
 if self.raw_data is None:
 self.load_src_data
 data = self.raw_data.collect

 # 1. 列名映射：自定义列 → 标准列
 data = data.rename({
 "correct": "label",
 "user_id": "user",
 "problem_id": "question",
 "skill_id": "skill",
 })

 # 2. 过滤无效数据
 data = data.filter(
 pl.col("user").is_not_null
 & pl.col("question").is_not_null
 & pl.col("label").is_not_null
 )

 # 3. 转换时间戳为相对时间（毫秒）
 data = data.with_columns(
 pl.col("timestamp")
 .cast(pl.Int64)
 .sub(pl.col("timestamp").cast(pl.Int64).min)
 .alias("timestamp")
 )

 # 4. 按用户和时间排序
 data = data.sort(["user", "timestamp"])

 # 5. 剔除过短序列
 data = exclude_short_sequences(data, self.args.min_seq_len)

 self.cleaned_raw_data = data

 @override
 def transform_data(self):
 if self.cleaned_raw_data is None:
 raise ValueError("clean_raw_data 必须在 transform_data 之前调用")

 # 1. 构建 ID 映射（将原始字符串 ID 转换为从 0 开始的连续整数）
 self._build_id_mapping(self.cleaned_raw_data, ["user", "question", "skill"])

 # 2. 构建关系表：question_skill
 question_skill = self.cleaned_raw_data.select(["question", "skill"]).unique(
 subset=["question", "skill"]
 )
 self._apply_id_mapping(question_skill, ["question", "skill"])

 # 3. 构建序列数据
 self.sequence_data = self.cleaned_raw_data.select(
 ["user", "question", "label", "timestamp"]
 )
 self._apply_id_mapping(self.sequence_data, ["user", "question"])

 # 4. 存储关系表
 self.relation_data = {"question_skill": question_skill}
```

### 核心方法职责

| 方法 | 职责 | 设置属性 |
| --- | --- | --- |
| ``load_src_data`` | 从磁盘/网络加载原始数据 | ``self.raw_data``（Polars LazyFrame） |
| ``clean_raw_data`` | 清洗、过滤、排序、重命名列 | ``self.cleaned_raw_data``（Polars DataFrame） |
| ``transform_data`` | 构建 ID 映射、关系表、序列数据 | ``self.sequence_data``、``self.relation_data`` |


### 基类提供的工具方法

| 方法 | 用途 |
| --- | --- |
| ``_build_id_mapping(data, columns)`` | 为指定列生成 {原始值 → 连续整数} 的映射字典 |
| ``_apply_id_mapping(data, columns)`` | 将映射应用到 DataFrame，列类型转为 ``Int32`` |
| ``save_data`` | 保存处理结果到 Parquet，含 MD5 校验 |
| ``fetch_data`` | 从 URL 下载压缩包并解压 |
| ``get_metadata`` | 返回数据集元信息（技能数、题目数等） |


这两个 ID 映射方法确保所有 ID 在 ``[0, N-1]`` 范围内且连续，模型可以直接用它们做 Embedding 索引。

### 关系表规范

``self.relation_data`` 必须包含 ``question_skill``（题目-技能映射，列为 ``question`` 和 ``skill``），可选 ``question_assignment``（题目-作业映射）、``question_template``（题目-模板映射）。每个关系表为 2 列 DataFrame，且 2 列的组合必须唯一。

### 数据格式要点

- 全程使用 **Polars** DataFrame/Parquet（非 pandas）
- 原始数据建议用 ``lazy`` 加载以节省内存
- ``sequence_data`` 必须包含：``user``、``question``、``label``、``timestamp``
- 所有 ID 列需要重映射为 ``[0, N-1]`` 的连续整数
- ``timestamp`` 应为相对时间（毫秒级 Unix 时间戳或序号）

### 下载与序列化

``fetch_data``、``save_data``、``add_kfold_labels`` 和 ``build_split_question_sequence_data`` / ``build_split_skill_sequence_data`` 由基类提供，子类通常无需重写。

**``save_data``** 将序列数据和关系表写入 Parquet 文件，同时执行数据一致性验证：

1. **关系表校验**：每个关系表必须恰好 2 列，且 2 列的组合唯一
2. **``question_skill`` 必须存在**，且其列必须是 ``question`` 和 ``skill``
3. **题目 ID 交叉校验**：``sequence_data`` 中的每个 ``question`` 值必须在 ``question_skill`` 中存在对应条目
4. **MD5 校验**：每份输出的 Parquet 文件自动计算 MD5 写入 ``metadata.json``

验证失败会抛出 ``AssertionError``，阻止不完整的数据进入训练管线。

## 第二步：注册数据源

使用 ``@register_data_source("NAME")`` 装饰器注册。装饰器参数必须是字符串字面量，不能使用变量。

注册后，将新文件导入到 ``utils/data_process/__init__.py`` 中：

```python
# utils/data_process/__init__.py
from .my_dataset import MyDatasetProcessor  # 新增这一行
```

这一步是必需的——框架启动时会导入 ``utils.data_process`` 包，触发 ``@register_data_source`` 装饰器的执行。

## 第三步：使用数据集

自定义数据集注册后，使用方式与内置数据集完全一致。

### 下载数据

```bash
# 下载原始数据
python data_process.py download -d my_dataset

# 强制重新下载（覆盖已有文件）
python data_process.py download -d my_dataset --force

# 自定义下载参数
python data_process.py download -d my_dataset --max_retries 5 --num_threads 8
```

### 预处理数据

```bash
# 使用默认参数处理
python data_process.py process -d my_dataset

# 自定义序列长度和折数
python data_process.py process \
 -d my_dataset \
 --min_seq_len 3 \
 --max_seq_len 200 \
 --kfold 5 \
 --test_ratio 0.2 \
 --seed 42
```

处理完成后在 ``data/my_dataset/`` 下生成：

```
data/my_dataset/
├── my_dataset_sequence.parquet # 完整交互序列
├── my_dataset_split_question_sequence.parquet # 按题目粒度切分后的序列
├── my_dataset_split_skill_sequence.parquet # 按技能粒度切分后的序列
├── my_dataset_relation_question_skill.parquet # 题目-技能关系表
├── metadata.json # 处理元数据
└── raw/ # 原始数据文件
 └── interactions.csv
```

### 训练模型

```bash
# 基本训练
python train.py -m GIKT -d my_dataset

# K 折交叉验证
python train.py -m GIKT -d my_dataset --fold 0
```

## 预测位提取

K 折划分的核心规则是**以用户为粒度划分**，而非按交互时间点。``add_kfold_labels`` 的实现逻辑如下：

```python
# add_kfold_labels 位于 data_source.py
unique_users = self.sequence_data["user"].unique.sort
num_users = len(unique_users)
num_test_users = int(num_users * test_ratio)

# 随机打乱用户ID顺序
user_indices = np.arange(num_users)
self._np_rng.shuffle(user_indices)

# 测试集：前 num_test_users 个用户，标记 fold = -1
non_test_indices = user_indices[num_test_users:]
fold_assignment = np.full(num_users, -1)

# 训练/验证集：剩余用户用 KFold 分配 fold 0 ~ K-1
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
for fold_idx, (_, val_indices) in enumerate(kfold.split(non_test_indices)):
    fold_assignment[non_test_indices[val_indices]] = fold_idx

# 通过 user 列 join 到序列数据上
self.sequence_data = self.sequence_data.join(user_fold_map, on="user", how="left")
```

**关键防护点：**

- 划分的最小单位是用户（``user``），不是单次交互。一个用户的所有历史交互必然整块落入同一折
- 训练时，当前折的验证集用户在构建关系矩阵和计算题目难度时被显式排除
- 随机种子固定（``--seed 42``）：使用独立的 ``np.random.RandomState`` 确保同一数据集不同次运行的折分配一致
- ``fold = -1`` 的测试集用户在训练过程中完全不可见，仅在最终测试评估阶段出现

:::{tip}
如果需要在数据划分前对部分用户进行子采样，应在 ``clean_raw_data`` 或 ``transform_data`` 中完成，确保采样发生在折标签分配之前。
:::

## 序列切分与 ID 重映射

基类的 ``build_split_question_sequence_data`` 和 ``build_split_skill_sequence_data`` 负责将长用户序列切分为固定长度的子序列：

1. 按 ``max_seq_len`` 将每个用户的交互序列切分为多个子序列
2. 丢弃长度不足 ``min_seq_len`` 的切分
3. 为每个保留的子序列分配新的稠密用户 ID（从 0 开始连续编号）

例如，用户 U1 有 450 条交互，``max_seq_len=200``、``min_seq_len=3``：

- 切分 1：位置 0~199（200 条）→ 新用户 ID 0
- 切分 2：位置 200~399（200 条）→ 新用户 ID 1
- 切分 3：位置 400~449（50 条，≥ 3 保留）→ 新用户 ID 2

切分按批次处理（每批约 200 万行），批次边界对齐到用户边界，确保同一个用户不会被拆分到不同批次中。

## 进阶：缓存控制

对于大型数据集，可以在 ``load_src_data`` 中使用 lazy 模式加载：

```python
@override
def load_src_data(self):
    self.raw_data = pl.scan_csv(
        self.raw_data_path,
        ignore_errors=True,
    )
    self._data_cache[("raw", None)] = self.raw_data
```

## 完整示例：SimpleKT

以下是一个虚构数据集 ``simple_kt`` 的完整实现，包含两个文件 ``interactions.csv`` 和 ``skills.csv``：

```
simple_kt.zip
├── interactions.csv # user_id, item_id, correct, timestamp
└── skills.csv # item_id, skill_id
```

```python
# utils/data_process/simple_kt.py
import os
import polars as pl
from utils.core import get_logger, register_data_source
from .data_source import DataSource

logger = get_logger(__name__)


@register_data_source("simple_kt")
class SimpleKTData(DataSource):
 def __init__(self, args):
 super.__init__(
 dataset="simple_kt",
 data_base_path=args.data_base_path,
 data_url="https://example.com/simple_kt.zip",
 seed=args.seed,
 )
 self.args = args
 self.interactions_path = os.path.join(
 self.data_folder, "raw", "interactions.csv"
 )
 self.skills_path = os.path.join(self.data_folder, "raw", "skills.csv")

 def load_src_data(self):
 if not os.path.exists(self.interactions_path):
 raise FileNotFoundError(f"Cannot find: {self.interactions_path}")
 if not os.path.exists(self.skills_path):
 raise FileNotFoundError(f"Cannot find: {self.skills_path}")

 logger.info(f"Loading raw data from: {self.interactions_path}")
 self._raw_interactions = pl.read_csv(self.interactions_path)
 self._raw_skills = pl.read_csv(self.skills_path)

 def clean_raw_data(self):
 skill_items = set(self._raw_skills["item_id"].to_list)
 cleaned = self._raw_interactions.filter(
 pl.col("correct").is_not_null & pl.col("item_id").is_in(skill_items)
 )
 logger.info(
 f"Cleaned: {len(cleaned)} interactions "
 f"({len(self._raw_interactions) - len(cleaned)} removed)"
 )
 self.cleaned_raw_data = cleaned
 return cleaned

 def transform_data(self):
 if self.cleaned_raw_data is None:
 raise ValueError("clean_raw_data must be called before transform_data")

 # 1. 构建 ID 映射
 self._build_id_mapping(self.cleaned_raw_data, ["user_id", "item_id"])
 self._build_id_mapping(self._raw_skills, ["item_id", "skill_id"])

 # 2. 构建关系表：question_skill
 question_skill = (
 self._raw_skills.select(["item_id", "skill_id"])
 .rename({"item_id": "question", "skill_id": "skill"})
 .unique(subset=["question", "skill"])
 )
 self._apply_id_mapping(question_skill, ["question", "skill"])

 # 3. 构建序列数据
 self.sequence_data = self.cleaned_raw_data.select(
 ["user_id", "item_id", "correct", "timestamp"]
 ).rename({
 "user_id": "user",
 "item_id": "question",
 "correct": "label",
 })
 self._apply_id_mapping(self.sequence_data, ["user", "question"])

 # 4. 存储关系表
 self.relation_data = {"question_skill": question_skill}

 logger.info(
 f"Transformed: {len(self.sequence_data)} interactions, "
 f"{self.sequence_data['user'].n_unique} users, "
 f"{self.sequence_data['question'].n_unique} questions, "
 f"{question_skill['skill'].n_unique} skills"
 )
```

使用方式：

```bash
# 下载
python data_process.py download -d simple_kt

# 预处理
python data_process.py process -d simple_kt --min_seq_len 3 --max_seq_len 100

# 训练
python train.py -m GIKT -d simple_kt --fold 0
```

## 注意事项

- ``self.cleaned_raw_data`` 在调用 ``clean_raw_data`` 前为 ``None``，``transform_data`` 中必须检查。
- ID 映射必须在 ``transform_data`` 中完成，``save_data`` 会写入 Parquet 文件。
- ``metadata.json`` 由 ``save_data`` 自动生成，包含文件 MD5 校验和与统计信息。
- 数据集名建议使用小写字母 + 下划线（如 ``assistments09``、``my_dataset``）。
- 如果自定义训练器需要额外关系表（如 ``question_template``），在 ``transform_data`` 中补充即可。
