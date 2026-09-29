# 数据流详解

从命令行参数到模型输出的端到端数据流：CLI 解析 → 数据加载 → 训练循环 → 评估 → 输出。本文基于 UniKT 源码，所有代码引用可追溯到具体文件和行号。

## 概述

```{mermaid}
flowchart TB
 A[CLI 入口] --> B[参数解析]
 B --> C[数据源加载]
 C --> D[模型数据准备]
 D --> E[训练器构建]
 E --> F[训练循环]
 F --> G[测试评估]
 G --> H[结果输出]

 style A fill:#e1f5fe
 style D fill:#fff9c4
 style F fill:#c8e6c9
 style H fill:#f3e5f5
```

## CLI 入口层

训练入口通常为项目根目录的 ``train.py``（或 ``optuna_search.py`` 用于超参数搜索，``case_analysis.py`` 用于案例分析）。以标准训练为例，用户执行：

```bash
python train.py -m GIKT -d assistments09 --fold 0 --batch_size 64 --epochs 100
```

### 参数解析流程

命令行参数由两层构成：**通用参数组**（无条件加载）和**模型专属参数组**（按 ``-m`` 指定的模型动态注入）。

```{mermaid}
flowchart LR
 A[train.py] --> B[DataParams]
 A --> C[GeneralParams]
 A --> D[EarlyStoppingParams]
 A --> E[CompileParams]
 A --> F[SamplingParams]
 F --> G[ArgumentParser]
 B --> G
 C --> G
 D --> G
 E --> G
 G --> H["get_model_params('GIKT').add_args(parser)"]
 H --> I[GIKTModelParams.add_args]
 I --> J[完整参数解析]
```

通用参数组定义在 ``utils/config/param_config.py``：

- **DataParams**（）：``--dataset``、``--fold``、``--kfold``、``--max_seq_len``、``--data_base_path`` 等
- **GeneralParams**（）：``--device``、``--seed``、``--log_dir``、``--no_swanlab``、``--deterministic`` 等
- **EarlyStoppingParams**（）：``--es_monitor``、``--es_mode``、``--es_patience``、``--es_min_delta``
- **CompileParams**（）：``--compile``、``--compile_mode``、``--compile_backend`` 等
- **SamplingParams**（）：``--sample_size``、``--sample_strategy`` 等

每个参数组都是 ``BaseParamConfig`` 的子类，通过 ``define_params`` 方法定义参数规范（类型、默认值、帮助文本），由 ``add_args(parser)`` 类方法注入到 ``argparse.ArgumentParser``。``bool`` 类型自动映射为 ``store_true`` 或 ``store_false`` action（``param_config.py`` ）。

模型专属参数以 GIKT 为例，在 ``model/GIKT/GIKT_trainer.py`` 中：

```python
from utils.config import BaseParamConfig, register_model_params


@register_model_params("GIKT")
class GIKTModelParams(BaseParamConfig):
 def define_params(self) -> tuple[str, dict]:
 return "GIKT Parameters", {
 "n_hop": {
 "type": int,
 "default": 3,
 "short": "nh",
 "help": "Number of GNN aggregation hops (default: 3)",
 },
 "embedding_dim": {
 "type": int,
 "default": 100,
 "short": "ed",
 "help": "Embedding dimension (default: 100)",
 },
 "batch_size": {
 "type": int,
 "default": 32,
 "short": "bs",
 "help": "Batch size (default: 32)",
 },
 "learning_rate": {
 "type": float,
 "default": 0.001,
 "short": "lr",
 "help": "Learning rate (default: 0.001)",
 },
 }
```

``train.py`` 先加载通用参数组，再根据 ``-m`` 指定的模型名从已注册参数中获取对应参数类并调用 ``add_args(parser)``，实现**通用参数 + 模型参数**的分层解析。

## 数据加载层

数据流进入模型训练器之前需要经过两步数据准备：**原始数据加载与清洗**（DataSource）和**模型级数据切分与构造**（ModelData）。

### 第一步：DataSource 管线

数据源系统由 ``utils/data_process/`` 包实现，管线在 ``DataSource`` 基类（``data_source.py`` ）中定义：

```
fetch_data → load_src_data → clean_raw_data → transform_data → save_data
```

每个步骤的作用：

- ``load_src_data``（）：从磁盘加载原始数据（CSV、Parquet 或自定义格式），**抽象方法，每个数据集自行实现**
- ``clean_raw_data``（）：去除无效记录、统一列名、处理缺失值
- ``transform_data``（）：转换为标准格式，产出 5 类数据：
- ``sequence_data``：用户交互序列（user / question / label / timestamp）
- ``split_question_sequence``：按知识点切分后的问题级序列
- ``split_skill_sequence``：按知识点切分后的技能级序列
- ``windowlate_data``：滑动窗口评估数据
- ``relation_data/*``：关系表家族（``question_skill``、``question_assignment``、``question_template``）
- ``save_data``（）：保存为 Parquet 并执行数据一致性校验（MD5 哈希、question_id 交叉校验、列检查）

**注意：** UniKT 全链路使用 **Polars** DataFrame 和 Parquet 格式，不使用 pandas。与 pandas 最大的差异是 Polars 没有行索引，所有数据操作通过列名和表达式完成。

数据集通过 ``@register_data_source`` 装饰器注册到 ``DATA_SOURCES`` 全局表。导入 ``utils.data_process`` 时触发 AST 静态发现（``__init__.py`` ），扫描 ``data_process/`` 下所有 ``.py`` 的 ``@register_data_source("数据集名")`` 装饰器，建立懒索引。

使用时通过工厂函数获取：

```python
from utils.data_process import get_data_source

data_src = get_data_source("assistments09", args)
```

``get_data_source``（``__init__.py`` ）内部查找已注册的数据源并实例化。

当前支持 11 个数据集：``algebra2005``、``algebra2006``、``assistments09/12/15/17``、``bridge2006``、``ednet_kt1``、``junyi2015``、``nips2020_t34``、``slepemapy``。

### 第二步：ModelData 准备

DataSource 产出的是原始交互序列，ModelData 负责将其转换为模型可消费的训练/验证/测试张量。基类 ``BaseModelData``（``utils/model_data/base_model_data.py`` ）提供以下通用能力：

- ``prepare_data(args)``：**抽象方法**，每个模型子类实现，是数据准备的入口
- ``split_kfold_data(*arrays, fold_idx)``（）：基于用户级别的 K-fold 划分，同一个用户的所有交互不会跨 fold
- ``split_data(*arrays, val_ratio, test_ratio)``（）：随机划分（非 K-fold 模式）
- ``calculate_question_difficulty(exclude_fold)``（）：使用正确率 × 置信度 + 回归均值计算题目难度
- ``build_relationship_matrix(edge_type)``（）：构建二值或计数关系矩阵（用于图模型）
- ``disk_cache``（）：基于 pickle 的磁盘缓存装饰器，通过 ``--cache`` 启用

UniKT 提供两个平行的 ModelData 分支，服务不同粒度的模型：

**SkillModelData — KC 级模型**（``utils/model_data/skill_model_data.py`` L1）

面向需要技能（Knowledge Component）序列的模型（如 DKT、GIKT）。``build_sequence_data``（）从 ``data_src.get_split_skill_sequence_data`` 构建技能序列，产出 5 个等长数组（shape 均为 ``[num_users, max_skill_seq_len]``）：

- ``user_sequence``：技能（KC）ID 序列
- ``user_response``：作答正误（0/1）
- ``user_mask``：有效位置掩码（padding 位置为 0）
- ``user_id_sequence``：用户 ID 序列
- ``user_question``：题目 ID 序列

```python
# 以 GIKT 为例，在 model/GIKT/GIKT_data.py 中：
from utils.model_data import SkillModelData


class GIKTModelData(SkillModelData):
 def prepare_data(self, args):
 # 1. 获取技能级序列数据
 data_src.get_split_skill_sequence_data

 # 2. 构建序列数组
 seqs = self.build_sequence_data
 user_sequence, user_response, user_mask, user_id_sequence, user_question = seqs

 # 3. K-fold 划分
 train_data, val_data, test_data = self.split_kfold_data(seqs, args.fold)
 return train_data, val_data, test_data, ...
```

**QuestionModelData — Question 级模型**（``utils/model_data/question_model_data.py`` L1）

面向需要题目级交互图和异构图数据的模型（如 HDHKT、GIKT 图分支）。``load_sequence_data``（）从 ``data_src.get_split_question_sequence_data`` 构建题目序列，产出 4 个数组。额外提供图构建能力：

- ``build_hetero_graph(edge_types, edge_attrs, directed, node_features)``（）：构建 PyTorch Geometric ``HeteroData``
- ``build_hyper_graph(edge_type, vertex_type)``（）：构建 DHG ``Hypergraph``

**K-fold 划分机制**（``base_model_data.py`` ）：

```python
# _build_user_folds 的核心逻辑
def _build_user_folds(self, num_users):
    # 1. 生成 0..num_users-1 的随机排列
    indices = torch.randperm(num_users)
    # 2. 按 fold 数量等分用户
    # 3. 建立 user_id -> fold 的映射表
```

划分是 **用户级别** 的——同一用户的所有交互序列完整落入一个 fold，避免数据泄露。这个设计保证验证集和测试集中的用户是训练阶段从未见过的。

## 训练循环层

### 训练器构建：链式 Builder 模式

每个模型训练器（例如 ``GIKTTrainer`` 在 ``model/GIKT/GIKT_trainer.py`` ）在 ``__init__`` 中通过链式调用构建训练环境：

```python
@register_trainer("GIKT")
class GIKTTrainer(BaseTrainer):
 def __init__(self, args, data_src, exp_manager):
 model_data = GIKTModelData(data_src)
 train_data, val_data, test_data, ... = model_data.prepare_data(args)

 model = GIKT(args=args, data_metadata=data_src.get_metadata)
 super.__init__(model)

 optimizer = torch.optim.Adam(model.parameters, lr=args.learning_rate)

 # 链式构建：每个 with_* 返回 self，支持连续调用
 self.with_training( # 训练配置
 epochs=args.epochs, seed=args.seed,
 device=args.device, checkpoint_path=args.checkpoint_path,
 ).with_data( # 数据配置
 train_data=train_data, val_data=val_data, test_data=test_data,
 batch_size=args.batch_size,
 ).with_optimization( # 优化配置
 optimizer=optimizer, loss_fn=torch.nn.BCEWithLogitsLoss,
 lr_scheduler=lr_scheduler, early_stopping=early_stopping_cfg,
 ).with_experiment( # 实验配置
 exp_manager=exp_manager, hyperparams=args,
 model_name="GIKT", dataset_name=args.dataset,
 ).build # 验证 + 固化
```

``build`` 方法（``base_trainer.py`` ）负责验证 4 个配置块必须全部设置、创建 DataLoader、初始化回调链、设置随机种子、创建指标累积器。之后调用 ``run`` 启动训练。

### 训练循环核心

训练循环在 ``_run_training_loop``（``base_trainer.py`` ）中实现：

```
_run_training_loop
 ├─ model.to(device), loss.to(device) # 设备迁移
 ├─ callback_manager.on_train_begin # 训练开始回调
 ├─ for epoch in range(start_epoch, epochs):
 │ ├─ callback_manager.on_epoch_begin
 │ ├─ _process_epoch(epoch, is_train=True) # 训练阶段
 │ │ └─ for batch:
 │ │ _run_train_batch(batch_data):
 │ │ opt.zero_grad
 │ │ output = forward_pass(batch_data)
 │ │ loss = _compute_loss(output)
 │ │ loss.backward
 │ │ clip_grad_norm # 可选
 │ │ opt.step
 │ ├─ _process_epoch(epoch, is_train=False) # 验证阶段
 │ │ └─ for batch:
 │ │ @torch.inference_mode:
 │ │ output = forward_pass(batch_data)
 │ │ loss = _compute_loss(output)
 │ ├─ callback_manager.on_epoch_end # 触发早停/检查点
 │ ├─ lr_scheduler.step
 │ └─ if should_stop: break # 早停
 └─ callback_manager.on_train_end # 保存最终模型
```

训练批次（``_run_train_batch``，）和验证批次（``_run_eval_batch``，）的核心区别：训练批次执行 ``opt.zero_grad → backward → clip_grad → opt.step`` 完整的梯度更新；验证批次用 ``@torch.inference_mode`` 禁用梯度计算，只做前向传播和损失计算。

### forward_pass 约定

每个训练器子类必须实现 ``forward_pass(batch_data) → dict``（``base_trainer.py`` ，抽象方法）。返回值字典必须包含三个键：

```python
{
    "y_hat": torch.Tensor,  # 模型原始输出（logits）
    "y_label": torch.Tensor,  # 真实标签
    "y_predict": torch.Tensor,  # 二值预测（通常 y_hat >= 0）
}
```

可选扩展键：``"y_score"``（原始得分）、``"y_prob"``（sigmoid 概率）。

**Next-item 对齐约定**（``base_trainer.py`` ）：``y_hat_full[t]`` 预测 ``response[t+1]``。``_extract_valid_predictions`` 提取 ``y_hat_full[:,:-1]`` 与 ``response[:,1:]`` 的对齐，掩码同样取相邻位置的交集 ``mask[:,:-1] & mask[:,1:]``。部分模型输出 ``[B, S-1]``（比输入少一位），通过 ``_pad_to_full_sequence``（）在前端补零对齐。

以 GIKT 的 ``forward_pass`` 为例（``model/GIKT/GIKT_trainer.py`` ）：

```python
def forward_pass(self, batch_data):
    sequence = self._move_tensor_to_device(batch_data["sequence"])
    response = self._move_tensor_to_device(batch_data["response"])
    mask = self._move_tensor_to_device(batch_data["mask"])
    skills = self._move_tensor_to_device(batch_data["skills"])
    hist_neighbor_index = self._move_tensor_to_device(batch_data["hist_neighbor_index"])

    y_hat_full = self._pad_to_full_sequence(
        self.model(
            user_sequence=sequence,
            user_response=response,
            user_mask=mask,
            skills=skills,
            graph_data=self.graph_data,
            hist_neighbor_index=hist_neighbor_index,
        )
    )
    y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)
    return {
        "y_hat": y_hat,
        "y_label": y_label,
        "y_predict": self._generate_binary_predictions(y_hat, threshold=0.0),
        "y_score": y_hat,
        "y_prob": torch.sigmoid(y_hat),
    }
```

## 评估层

每个 epoch 结束后，训练循环自动在验证集上评估指标。``MetricsAccumulator``（``utils/training/metrics.py``）在验证阶段收集所有 batch 的 ``y_hat`` 和 ``y_label``，在 phase 结束时聚合并计算：

- **AUC**（ROC 曲线下面积）
- **ACC**（准确率）
- **RMSE**（均方根误差）

训练完全结束后，``_evaluate_on_test_set``（``base_trainer.py`` ）在测试集上执行相同流程。如果启用了早停且 ``--es_restore_best``，评估使用验证集上最佳模型的权重。

## 输出层

### 目录结构

训练结果保存在 ``runs/<type>/<run_id>/``，由 ``ExperimentManager``（``utils/experiment_manager.py``）创建。以正常训练为例：

```
runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
├── best_model.pth # 验证集最佳模型检查点
├── last_checkpoint.pth # 最后一个 epoch 的检查点
├── hyperparameters.json # 完整超参数快照
├── training.log # 文本日志
├── metrics.csv # 本地 CSV 指标记录
└── case_analysis/ # 案例分析输出（可选）
```

### 指标记录

UniKT 采用**双轨记录**：本地 CSV 始终记录（通过 ``LocalMetricLogger``，``utils/training/metric_logger.py``），SwanLab 云端记录可选（通过 ``SwanLabMetricLogger``），通过 ``--no_swanlab`` 关闭。训练期间每 epoch 记录一次聚合指标，若启用 ``--log_batch_metrics`` 则额外记录每个 batch 的损失。

### 回调系统输出

``CallbackManager``（``utils/training/callbacks.py``）编排以下回调的输出：

- **CheckpointCallback**：按监控指标保存最佳模型权重和每个 epoch 的检查点
- **EarlyStoppingCallback**：当验证指标连续 ``patience`` 轮无改进时触发停止
- **MemoryCleanupCallback**：每个 epoch 结束后清理 GPU 缓存
- **TestEvaluationCallback**：训练完成后自动在测试集上评估

## 端到端数据流总览

将以上各层串联，一条 ``python train.py -m GIKT -d assistments09 --fold 0`` 命令的完整数据流：

```
CLI 参数
 │
 ▼
train.py ──► DataParams / GeneralParams / EarlyStoppingParams / CompileParams
 .add_args(parser) → 通用参数解析
 ──► PARAM_CONFIGS.get("GIKT").add_args(parser) → 模型参数解析
 → Namespace(args) 统一参数对象
 │
 ▼
get_data_source("assistments09", args)
 └─► DATA_SOURCES.get("assistments09")
 └─► importlib.import_module("utils.data_process.assistments09")
 └─► @register_data_source("assistments09") 装饰器执行
 └─► Assistments09DataSource(args=args)
 └─► fetch_data → load_src_data → clean_raw_data
 → transform_data → save_data
 → split_skill_sequence_data / relation_data 就绪
 │
 ▼
GIKTTrainer.__init__(args, data_src, exp_manager)
 └─► GIKTModelData(data_src).prepare_data(args)
 └─► data_src.get_split_skill_sequence_data
 └─► build_sequence_data → 5个等长数组 [num_users, max_skill_seq_len]
 └─► split_kfold_data(arrs, fold=0) → (train, val, test) 元组
 └─► GIKT(args).to(device)
 └─► Adam optimizer, BCEWithLogitsLoss
 └─► .with_training.with_data.with_optimization.with_experiment.build
 └─► DataLoader(train/val/test, batch_size=32)
 └─► EarlyStopping(config), LRScheduler
 └─► CallbackManager(Checkpoint, EarlyStopping, MemoryCleanup, TestEval)
 └─► .run
 └─► for epoch in 1..100:
 ├─ _process_epoch(train) → _run_train_batch × N
 │ └─ opt.zero_grad → forward_pass → loss.backward → opt.step
 ├─ _process_epoch(val) → _run_eval_batch × N
 │ └─ @inference_mode: forward_pass → loss
 ├─ callback_manager.on_epoch_end → 早停判断 / 保存检查点
 └─ lr_scheduler.step
 └─► callback_manager.on_train_end
 └─► _evaluate_on_test_set → 测试集指标
 └─► best_model.pth / last_checkpoint.pth / metrics.csv
 │
 ▼
runs/normal/GIKT_assistments09_20240403-120000_fold0_bs128/
 最终的模型权重、超参数、日志、指标文件
```

## 多阶段训练的数据流

``MultiTrainer``（``utils/training/multi_trainer.py`` ）在单阶段训练之上叠加了**阶段切换**机制。每个阶段拥有独立的模型、优化器、损失函数、数据和早停配置，通过 ``StageComponents`` dataclass 封装。

```{mermaid}
flowchart LR
 A[run] --> B["构建 stages = build_stages"]
 B --> C["for stage in stages:"]
 C --> D["on_stage_begin(name)"]
 D --> E["_apply_stage(stage.build)"]
 E --> F["_run_training_loop"]
 F --> G["on_stage_complete(name, result)"]
 G --> C
 G --> H["_finish"]
```

每个阶段调用 ``_run_training_loop`` 复用了 ``BaseTrainer`` 的核心训练逻辑，阶段之间的数据传递通过 ``on_stage_begin`` / ``on_stage_complete`` 钩子实现。例如 ABKT 模型在两个阶段（Knowledge Mastery → Ability Modeling）之间传递知识状态矩阵。

多阶段训练器在构造时接收 ``model=None``（``multi_trainer.py`` ），因为模型在各阶段内部动态构建：

- ``build_stages`` 返回 ``list[StageConfig]``，每个 ``StageConfig`` 通过 ``build`` 回调产出该阶段的 ``StageComponents``
- 运行时通过 ``_apply_stage`` 切换 ``self.model``、``self.opt``、``self.loss``、``self.train_data``、``self.epochs``、``self.early_stopping`` 等属性
- ``forward_pass`` 通过 ``self._current_stage`` 区分阶段，路由到不同的前向逻辑

**注意：** ``BaseTrainer.build`` 必须先调 ``with_training`` / ``with_data`` / ``with_optimization`` / ``with_experiment`` 四个方法，否则 ``build`` 会抛 ``ValueError``。四个配置块分别对应 ``TrainingConfig``、``DataConfig``、``OptimizationConfig``、``ExperimentConfig`` dataclass（``utils/config/training_config.py``）。
