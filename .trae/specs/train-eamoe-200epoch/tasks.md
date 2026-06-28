# Tasks

## Task 1: 注册 MoE Callback 到默认回调链
**Files:**
- Modify: `ultralytics/utils/callbacks/base.py`

- [x] **SubTask 1.1: 注册 moe_callback**
  在 `ultralytics/utils/callbacks/base.py` 中：
  1. 添加导入：`from ultralytics.utils.callbacks.moe_callback import on_train_epoch_end as moe_on_train_epoch_end`
  2. 在 `default_callbacks` 字典中，将 `"on_train_epoch_end"` 的值从 `[on_train_epoch_end]` 改为 `[on_train_epoch_end, moe_on_train_epoch_end]`

- [x] **SubTask 1.2: 验证回调注册**
  已确认 2 个回调：`ultralytics.utils.callbacks.base.on_train_epoch_end` + `ultralytics.utils.callbacks.moe_callback.on_train_epoch_end`

- [x] **SubTask 1.3: 提交**
  `889dbf5` - `feat: register moe_callback for automatic expert diagnostics`

## Task 2: 运行 200 Epoch 训练 + 专家诊断
**Files:**
- Run: CLI 训练命令

- [x] **SubTask 2.1: 确认数据集可用**
  coco8.zip 已下载并解压。

- [x] **SubTask 2.2: 启动训练（200 epoch，top_k=4）**
  训练完成：200 epochs in 0.161 hours (CPU)。`moe_metrics.csv` 已生成，包含 200 行 MoE 指标数据。

- [x] **SubTask 2.3: 训练后诊断**
  MoE 指标分析完成：
  - P4 (model.6): avg_topk=1.48, avg_dead=1.8/4, e0 主导 (67%)
  - P5 (model.8): avg_topk=1.85, avg_dead=0.9/4, e0/e1/e3 较均衡
  - e2 专家在两个层均几乎不激活，需进一步调参

# Task Dependencies
- Task 2 depends on Task 1 (需要回调注册才能收集指标)