# Tasks

## Task 1: VOC 数据集准备
**Files:**
- Create: `/workspace/voc.yaml`

- [x] **SubTask 1.1: 自动下载 VOC 2012**
  运行 ultralytics 内置下载：`YOLO('yolov8n.pt').train(data='VOC.yaml', epochs=1)` 或直接用 shell 下载

- [x] **SubTask 1.2: 验证数据集完整性**
  确认 train/val 图片和标签数量正确，路径有效

- [x] **SubTask 1.3: 创建 voc.yaml**
  配置 train/val 路径，20 个类别名称

## Task 2: 综合指标回调 (ComprehensiveMetricsCallback)
**Files:**
- Create: `ultralytics/utils/callbacks/comprehensive_metrics.py`
- Modify: `ultralytics/utils/callbacks/base.py`

- [x] **SubTask 2.1: 实现 epoch 级指标记录**
  - `on_fit_epoch_end`: 记录 mAP50, mAP50-95, box_loss, cls_loss, dfl_loss
  - `on_train_epoch_start`: 记录 Params（固定值，只记一次）
  - `on_val_end`: 记录 inference_speed_ms（排除 warm-up，运行 10 次取平均）

- [x] **SubTask 2.2: 实现 MoE 专项指标记录**
  在 `on_fit_epoch_end` 中检测模型是否包含 SparseDualMoE：
  - 遍历 `model.model` 找到所有 `SparseDualMoE` 模块
  - 调用 `get_metrics()` 获取 expert_usage、avg_descriptor、avg_topk
  - 记录 topk_distribution（1-4 的频次直方图）
  - 记录 load_balance_stddev（expert_usage 的标准差）
  - 记录 descriptor_var/energy 原始值（通过 hook 或新增接口获取）

- [x] **SubTask 2.3: 实现 GFLOPs_dynamic 估算**
  显式 MoE 的 GFLOPs = 基础 GFLOPs + (avg_topk / K_max) × 专家额外 GFLOPs
  记录到 CSV

- [x] **SubTask 2.4: 写入统一 CSV**
  每 epoch 追加一行到 `comprehensive_metrics.csv`

- [x] **SubTask 2.5: 注册回调**
  在 `base.py` 的 `default_callbacks` 中注册

## Task 3: 稳定性监控回调 (StabilityMonitorCallback)
**Files:**
- Create: `ultralytics/utils/callbacks/stability_monitor.py`
- Modify: `ultralytics/utils/callbacks/base.py`

- [x] **SubTask 3.1: 实现梯度健康度监控**
  `on_train_batch_end`: 每 50 step 计算所有可训练参数的梯度 L2 范数均值

- [x] **SubTask 3.2: 实现 MoE 辅助损失监控**
  `on_train_batch_end`: 每 50 step 记录 balance_loss、z_loss、entropy_loss
  从 `trainer.loss_items` 中提取或从 MoE 模块获取

- [x] **SubTask 3.3: 实现优化器状态监控**
  `on_train_batch_end`: 每 50 step 记录当前 lr、weight_decay

- [x] **SubTask 3.4: 实现显存监控**
  `on_train_batch_end`: 每 50 step 记录 GPU 显存（CPU 时为 0）

- [x] **SubTask 3.5: 写入 CSV**
  每 50 step 追加一行到 `stability_metrics.csv`

- [x] **SubTask 3.6: 注册回调**
  在 `base.py` 中注册

## Task 4: 训练队列脚本
**Files:**
- Create: `experiments/run_all.py`

- [x] **SubTask 4.1: 实现顺序训练队列**
  按顺序启动：
  1. Baseline A: `YOLO('yolo26.yaml').train(data='voc.yaml', epochs=200, imgsz=640, batch=16, name='baseline_a_yolo26')`
  2. Baseline B: `YOLO('yolo26_moe.yaml').train(data='voc.yaml', epochs=200, imgsz=640, batch=16, name='baseline_b_implicit_moe')`
  3. 实验组: `YOLO('YOLO_SparseDualMoE.yaml').train(data='voc.yaml', epochs=200, imgsz=640, batch=16, name='experiment_explicit_moe')`

- [x] **SubTask 4.2: 支持中断恢复**
  检查 last.pt 存在则 `resume=True`

- [x] **SubTask 4.3: 预训练权重加载**
  Baseline A/B 使用 COCO 预训练权重（如可用），实验组使用显式 MoE 的 best.pt（如存在）

## Task 5: 训练启动与监控
**Files:**
- Run: `python experiments/run_all.py`

- [x] **SubTask 5.1: 启动 Baseline A**
  运行并验证指标 CSV 正确生成

- [ ] **SubTask 5.2: 启动 Baseline B**
  运行并验证指标 CSV 正确生成

- [ ] **SubTask 5.3: 启动实验组**
  运行并验证所有三类指标正确生成

## Task 6: 实时诊断脚本
**Files:**
- Create: `experiments/monitor.py`

- [x] **SubTask 6.1: 实现实时监控**
  读取各实验的 `comprehensive_metrics.csv`，打印最新 epoch 的关键指标

- [x] **SubTask 6.2: 实现对比表格**
  每 epoch 输出三组的 mAP50、mAP50-95、avg_topk 对比

# Task Dependencies
- Task 1 is independent
- Task 2 and Task 3 are independent
- Task 4 depends on Task 1, 2, 3
- Task 5 depends on Task 4
- Task 6 depends on Task 5