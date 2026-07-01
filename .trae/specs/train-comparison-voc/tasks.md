# Tasks

## Task 1: 创建全面指标 Callback 系统
**Files:**
- Create: `ultralytics/utils/callbacks/comprehensive_metrics.py`
- Create: `ultralytics/utils/callbacks/stability_monitor.py`
- (base.py already has imports registered)

- [x] **SubTask 1.1: 实现 `metrics_core.csv` 记录**
  在 `on_fit_epoch_end` 中记录：epoch, mAP50, mAP50-95, box_loss, cls_loss, dfl_loss, inference_speed

- [x] **SubTask 1.2: 实现 `metrics_stability.csv` 记录**
  在 `on_train_batch_end` 中记录：epoch, learning_rate, weight_decay

- [x] **SubTask 1.3: 整合现有 moe_callback**
  现有 `moe_on_train_epoch_end` 逻辑保留，全面指标 callback 作为独立模块

- [x] **SubTask 1.4: 注册到 default_callbacks**
  base.py 已注册 `comp_on_fit_epoch_end` 和 `stab_on_train_batch_end`

## Task 2: 创建训练队列脚本
**Files:**
- Create: `scripts/train_queue.py`

- [x] **SubTask 2.1: 实现队列执行逻辑**
  三组实验按顺序执行，失败隔离（try/except）

- [x] **SubTask 2.2: 配置三组实验参数**
  实验组、Baseline A、Baseline B 参数正确配置

- [x] **SubTask 2.3: 添加结果摘要输出**
  输出实验名称、best mAP50、总训练时间

## Task 3: 运行三组训练队列
**Files:**
- Run: `python scripts/train_queue.py`

- [x] **SubTask 3.1: 确认环境**
  PyTorch 2.12.1, torchvision, opencv, matplotlib, seaborn, pandas, pyyaml 已安装

- [x] **SubTask 3.2: 启动训练队列**
  训练队列已启动，VOC 2007 数据集下载中（~2.6GB）

- [ ] **SubTask 3.3: 验证输出**
  ⏳ 等待训练完成...

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 1, 2