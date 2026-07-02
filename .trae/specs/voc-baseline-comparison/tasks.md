# Tasks

## Task 1: 创建综合指标回调 `ComprehensiveMetricsCallback`
**Files:**
- Create: `ultralytics/utils/callbacks/comprehensive.py`
- Modify: `ultralytics/utils/callbacks/__init__.py`

- [ ] **SubTask 1.1: 实现回调类**
  - 实现 `on_train_start`：初始化 CSV 文件，写入表头
  - 实现 `on_train_epoch_end`：收集三类指标，写入一行
  - 实现 `on_train_end`：关闭文件，输出汇总
  - 类别二指标（MoE 专用）仅在模型有 `get_metrics()` 方法时收集
  - 类别三梯度指标每 10 epoch 采样一次
  
- [ ] **SubTask 1.2: 指标收集逻辑**
  - 类别一：从 `trainer.metrics`（val）和 `trainer.loss_items`（train）获取
  - 类别二：从 `model.get_metrics()` 获取（MoE 层）
  - 类别三：从 `trainer.loss_items` 和 `trainer.optimizer` 获取

- [ ] **SubTask 1.3: 注册到回调系统**
  在 `__init__.py` 中注册 `comprehensive_on_train_epoch_end` 等回调

## Task 2: 创建 VOC 数据集配置
**Files:**
- Create: `voc.yaml`

- [ ] **SubTask 2.1: 编写 voc.yaml**
  - path: `/datasets/VOCdevkit`
  - 使用 VOC 2007 train+val 作为 train，VOC 2007 test 作为 val
  - 20 个 VOC 类别

## Task 3: 创建训练队列脚本
**Files:**
- Create: `train_queue.py`

- [ ] **SubTask 3.1: 实现队列脚本**
  - 定义三组实验配置（名称、模型 YAML、输出目录）
  - 顺序执行，每组实验调用 `YOLO().train()`
  - 失败不中断，记录错误日志
  - 统一参数：`imgsz=320, batch=8, epochs=200, device=cpu, data=voc.yaml`

- [ ] **SubTask 3.2: 集成 ComprehensiveMetricsCallback**
  每组建模前注册回调，输出到 `runs/detect/voc_<name>/comprehensive_metrics.csv`

## Task 4: 下载 VOC 数据集并启动训练队列
**Files:**
- Run: 数据下载 + 训练命令

- [ ] **SubTask 4.1: 下载 VOC 2007**
  下载 `VOCtrainval_06-Nov-2007.tar` 并解压到 `/datasets/VOCdevkit/`

- [ ] **SubTask 4.2: 启动训练队列**
  运行 `python train_queue.py`，监控三组实验顺序完成

- [ ] **SubTask 4.3: 验证指标输出**
  - 确认三组实验的 `comprehensive_metrics.csv` 均生成
  - 实验组（显式 MoE）包含完整三类指标
  - Baseline A/B 包含类别一和部分类别三指标

# Task Dependencies
- Task 1, Task 2 可并行
- Task 3 depends on Task 1
- Task 4 depends on Task 1, 2, 3