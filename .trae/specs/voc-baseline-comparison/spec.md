# VOC 基线对比训练 Spec

## Why
显式分配 MoE (SparseDualMoE) 已在 coco8 上验证通过（normalization 修复后 0 dead experts）。现在需要：
1. 在中等规模数据集（VOC 2007）上跑 200 epoch，对比显式 MoE vs 标准 YOLO26 vs 隐式 MoE
2. 建立完整的指标记录系统，覆盖精度、效率、显式分配机制、训练稳定性四大类
3. 实现自动训练队列，三组实验顺序执行，无需人工干预

## What Changes
- **新增** 综合指标回调 `ComprehensiveMetricsCallback`：记录所有三类指标到独立 CSV
- **新增** 训练队列脚本 `train_queue.py`：顺序执行三组实验
- **新增** VOC 数据集配置 `voc.yaml`
- **保留** 现有 `moe_callback` 不变（向后兼容），新回调作为增强版

## Impact
- Affected specs: `add-explicit-allocation-moe`, `fix-descriptor-normalization`
- Affected code: 
  - `ultralytics/utils/callbacks/comprehensive.py`（新增）
  - `ultralytics/utils/callbacks/__init__.py`（新增注册）
  - `train_queue.py`（新增）
  - `voc.yaml`（新增）
- Not affected: `modules.py`, `descriptor.py`, `YOLO_SparseDualMoE.yaml`

---

## ADDED Requirements

### Requirement: 综合指标回调系统
系统 SHALL 提供 `ComprehensiveMetricsCallback`，在训练过程中按 epoch 记录以下三类指标到独立 CSV 文件：

**类别一：精度与效率核心指标**
- `mAP50`, `mAP50-95`（从验证集获取）
- `box_loss`, `cls_loss`, `dfl_loss`（从训练 loss 获取）
- `Params`（启动时记录一次，固定值）
- `Inference_speed_ms`（每 N epoch 测量一次）

**类别二：显式分配创新机制指标（仅 MoE 模型）**
- `avg_descriptor_P4`, `avg_descriptor_P5`
- `avg_topk_P4`, `avg_topk_P5`
- `expert_usage/e{N}_P4`, `expert_usage/e{N}_P5`
- `load_balance_stddev_P4`, `load_balance_stddev_P5`
- `dead_experts_P4`, `dead_experts_P5`

**类别三：训练稳定性监控指标**
- `balance_loss`, `z_loss`, `entropy_loss`（MoE 辅助损失）
- `learning_rate`
- `gradient_l2_norm`（周期性采样，非每 epoch）

#### Scenario: MoE 模型训练
- **WHEN** 训练 SparseDualMoE 模型
- **THEN** 每 epoch 记录全部三类指标到 `comprehensive_metrics.csv`
- **AND** 非 MoE 指标列留空，不报错

#### Scenario: Baseline 模型训练（无 MoE）
- **WHEN** 训练 YOLO26（无 MoE）或隐式 MoE
- **THEN** 类别一和类别三指标正常记录
- **AND** 类别二指标列留空（隐式 MoE 的 expert_usage 等通过现有接口获取）

### Requirement: 训练队列系统
系统 SHALL 提供 `train_queue.py` 脚本，按顺序执行三组实验：

1. **实验组**：`YOLO_SparseDualMoE.yaml`（显式分配 MoE）
2. **Baseline A**：`yolo26.yaml`（标准 YOLO26，无 MoE）
3. **Baseline B**：`yolo26_moe.yaml`（隐式 MoE）

#### Scenario: 队列顺序执行
- **WHEN** 运行 `python train_queue.py`
- **THEN** 实验 1 完成后自动开始实验 2，实验 2 完成后自动开始实验 3
- **AND** 任一实验失败不影响后续实验
- **AND** 每个实验的输出保存到独立目录 `runs/detect/voc_<name>/`

#### Scenario: 统一配置
- **WHEN** 三组实验启动
- **THEN** 均使用相同配置：`imgsz=320, batch=8, epochs=200, device=cpu, data=voc.yaml`
- **AND** 均注册 `ComprehensiveMetricsCallback`

### Requirement: VOC 数据集配置
系统 SHALL 提供 `voc.yaml` 数据集配置文件，指向 VOC 2007 trainval 数据集。

#### Scenario: VOC 数据集可用
- **WHEN** VOC 数据集已下载到 `/datasets/VOCdevkit/`
- **THEN** 训练正常启动，使用 VOC 2007 train+val 作为训练集，VOC 2007 test 作为验证集