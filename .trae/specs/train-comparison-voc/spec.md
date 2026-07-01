# 三组对比训练 + 全面指标系统 Spec

## Why
已完成显式分配 MoE 的归一化修复，需要与 Baseline 对比验证创新有效性。同时需要建立全面的指标记录系统，追踪精度、效率、显式分配机制和训练稳定性四大类指标。

## What Changes
- **新增** 全面指标 Callback 系统：每 epoch 记录三类指标到独立 CSV
- **新增** 训练队列脚本：按顺序自动执行三组实验（显式 MoE → Baseline A → Baseline B）
- **修改** 现有 moe_callback 整合到新指标系统
- 数据集：VOC 2007，320px, batch=8, CPU, 200 epochs

## Impact
- Affected specs: `fix-descriptor-normalization`（指标系统是后续扩展）
- Affected code: 
  - NEW: `ultralytics/utils/callbacks/comprehensive_metrics.py`
  - MODIFY: `ultralytics/utils/callbacks/moe_callback.py`
  - MODIFY: `ultralytics/utils/callbacks/base.py`
  - NEW: `scripts/train_queue.py`

---

## ADDED Requirements

### Requirement: 全面指标记录系统
系统 SHALL 在每个 epoch 结束时记录以下三类指标到独立 CSV 文件：

**一、精度与效率核心指标（每个实验独立 `metrics_core.csv`）**
- mAP50, mAP50-95（来自验证）
- box_loss, cls_loss, dfl_loss（来自训练）
- 推理速度（ms/img，排除 warmup）

**二、显式分配创新机制指标（仅 MoE 实验，`moe_metrics.csv`）**
- avg_descriptor, avg_topk（P4/P5）
- 专家使用率（e0-e3）、dead_experts
- balance_loss, z_loss, entropy_loss

**三、训练稳定性监控指标（所有实验，`metrics_stability.csv`）**
- learning_rate, weight_decay
- GPU_memory_usage（如有 GPU）

#### Scenario: 显式 MoE 实验
- **WHEN** 训练显式 SparseDualMoE
- **THEN** 同时生成 `metrics_core.csv`、`moe_metrics.csv`、`metrics_stability.csv`

#### Scenario: Baseline A/B 实验
- **WHEN** 训练 Baseline A（无 MoE）或 Baseline B（隐式 MoE）
- **THEN** 生成 `metrics_core.csv` 和 `metrics_stability.csv`；Baseline B 的隐式 MoE 指标从现有 callback 获取

### Requirement: 训练队列自动化
系统 SHALL 提供训练队列脚本，按顺序执行三组实验，无需人工干预：
1. 实验组：显式 SparseDualMoE
2. Baseline A：标准 YOLO26（无 MoE）
3. Baseline B：隐式 MoE（yolo26_moe）

每组实验完成后自动清理显存、记录结果摘要。

#### Scenario: 队列执行
- **WHEN** 运行 `python scripts/train_queue.py`
- **THEN** 三组实验按顺序执行，每组 200 epoch
- **AND** 每组结果保存在独立目录 `runs/detect/eamoe_voc`、`runs/detect/baseline_a_voc`、`runs/detect/baseline_b_voc`
- **AND** 任一实验失败不影响后续实验

### Requirement: VOC 2007 数据集
训练 SHALL 使用 VOC 2007 数据集，默认配置：imgsz=320, batch=8, device=cpu, epochs=200, workers=0。