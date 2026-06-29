# VOC Scale Training + Comprehensive Metrics Spec

## Why
COCO 全量训练在当前无 GPU、磁盘仅 7.3G 的环境下不可行。VOC 2012（~16K 图，~2G）是最佳替代方案：
- 规模足以验证 MoE 显式/隐式差异
- 磁盘空间充裕
- CPU 训练可在合理时间内完成（预估 3-5 天/200 epoch）
- 同时跑 3 组实验（Baseline A=YOLO26, Baseline B=隐式 MoE, 实验组=显式 MoE）

## What Changes
- **新增** `voc.yaml` 数据集配置文件
- **新增** `ultralytics/utils/callbacks/comprehensive_metrics.py`：综合指标回调，每 epoch/每 N step 记录 20+ 指标到 CSV
- **新增** `ultralytics/utils/callbacks/stability_monitor.py`：梯度健康度、MoE 辅助损失、优化器状态监控
- **新增** `experiments/` 训练队列脚本：自动顺序/并行启动 3 组实验
- **修改** `base.py`：注册新回调
- **不动** 任何模型架构（已完成）

## Impact
- Affected specs: 训练流程、指标记录系统
- Affected code: `callbacks/`、`cfg/`、`experiments/`
- Not affected: `modules.py`、`descriptor.py`、router、experts

---

## ADDED Requirements

### Requirement: VOC 数据集配置
系统 SHALL 提供 `voc.yaml`，路径指向已下载的 VOC 2012 数据集。

#### Scenario: VOC 数据集可用
- **WHEN** 启动训练 `data='voc.yaml'`
- **THEN** 训练正常加载 train/val 图片和标签

### Requirement: 综合指标回调 (ComprehensiveMetricsCallback)
系统 SHALL 在 `ultralytics/utils/callbacks/comprehensive_metrics.py` 中实现回调，每 epoch 记录三类指标到 CSV：

**I. 精度与效率核心指标：**
- `mAP50`, `mAP50-95`（标准 COCO 指标）
- `box_loss`, `cls_loss`, `dfl_loss`
- `GFLOPs_dynamic`（每张图动态计算量，显式 MoE 可获取）
- `Params`（参数量，启动时记录）
- `inference_speed_ms`（单张推理延迟，排除 warm-up）

**II. 显式分配创新机制指标（仅 MoE 模型）：**
- `avg_descriptor_P4`, `avg_descriptor_P5`
- `avg_topk_P4`, `avg_topk_P5`
- `topk_distribution_P4`, `topk_distribution_P5`（Top-K 频次直方图）
- `expert_usage_rate_P4/e0..e3`, `expert_usage_rate_P5/e0..e3`
- `load_balance_stddev_P4`, `load_balance_stddev_P5`
- `descriptor_var_sample_P4`, `descriptor_energy_sample_P4`（底层原始值均值）

**III. 训练稳定性监控指标（每 N=50 step 记录一次）：**
- `gradient_l2_norm`（各层梯度 L2 范数均值）
- `balance_loss`, `z_loss`, `entropy_loss`（MoE 辅助损失）
- `learning_rate`, `weight_decay`
- `gpu_memory_mib`（显存占用，CPU 时为 0）

#### Scenario: Baseline A 训练（无 MoE）
- **WHEN** 训练标准 YOLO26
- **THEN** 仅记录 I 类指标，II 类指标为 NaN/空

#### Scenario: Baseline B 训练（隐式 MoE）
- **WHEN** 训练隐式 MoE（yolo26_moe.yaml）
- **THEN** 记录 I 类 + 可用的 II 类指标（隐式 MoE 无 descriptor，但可记录 expert_usage）

#### Scenario: 实验组训练（显式 MoE）
- **WHEN** 训练显式 MoE（YOLO_SparseDualMoE.yaml）
- **THEN** 完整记录 I + II + III 类全部指标

### Requirement: 训练队列脚本
系统 SHALL 在 `experiments/run_all.py` 中实现顺序训练队列：

1. 训练 Baseline A（YOLO26）：`epochs=200, imgsz=640, batch=16`
2. 训练 Baseline B（隐式 MoE）：`epochs=200, imgsz=640, batch=16`
3. 训练实验组（显式 MoE）：`epochs=200, imgsz=640, batch=16`

每组实验使用独立 `name`，输出到 `runs/detect/` 下不同子目录。

#### Scenario: 实验中断恢复
- **WHEN** 训练中断后重新运行脚本
- **THEN** 支持 `resume=True` 继续训练

### Requirement: 指标 CSV 结构
系统 SHALL 为每组实验生成独立的 `metrics.csv`，列名统一，无指标时填 `NaN`。

CSV 示例列：
```
epoch,step,mAP50,mAP50-95,box_loss,cls_loss,dfl_loss,GFLOPs_dynamic,Params,inference_speed_ms,
avg_descriptor_P4,avg_descriptor_P5,avg_topk_P4,avg_topk_P5,
expert_usage_P4_e0,expert_usage_P4_e1,expert_usage_P4_e2,expert_usage_P4_e3,
expert_usage_P5_e0,expert_usage_P5_e1,expert_usage_P5_e2,expert_usage_P5_e3,
load_balance_stddev_P4,load_balance_stddev_P5,
descriptor_var_P4,descriptor_energy_P4,descriptor_var_P5,descriptor_energy_P5,
gradient_l2_norm,balance_loss,z_loss,entropy_loss,lr,weight_decay,gpu_memory_mib
```

## MODIFIED Requirements

### Requirement: Callback 注册
`base.py` 中的 `default_callbacks` SHALL 新增 `comprehensive_metrics_callback` 和 `stability_monitor_callback`。

#### Scenario: 启动训练
- **WHEN** 任意模型训练启动
- **THEN** 综合指标回调和稳定性监控回调自动注册并运行