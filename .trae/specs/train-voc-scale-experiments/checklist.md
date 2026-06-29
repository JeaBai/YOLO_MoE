# Checklist

## Task 1: VOC 数据集准备
- [ ] VOC 2012 数据集自动下载成功（~16K 图，~2G）
- [ ] train/val 图片和标签路径正确，数量验证通过
- [ ] `/workspace/voc.yaml` 配置文件创建，20 个类别完整

## Task 2: 综合指标回调
- [ ] `comprehensive_metrics.py` 创建，包含 ComprehensiveMetricsCallback 类
- [ ] 每 epoch 记录 mAP50, mAP50-95, box_loss, cls_loss, dfl_loss
- [ ] 启动时记录 Params（固定值）
- [ ] 每 epoch 记录 inference_speed_ms（排除 warm-up，10 次取平均）
- [ ] MoE 模型正确检测 SparseDualMoE 模块并记录 II 类指标
- [ ] avg_descriptor_P4/P5 正确记录
- [ ] avg_topk_P4/P5 正确记录
- [ ] topk_distribution_P4/P5 正确记录（1-4 频次直方图）
- [ ] expert_usage_rate_P4/P5 (e0-e3) 正确记录
- [ ] load_balance_stddev_P4/P5 正确记录
- [ ] descriptor_var/energy 原始值正确记录
- [ ] GFLOPs_dynamic 估算正确记录
- [ ] Baseline A（无 MoE）II 类指标为 NaN
- [ ] Baseline B（隐式 MoE）记录可用 II 类指标
- [ ] 实验组（显式 MoE）完整记录 I+II+III 类指标
- [ ] `comprehensive_metrics.csv` 每 epoch 正确追加
- [ ] `base.py` 中正确注册回调

## Task 3: 稳定性监控回调
- [ ] `stability_monitor.py` 创建，包含 StabilityMonitorCallback 类
- [ ] 每 50 step 记录 gradient_l2_norm
- [ ] 每 50 step 记录 balance_loss, z_loss, entropy_loss
- [ ] 每 50 step 记录 learning_rate, weight_decay
- [ ] 每 50 step 记录 gpu_memory_mib（CPU 时为 0）
- [ ] `stability_metrics.csv` 每 50 step 正确追加
- [ ] `base.py` 中正确注册回调

## Task 4: 训练队列脚本
- [ ] `experiments/run_all.py` 创建
- [ ] 顺序启动 Baseline A → Baseline B → 实验组
- [ ] 支持 resume（检查 last.pt）
- [ ] 支持预训练权重加载
- [ ] 各实验使用独立 name，输出到不同子目录

## Task 5: 训练启动
- [ ] Baseline A 训练启动，指标 CSV 正确生成
- [ ] Baseline B 训练启动，指标 CSV 正确生成
- [ ] 实验组训练启动，所有三类指标 CSV 正确生成

## Task 6: 实时诊断
- [ ] `experiments/monitor.py` 创建
- [ ] 能读取各实验 comprehensive_metrics.csv
- [ ] 打印最新 epoch 关键指标对比表格