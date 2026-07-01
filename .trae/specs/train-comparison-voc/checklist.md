# Checklist

- [x] `comprehensive_metrics.py` 创建并实现三类 CSV 记录逻辑
- [x] `metrics_core.csv` 包含 mAP50, mAP50-95, box_loss, cls_loss, dfl_loss 列
- [x] `metrics_stability.csv` 包含 learning_rate, weight_decay 列
- [x] `moe_metrics.csv` 逻辑与现有 moe_callback 一致，不重复注册
- [x] `comprehensive_metrics_on_train_epoch_end` 已注册到 `base.py` 的 `on_train_epoch_end`
- [x] `scripts/train_queue.py` 创建并实现三组队列执行
- [x] 队列脚本包含三组实验：显式 MoE → Baseline A → Baseline B
- [x] 每组实验参数正确（YAML 路径、数据集、epochs、imgsz、batch、device）
- [x] 失败实验不影响后续实验（try/except 隔离）
- [ ] 训练队列正常运行并完成三组实验（⏳ 进行中：VOC 数据集下载中）
- [ ] 实验组生成 `moe_metrics.csv`（含 MoE 特有指标）
- [ ] Baseline A 不生成 `moe_metrics.csv`（无 MoE 模块）
- [ ] Baseline B 的隐式 MoE 指标正常记录
- [ ] 所有 CSV 文件包含 200 行数据