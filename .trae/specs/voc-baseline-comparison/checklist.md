# Checklist

- [ ] `ComprehensiveMetricsCallback` 类已创建，包含 `on_train_start`, `on_train_epoch_end`, `on_train_end`
- [ ] 回调在 `__init__.py` 中注册，可由 YOLO 训练流程自动调用
- [ ] 类别一指标（mAP50, mAP50-95, box/cls/dfl_loss, params）正常记录
- [ ] 类别二指标（descriptor, topk, expert_usage, dead_experts）仅 MoE 模型时记录
- [ ] 类别三指标（balance_loss, z_loss, entropy_loss, lr, gradient）正常记录
- [ ] 非 MoE 模型不会因类别二指标缺失而崩溃
- [ ] `voc.yaml` 配置正确，路径指向 `/datasets/VOCdevkit`
- [ ] `train_queue.py` 可顺序执行三组实验
- [ ] 实验失败不中断后续实验
- [ ] VOC 2007 数据集已下载并解压
- [ ] 实验组（显式 MoE）200 epoch 训练完成
- [ ] Baseline A（YOLO26）200 epoch 训练完成
- [ ] Baseline B（隐式 MoE）200 epoch 训练完成
- [ ] 三组实验的 `comprehensive_metrics.csv` 均已生成