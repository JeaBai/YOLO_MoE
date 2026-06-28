# Checklist

- [ ] `moe_callback` 已注册到 `default_callbacks["on_train_epoch_end"]`
- [ ] coco8 数据集已下载
- [ ] 训练 200 epoch 正常启动无报错
- [ ] 训练日志中出现 `moe/` 前缀的指标
- [ ] 训练完成后 `results.csv` 包含 MoE 指标列
- [ ] 专家使用热图 `expert_usage_heatmap.png` 生成
- [ ] 专家使用柱状图 `expert_usage_bar.png` 生成
- [ ] 专家分布无明显"DEAD"专家（使用率 > 0.1 × ideal_share）