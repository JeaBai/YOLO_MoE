# Tasks

## Task 1: 注册 MoE Callback 到默认回调链
**Files:**
- Modify: `ultralytics/utils/callbacks/base.py`

- [ ] **SubTask 1.1: 注册 moe_callback**
  在 `ultralytics/utils/callbacks/base.py` 中：
  1. 添加导入：`from ultralytics.utils.callbacks.moe_callback import on_train_epoch_end as moe_on_train_epoch_end`
  2. 在 `default_callbacks` 字典中，将 `"on_train_epoch_end"` 的值从 `[on_train_epoch_end]` 改为 `[on_train_epoch_end, moe_on_train_epoch_end]`

- [ ] **SubTask 1.2: 验证回调注册**
  ```bash
  cd /workspace && python -c "
  from ultralytics.utils.callbacks.base import default_callbacks
  funcs = default_callbacks['on_train_epoch_end']
  names = [f.__name__ for f in funcs]
  print('Registered on_train_epoch_end callbacks:', names)
  assert 'on_train_epoch_end' in names, 'moe callback not registered'
  print('OK')
  "
  ```

- [ ] **SubTask 1.3: 提交**
  ```bash
  git add ultralytics/utils/callbacks/base.py && git commit -m "feat: register moe_callback for automatic expert diagnostics"
  ```

## Task 2: 运行 200 Epoch 训练 + 专家诊断
**Files:**
- Run: CLI 训练命令

- [ ] **SubTask 2.1: 确认数据集可用**
  ```bash
  cd /workspace && python -c "
  from ultralytics.utils.downloads import download
  download('https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip', dir='.')
  print('coco8 dataset ready')
  "
  ```

- [ ] **SubTask 2.2: 启动训练（后台运行）**
  ```bash
  cd /workspace && yolo train \
    data=coco8.yaml \
    model=ultralytics/cfg/models/moe26/YOLO_SparseDualMoE.yaml \
    epochs=200 \
    imgsz=640 \
    batch=4 \
    device=cpu \
    name=eamoe_coco8_200ep \
    exist_ok=True
  ```
  训练完成后检查 `runs/detect/eamoe_coco8_200ep/results.csv` 中的 MoE 指标列。

- [ ] **SubTask 2.3: 训练后诊断**
  训练完成后运行专家诊断：
  ```bash
  cd /workspace && python -c "
  from ultralytics.nn.modules.moe.analysis import diagnose_model
  import glob
  best_pt = glob.glob('runs/detect/eamoe_coco8_200ep/weights/best.pt')[0]
  diagnose_model(best_pt, dataset='coco8.yaml', batch_size=1)
  "
  ```
  预期输出：专家使用热图 `expert_usage_heatmap.png` 和柱状图 `expert_usage_bar.png`。

# Task Dependencies
- Task 2 depends on Task 1 (需要回调注册才能收集指标)