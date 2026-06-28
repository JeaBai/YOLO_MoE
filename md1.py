# 参数量
from ultralytics import YOLO

# 加载模型配置文件（仅结构）
# model = YOLO('yolo26.yaml')  # 返回的是 YOLO 对象   D:\Users\Bai Jia\source\PycharmProjects\YOLOv26\ultralytics-main\ultralytics\cfg\models\moe26\YOLO_MoEBlock.yaml
model = YOLO('./ultralytics/cfg/models/moe26/YOLO_BalanceMoE-v0_4.yaml')

# model = YOLO('D:/Users/Bai Jia/source/PycharmProjects/YOLOv26/YOLO-Master-main/YOLO-Master-main/ultralytics/cfg/models/master/exp/yolo-master-v0_2.yaml')
# D:/Users/Bai Jia/source/PycharmProjects/YOLOv26/YOLO-Master-main/YOLO-Master-main/ultralytics/cfg/models/master/exp/yolo-master-v0_7.yaml

print(model.model)  # 打印 PyTorch 模型结构

# 或者直接获取 PyTorch 模型
pt_model = model.model
pt_model.info()



# from ultralytics.nn.tasks import DetectionModel
# import torch
#
# model = DetectionModel(cfg='./ultralytics/cfg/models/moe26/yolo26_moe.yaml')
# model.eval()  # 设为评估模式（可选）
# x = torch.randn(1, 3, 640, 640) / 255.0
# y = model(x)
# # shapes = []
# # for out in y:
# #     shapes.append(out.shape)
# # print(shapes)
