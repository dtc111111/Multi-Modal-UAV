#%%
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from dataset import UgClassificationDataset
from model.demo import DemoModel
from model.pointnet import pointnet_loss
torch.autograd.set_detect_anomaly(True)

#%% 加载数据集
dataset = UgClassificationDataset(
    modals=('Image', 'lidar_360', 'livox_avia', 'ground_truth', 'class'),  # ('class', 'Image', 'lidar_360', 'livox_avia', 'radar_enhance_pcl', 'ground_truth'),
    train=True,
    base_dir=Path('/home/dlut-ug/Anti_UAV_data/'),
    timeline_dir=Path('./out/')
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

#%% 初始化模型.损失函数和优化器
model = DemoModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% 训练模型
for epoch in range(3):
    for batch in dataloader:
        x_Image, x_lidar_360, x_livox_avia, y_ground_truth, y_class = [x.cuda() for x in batch]
        # Dataloader内容
        # x_Image: ([B, 3, 960, 2560]) BCHW
        # x_lidar_360: ([B, 19968, 3]) 19968个点坐标
        # x_livox_avia: ([B, 24000, 3]) 24000个点坐标
        # y_ground_truth: ([B, 3]) 无人机坐标
        # y_class: ([B, 3]) 无人机类型, OneHot编码

        # 正向传播
        optimizer.zero_grad()
        y_class_hat, y_gt_hat, loss_feat = model(x_Image, x_lidar_360, x_livox_avia)
        loss = cross_entropy(y_class_hat, y_class) + pointnet_loss(y_gt_hat, y_ground_truth, loss_feat)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印统计信息
        print("epoch = %d, loss = %f" % (epoch, loss))

#%% 保存模型
torch.save(model.state_dict(), 'model.pt')

