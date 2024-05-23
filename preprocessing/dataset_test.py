from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import UgClassificationDataset
from model.demo import DemoModel

#%% 加载数据集
dataset = UgClassificationDataset(
    modals=('Image', 'lidar_360', 'livox_avia', 'ground_truth', 'class'),  # ('class', 'Image', 'lidar_360', 'livox_avia', 'radar_enhance_pcl', 'ground_truth'),
    train=False,
    base_dir=Path('/home/dlut-ug/Anti_UAV_data/'),
    timeline_dir=Path('./out/'),
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

#%% 初始化模型
model = DemoModel().cuda().eval()


#%% 训练模型
n_correct   = 0
n_incorrect = 0
for batch in dataloader:
    x_Image, x_lidar_360, x_livox_avia, y_ground_truth, y_class = [x.float().cuda() for x in batch]

    # 正向传播
    y_hat = model(x_Image, x_lidar_360, x_livox_avia)

    if torch.argmax(y_class) == torch.argmax(y_hat):
        n_correct += 1
    else:
        n_incorrect += 1
    print(f"ACC: {n_correct}/{(n_correct+n_incorrect)} = {n_correct/(n_correct+n_incorrect)}")

#%% 保存模型
torch.save(model.state_dict(), 'model.pt')

