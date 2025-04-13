# coding: UTF-8
# author: songyz2019

import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import unittest
import skimage

def _load_timeline(timeline_dir :Path) -> pd.DataFrame:
    result = None
    for file_path in sorted(timeline_dir.iterdir(), key=lambda x: int(x.stem)):
        if result is None:
            result = pd.read_csv(file_path, delimiter='\t', dtype='str')
        else:
            result = pd.concat((result, pd.read_csv(file_path, delimiter='\t', dtype='str') ))
    return result.reset_index()


class UgClassificationDataset(Dataset):

    def __init__(self, modals=('Image', 'lidar_360', 'livox_avia', 'ground_truth'), train=True, base_dir=Path('/home/dlut-ug/Anti_UAV_data/'), timeline_dir=Path('./out/'), n_point=19000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_dir = base_dir / ('train' if train else 'val')
        self.modals = modals
        self.timeline = _load_timeline(timeline_dir / ('train' if train else 'val'))
        self.n_point = n_point


    def __getitem__(self, index: int):
        def _load(modal):
            # Dirty Quick Fix
            def try_fix_path(file_path: Path):
                times_left = 5
                while not file_path.exists() and times_left > 0:
                    times_left -= 1
                    file_path = file_path.parent / (file_path.stem + '0' + file_path.suffix)
                assert file_path.exists(), file_path
                return file_path

            timestamp = self.timeline[modal][index]
            seq_folder = f"seq{self.timeline['seq'][index]}"

            result = None
            if modal == 'Image':
                file_path = self.base_dir / seq_folder / modal / (timestamp + '.png')
                file_path = try_fix_path(file_path)
                result = skimage.io.imread(file_path).transpose(2,0,1)
            else:
                file_path = self.base_dir / seq_folder / modal / (timestamp + '.npy')
                file_path = try_fix_path(file_path)


                if modal == 'lidar_360':
                    result = np.load(file_path)[:self.n_point]
                elif modal == 'class':
                    result = np.eye(3)[np.load(file_path)[0]-1]
                else:
                    result = np.load(file_path)

            return result.astype(np.float32)


        return [ _load(m) for m in self.modals ]

    def __len__(self):
        return len(self.timeline)

    def _preprocess(self):
        pass



class _UgDataloaderTestCase(unittest.TestCase):
    def setUp(self):
        self.ug_dataset = UgClassificationDataset(
            modals=('Image', 'lidar_360', 'livox_avia', 'ground_truth', 'class'), # ('class', 'Image', 'lidar_360', 'livox_avia', 'radar_enhance_pcl', 'ground_truth'),
            train=True,
            base_dir=Path('/home/dlut-ug/Anti_UAV_data/'),
            timeline_dir=Path('./out/'),
        )
        self.ug_dataloader = DataLoader(self.ug_dataset, shuffle=False, batch_size=1)

    def test_basic(self):
        self.assertEquals(len(self.ug_dataset), 19711)

    def test_dataset(self):
        x_Image, x_lidar_360, x_livox_avia, y_ground_truth = self.ug_dataset[0]
        print(x_Image)

    def test_loader(self):
        for x_Image, x_lidar_360, x_livox_avia, y_ground_truth in self.ug_dataloader:
            print(x_Image, x_lidar_360, x_livox_avia, y_ground_truth)
            break

