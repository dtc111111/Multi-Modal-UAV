#%%
import os
from pathlib import Path
from collections import OrderedDict
import pandas as pd

ROOT = Path('/home/dlut-ug/Anti_UAV_data/')
PREFIX = 'val'  # train val
MODALS = ('lidar_360', 'Image', 'livox_avia', 'ground_truth', 'radar_enhance_pcl', 'class')
BASE_MODAL = 'lidar_360'
MAX_SEQ = 16  # 训练集 102 测试集 16


#%% 函数
def get_seq(seq=1, modal='ground_truth', root=ROOT / PREFIX) -> OrderedDict[float, Path]:
    """
    获取模态的文件名和路径

    当前仅用于get_timestamp_list
    :param seq: Sequence id
    :param modal: One of ['class', 'ground_truth', 'Image', 'lidar_360', 'livox_avia', 'radar_enhance_pcl']
    :param root: Dataset base folder
    :return: An OrderedDict from timestamp to file path, the order is based on timestamp. For example {'170001.12345': Path('/path/to/modal/170001.12345.npy'), ...}
    """
    base_dir = root / f"seq{seq}" / modal
    pairs = [(float(x.stem), x) for x in base_dir.iterdir()]  # 时间戳float, 路径Path 点对
    pairs.sort(key=lambda x: x[0])
    return OrderedDict(pairs)


def get_timestamp_list(seq=1, modal=BASE_MODAL, root=ROOT / PREFIX) -> list[float]:
    return list(get_seq(seq, modal, root).keys())


def get_closet_timestamp(t, serial):
    """找到最近的一个点"""
    a = max([x for x in serial if x < t] or [-float('inf')])
    b = min([x for x in serial if x >= t] or [float('inf')])

    return a if t - a <= b - t else b


#%% 对齐
def align_seq(timestamps_dict: dict[str, list[float]]):
    """
    对齐一个timestamps_dict

    :param timestamps_dict: (用于对齐的)多模态文件数据结构
    TimestampDict = dict[ModalName, list[TimeStamp]], where
    ModalName=str 表示模态名,应与文件夹名一致;
    TimeStamp=float 表示时间戳
    """
    result = {'average': [], **{m: [] for m in MODALS}}
    for t in timestamps_dict[BASE_MODAL]:
        closet = {m: get_closet_timestamp(t, timestamps_dict[m]) for m in MODALS}
        average = sum(closet.values()) / len(closet.values())
        result['average'].append(average)
        for m in MODALS:
            result[m].append(closet[m])
    return result


#%%
def main():
    for seq_id in range(1, MAX_SEQ + 1):
        print(f"seq={seq_id} prefix={PREFIX}")

        timestamp_set = {m: get_timestamp_list(seq_id, m) for m in MODALS}
        result = align_seq(timestamp_set)

        result_df = pd.DataFrame(result)
        result_df['seq'] = pd.Series(seq_id, index=result_df.index)
        result_df.to_csv(f'out/{PREFIX}/{seq_id}.csv', sep="\t")


if __name__ == '__main__':
    os.makedirs(f'out/{PREFIX}', exist_ok=True)
    main()

    # #%%
    # PREFIX = 'raw_val_'
    # for seq_id in range(1,16+1):
    #     timestamp_set = {m: get_seq_timestamps(seq_id, m) for m in MODALS}
    #     result_df = pd.DataFrame.from_dict(timestamp_set,  orient='index')
    #     result_df.to_csv(f'out/{PREFIX}{seq_id}.csv', sep="\t")

    #
    # #%% 绘图
    # hv.extension('bokeh')
    # df = result_df.apply(lambda x: x-result_df['ground_truth'][0])
    # timeline_fig = hv.Overlay(
    #     [ hv.Curve((np.ones_like(np.linspace(-1,5,10))*x,np.linspace(-1,5,10))).opts(color='gray') for x in df['average']]
    # )
    # for i,m in enumerate(MODALS):
    #     timeline_fig *= hv.Scatter( (df[m], df[m]*0+i+1) )
    #
    # timeline_fig.opts(width=1920, height=400)
    # hv.save(timeline_fig, f'out/{PREFIX}{seq_id}.html')
