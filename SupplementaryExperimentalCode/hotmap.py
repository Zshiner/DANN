# -*- coding: utf-8 -*-

# 参数优化

from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings

from config.config import Config

from scripts.deep_for_opt import Comparator as DeepComparator
from scripts.get_data import gd
from model.sann import DANN, DANN_without_D, DANN_without_F,DANN_05,DANN_ave,DANN_100,DANN_z_score
import pickle
import pathlib

# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#
import torch
from multiprocessing import Pool
import multiprocessing
import tqdm

# 通过模型信息加载指定消融实验模型
def cur_model_param_grid(self):
    """
            用户自定义
            """
    model_param_grid = {}

    model_name = f'sannDANN'
    model_class = DANN
    param_grid = {
        'hidden_dim': [200],
        'drop_rate': [0.1],
        'device': [self.device],
        'lr': [0.1],
        'batch_size': [1024],
        'tor': [5],
        'ori': [True]
    }
    model_param_grid[model_name] = (model_class, param_grid)


    return model_param_grid

DeepComparator.get_model_param_grid = cur_model_param_grid



def run(data_tuple):
    """

    :param data_tuple:
    :return:
    """
    dataset_name, split_list = data_tuple

    # 根据数据集确定任务类型
    if 'sub' in dataset_name:
        type = 'multi_class'
    elif dataset_name == 'lung':
        type = "multi_label"
    else:
        raise UserWarning("任务类型未指定")

    # # 指定慢性病数据集
    if 'sub_15_repeat_0' not in dataset_name:
        return None

    print('start:', dataset_name)

    deep_cp = DeepComparator(
        split_list, dataset_name, save_dir=f'./out/hotmap/{dataset_name}', seed=Config.seed, default_args=False,
        device='cuda',
        type=type, top_k=1,
        process_num=1,
        save_model=True,

    )
    deep_cp.run()

if __name__ == '__main__':

    # ori
    # for (dataset_name, split_list) in gd.get_all():
    #     run((dataset_name, split_list))

    model = torch.load(
        open(
            pathlib.Path(__file__).parent.absolute() / 'out' / 'hotmap'/ 'sannDANN_0.pkl',
            'rb',
        ),
        weights_only=False,
        map_location=torch.device('cpu')
    )
    model.trained_weights.to_excel(pathlib.Path(__file__).parent.absolute() / 'out' / 'hotmap'/ 'dann.xlsx')
    print(1)


    # multi
    # with Pool(processes=50) as pool:
    #     results = list(tqdm.tqdm(pool.imap(run, gd.get_all()), total=1536))

