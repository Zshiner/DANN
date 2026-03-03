# 参数优化

from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings

from config.config import Config

from scripts.deep_for_opt import Comparator as DeepComparator
from scripts.get_data import gd
from model.sann import SANN, SANN_without_D, SANN_without_F

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

    for hidden_dim in  [2200,2500,200, 300, 150, 250, 1000, 800, 500, 100, 80, 50, 20, 1200, 1400, 1500,1600,2000]:
        for lr in [2e-2, 2e-1, 1e-1, 3e-2, 2e-3, 2e-4, 1e-2, 2e-5]:
            for drop_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8]:
                for ori in [False, True]:
                        model_name = f'sann_{hidden_dim}_{lr}_{ori}_{drop_rate}'
                        model_class = SANN
                        param_grid = {
                            'hidden_dim': [hidden_dim],
                            'drop_rate': [drop_rate],
                            'device': [self.device],
                            # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
                            # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
                            'lr': [lr],
                            'batch_size': [1024],
                            'tor': [5],
                            'ori': [ori]
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

    # # 指定肺病数据集
    if 'sub_15_repeat_0' not in dataset_name:
        return None

    print('start:', dataset_name)

    deep_cp = DeepComparator(
        split_list, dataset_name, save_dir=f'./out/optimization/{dataset_name}', seed=Config.seed, default_args=False,
        device='cuda',
        type=type, top_k=2,
        process_num=10,
        save_model=True,

    )
    deep_cp.run()

if __name__ == '__main__':

    # ori
    for (dataset_name, split_list) in gd.get_all():
        run((dataset_name, split_list))


    # multi
    # with Pool(processes=50) as pool:
    #     results = list(tqdm.tqdm(pool.imap(run, gd.get_all()), total=1536))

