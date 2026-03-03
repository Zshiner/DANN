# 1. D\F ：直接取消
# 2. 矩阵分解： 跟一个矩阵对比
# 3. D\F\矩阵分解：全去掉
#
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings

from config.config import Config

from scripts.deep_for_ab import Comparator as DeepComparator
from scripts.get_data import gd
from model.sann import SANN, SANN_without_D, SANN_without_F, SANN_without_DF

# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#
import torch

dataset = 'sub_15'

# 通过模型信息加载指定消融实验模型
def cur_model_param_grid(self):
    """
            用户自定义
            """
    model_param_grid = {}

    model_name = 'sann_without_F_factorization'
    model_class = SANN_without_F
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [True]
    }
    model_param_grid[model_name] = (model_class, param_grid)

    model_name = 'sann_without_D_factorization'
    model_class = SANN_without_D
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [True]
    }
    model_param_grid[model_name] = (model_class, param_grid)


    model_name = 'sann_without_D'
    model_class = SANN_without_D
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [False]
    }
    model_param_grid[model_name] = (model_class, param_grid)


    model_name = 'sann_without_F'
    model_class = SANN_without_F
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [False]
    }
    model_param_grid[model_name] = (model_class, param_grid)


    model_name = 'sann_without_factorization'
    model_class = SANN
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [True]
    }
    model_param_grid[model_name] = (model_class, param_grid)


    model_name = 'sann_without_DF'
    model_class = SANN_without_DF
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [False]
    }
    model_param_grid[model_name] = (model_class, param_grid)

    model_name = 'sann_without_DF_factorization'
    model_class = SANN_without_DF
    param_grid = {
        'hidden_dim': [2200 if dataset == 'lung' else 150],
        'drop_rate': [0.6 if dataset == 'lung' else 0.3],
        'device': ['cuda' if torch.cuda.is_available() else 'cpu'],
        # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
        'lr': [1e-2 if dataset == 'lung' else 0.03],
        'batch_size': [1024],
        'tor': [5],
        'ori': [True]
    }
    model_param_grid[model_name] = (model_class, param_grid)

    return model_param_grid

DeepComparator.get_model_param_grid = cur_model_param_grid





if __name__ == '__main__':
    for (dataset_name, split_list) in gd.get_all():



        # 根据数据集确定任务类型
        if 'sub' in dataset_name:
            type = 'multi_class'
        elif dataset_name == 'lung':
            type = "multi_label"
        else:
            raise UserWarning("任务类型未指定")

        # # 指定数据集
        if dataset not in dataset_name:
            continue

        print('start:', dataset_name)

        deep_cp = DeepComparator(
            split_list, dataset_name, save_dir=f'./out/ablation/{dataset_name}', seed=Config.seed, default_args=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            type=type, top_k=2,
            process_num=1,
            save_model=True
        )

        deep_cp.run()

