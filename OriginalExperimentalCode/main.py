from safetensors.torch import save_model
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
from config.config import Config
from scripts.deep import Comparator as DeepComparator
from scripts.mc import Comparator as MCComparator
from scripts.get_data import gd
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--only', type=str, default=None)
parser.add_argument('--mod', type=str, default=None)
parser.add_argument('--sub_num', type=str, default=None)
args = parser.parse_args()
only_model = args.only
sub_num = args.sub_num
mod = args.mod

import multiprocessing
# from sklearnex import patch_sklearn, config_context
# patch_sklearn()


# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
#



if __name__ == '__main__':
    for (dataset_name, split_list) in gd.get_all():



        # 根据数据集确定任务类型
        if 'sub' in dataset_name:
            type = 'multi_class'
        elif dataset_name == 'lung':
            type = "multi_label"
        else:
            raise UserWarning("任务类型未指定")

        # # 指定sub数据集
        if 'sub_1_repeat_0' not in dataset_name:
            continue

        print('start:', dataset_name)

        if mod == 'deep':
            deep_cp = DeepComparator(
                split_list, dataset_name, save_dir=f'./out/{dataset_name}', seed=Config.seed, default_args=False,
                device='cuda',
                type=type, top_k=2,
                only_model=only_model,
                process_num=1 if only_model == 'sann' else 1,
                save_model=False if type=='multi_class' else True
            )
            deep_cp.run()

        elif mod == 'mc':
            mc_cp = MCComparator(
                split_list, dataset_name, save_dir=f'./out/{dataset_name}', seed=Config.seed, default_args=False,
                device='cuda',
                type=type, top_k=2,
                only_model=None,
                process_num=multiprocessing.cpu_count(),
                save_model=True
            )
            mc_cp.run()
        elif mod=='all':
            deep_cp = DeepComparator(
                split_list, dataset_name, save_dir=f'./out/{dataset_name}', seed=Config.seed, default_args=False,
                device='cuda',
                type=type, top_k=2,
                only_model=only_model,
                process_num=multiprocessing.cpu_count() if only_model == 'sann' else 1,
                save_model=False if type == 'multi_class' else True
            )
            deep_cp.run()

            mc_cp = MCComparator(
                split_list, dataset_name, save_dir=f'./out/{dataset_name}', seed=Config.seed, default_args=False,
                device='cuda',
                type=type, top_k=2,
                only_model=None,
                process_num=multiprocessing.cpu_count(),
                save_model=False if type == 'multi_class' else True
            )
            mc_cp.run()

        else:
            print('debug')
            # mc_cp = MCComparator(
            #     split_list, dataset_name, save_dir=f'./out/{dataset_name}', seed=Config.seed, default_args=False,
            #     device='cuda',
            #     type=type, top_k=2,
            #     only_model=None,
            #
            #     process_num=multiprocessing.cpu_count(),
            # )
            # mc_cp.run()