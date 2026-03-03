import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from config.config import Config
import pathlib
import torch

class GetData():
    def __init__(self, cv, seed):
        self.datasets_info = [
            {
                'name': 'lung',
                'dir': pathlib.Path(__file__).parent.parent / 'data' / 'Lung',
                'type': 'multi_label'
            },

        ]

        # 随机10次sub 1-15共15个数据集集合
        temp = []
        for i in range(1, 16):
            for repeat_i in range(10):
                temp.append(
                    {
                        'name': f'sub_{i}_repeat_{repeat_i}',
                        'dir': pathlib.Path(__file__).parent.parent / 'data' / 'TCM-SUB',
                        'type': 'multi_class',
                        'suffix': f'{i}_repeat{repeat_i}'
                    }
                )

        self.datasets_info = temp+self.datasets_info

        self.seed = seed
        self.cv = cv

    def __read_dateset(self, dataset_name):

        for dataset_info in self.datasets_info:
            if dataset_info['name'] == dataset_name:
                if dataset_info.get('suffix', None):
                    X = pd.read_pickle(dataset_info['dir'] / f"X_{dataset_info['suffix']}.pkl")
                    Y = pd.read_pickle(dataset_info['dir'] / f"Y_{dataset_info['suffix']}.pkl")
                else:
                    X = pd.read_pickle(dataset_info['dir'] / f"X.pkl")
                    Y = pd.read_pickle(dataset_info['dir'] / f"Y.pkl")

                dataset_type = dataset_info['type']
                return X, Y, dataset_type
        else:
            raise UserWarning('The dataset does not exist.')

    def __split(self, X, Y, dataset_type, cv, seed):
        # 使用打乱后的索引来重新排序 x 和 y
        np.random.seed(seed)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X.iloc[indices, :].reset_index(drop=True)
        Y = Y.iloc[indices, :].reset_index(drop=True)


        #
        if dataset_type == 'multi_class':
            kf = KFold(n_splits=cv, shuffle=False)
        else:
            kf = KFold(n_splits=cv, shuffle=False)
        data = []
        for train_index, test_index in kf.split(X):
            train_x = X.iloc[train_index, :]
            train_y = Y.iloc[train_index, :]
            test_x = X.iloc[test_index, :]
            test_y = Y.iloc[test_index, :]
            data.append((train_x, train_y, test_x, test_y))

        return data

    def get_dataset(self, dataset_name):
        X, Y, dataset_type = self.__read_dateset(dataset_name)
        dataset = self.__split(X, Y, dataset_type, cv=self.cv, seed=self.seed)
        return dataset

    def get_all(self, tensor=False):
        datasets = []
        for dataset_info in self.datasets_info:
            dataset_name = dataset_info['name']
            dataset = self.get_dataset(dataset_name)

            if tensor:
                new_dataset = []
                for sub_dataset in dataset:
                    new_dataset.append([torch.Tensor(np.array(i)) for i in sub_dataset])


                datasets.append((dataset_name, new_dataset))

            else:
                datasets.append((dataset_name, dataset))

        return datasets

gd = GetData(cv=Config.cv, seed=Config.seed)

if __name__ == '__main__':
    pass