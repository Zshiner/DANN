import pandas as pd
import os
import pathlib
import pickle

def desc_dataset(data_name, x:pd.DataFrame, y:pd.DataFrame):
    print(data_name)
    print(f'样本数: {x.shape[0]+y.shape[0]}')
    print(f'症状数: {x.shape[1]}')
    print(f'诊断数: {y.shape[1]}')

    print(f'样本平均诊断数: {y.sum(axis=1).mean()}')
    print(f'样本平均特征数: {x.sum(axis=1).mean()}')

    row_sums = y.sum(axis=1)  # 计算每一行的和

    print(f"诊断数为1的样本比例: {(row_sums == 1).sum() / y.shape[0]}")
    print(f"诊断数为2的样本比例: {(row_sums == 2).sum() / y.shape[0]}")
    print(f"诊断数为3的样本比例: {(row_sums == 3).sum() / y.shape[0]}")


if __name__ == '__main__':

    #Lung
    with open(
        pathlib.Path(__file__).parent.parent / 'data' / 'Lung' / 'X.pkl',
        'rb'
    ) as f:

        X = pickle.load(f)

    with open(
        pathlib.Path(__file__).parent.parent / 'data' / 'Lung' / 'Y.pkl',
        'rb'
    ) as f:

        Y = pickle.load(f)

    desc_dataset('Lung', X, Y)


    #sub
    with open(
        pathlib.Path(__file__).parent.parent / 'data' / 'TCM-SUB' / 'X_15_repeat9.pkl',
        'rb'
    ) as f:

        X = pickle.load(f)

    with open(
        pathlib.Path(__file__).parent.parent / 'data' / 'TCM-SUB' / 'Y_15_repeat9.pkl',
        'rb'
    ) as f:

        Y = pickle.load(f)

    desc_dataset('Sub', X, Y)