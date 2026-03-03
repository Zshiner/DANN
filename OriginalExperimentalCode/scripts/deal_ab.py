# 将15个数据集的不同模型f1值进行对比画图
import pathlib
import pandas as pd
import os
import matplotlib.pyplot as plt

def read_results(dataset, metric_name):
    """

    :param base_dir:
    :param metric_name:
    :return:
    """
    base_dir = pathlib.Path(__file__).parent.parent.absolute() / 'out' / 'ablation'
    save_dir = pathlib.Path(__file__).parent.parent.absolute() / 'result'
    result = {

    }

    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if dataset not in dir: # 排除非sub数据集的实验结果
                continue

            for _, _, files in os.walk(pathlib.Path(root) / dir):
                for file in files:
                    if 'xlsx' in file:
                        data = pd.read_excel(pathlib.Path(root) / dir / file)

                        metric_row = list(data.iloc[:, 0].values).index(metric_name)

                        metric = data.iloc[metric_row, -1]
                        model_name = pathlib.Path(file).stem
                        dateset_name = pathlib.Path(dir).stem.split('_repeat')[0]

                        if model_name not in result.keys():
                            result[model_name] = {
                                dateset_name: metric
                            }
                        else:
                            result[model_name][dateset_name] = metric

    data = pd.DataFrame(result).T

    data.to_excel(save_dir / f'ab_{dataset}_{metric_name}.xlsx')

if __name__ == '__main__':
    # read_results(dataset='lung', metric_name='f1')
    # read_results(dataset='lung', metric_name='recall')
    # read_results(dataset='lung', metric_name='precision')
    read_results(dataset='sub_15_repeat_0', metric_name='f1')