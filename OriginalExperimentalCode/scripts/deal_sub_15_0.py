# 将15个数据集的不同模型f1值进行对比画图
import pathlib
import pandas as pd
import os
import matplotlib.pyplot as plt

def read_results(metric_name):
    """

    :param base_dir:
    :param metric_name:
    :return:
    """
    base_dir = pathlib.Path(__file__).parent.parent.absolute() / 'out' / 'sub_15_repeat_0'
    save_dir = pathlib.Path(__file__).parent.parent.absolute() / 'result'
    result = {

    }

    for root, dirs, files in os.walk(base_dir):

        for file in files:
            if 'xlsx' in file:
                data = pd.read_excel(pathlib.Path(root) / file)

                metric_row = list(data.iloc[:, 0].values).index(metric_name)

                metric = data.iloc[metric_row, -1]
                model_name = pathlib.Path(file).stem
                dateset_name = 'sub_15_0'

                if model_name not in result.keys():
                    result[model_name] = {
                        dateset_name: metric
                    }
                else:
                    result[model_name][dateset_name] = metric

    data = pd.DataFrame(result).T

    data.to_excel(save_dir / f'sub_15_0_result_{metric_name}.xlsx')

if __name__ == '__main__':
    read_results(metric_name='f1')
    read_results(metric_name='recall')
    read_results(metric_name='precision')
