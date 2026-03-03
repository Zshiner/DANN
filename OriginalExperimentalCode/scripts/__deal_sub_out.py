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
    base_dir = pathlib.Path(__file__).parent.parent.absolute() / 'out'
    result = {

    }

    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            if 'sub_' not in dir : # 排除lung数据集的实验结果
                continue

            for _, _, files in os.walk(pathlib.Path(root) / dir):
                for file in files:
                    if 'xlsx' in file:
                        data = pd.read_excel(pathlib.Path(root) / dir / file)

                        metric_row = list(data.iloc[:, 0].values).index(metric_name)

                        metric = data.iloc[metric_row, -1]
                        model_name = pathlib.Path(file).stem
                        dateset_name = pathlib.Path(dir).stem

                        if model_name not in result.keys():
                            result[model_name] = {
                                dateset_name: metric
                            }
                        else:
                            result[model_name][dateset_name] = metric

    data = pd.DataFrame(result)

    data.to_excel(base_dir / f'sub_result_{metric_name}.xlsx')

def plot_results(file_name):
    """

    :param file_name:
    :return:
    """
    file_path = pathlib.Path(__file__).parent.parent.absolute() / 'out' / file_name
    save_path = file_path.parent / f'{file_name}.jpg'
    data = pd.read_excel(file_path).T

    # 绘制折线图
    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 遍历每个模型，绘制折线
    for model in data.index:
        if 'Unnamed' in model:
            continue
        plt.plot(data.columns.astype(int), data.loc[model], marker='o', label=model)

    plt.xlabel('DataSet Num', fontsize=12)  # 横坐标标签
    plt.ylabel('Model Metric', fontsize=12)  # 纵坐标标签
    plt.title('Model Performance vs DataSet Num', fontsize=14)  # 标题
    plt.xticks(data.columns.astype(int))  # 确保x轴刻度显示所有数据比例
    plt.ylim(0, 1)  # 设置y轴范围为0-1
    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线
    plt.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图参数
    # plt.show()
    plt.savefig(save_path, transparent=True)

if __name__ == '__main__':
    # read_results(metric_name='f1')
    plot_results('sub_result_f1.xlsx')