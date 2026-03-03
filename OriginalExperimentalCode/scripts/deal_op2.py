import pathlib
import pandas as pd
import os
import matplotlib.pyplot as plt

def read_results(dataset, metric_name, op_args):
    """

    :param base_dir:
    :param metric_name:
    :return:
    """
    base_dir = pathlib.Path(__file__).parent.parent.absolute() / 'out' / 'optimization'
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
            break

    data = pd.DataFrame(result).T

    # 控制其他参数不变，迭代一个参数，画出折线图
    hidden_num = op_args[0]
    lr = op_args[1]
    ori = op_args[2]
    drop_rate = op_args[3]

    data_index = list(data.index)
    data_index = [i.split('_') for i in data_index]

    data['hidden_num'] = [float(i[1]) for i in data_index]
    data['lr'] = [float(i[2]) for i in data_index]
    data['drop_rate'] = [float(i[4]) for i in data_index]
    data['ori'] = [i[3] for i in data_index]

    # hidden_num
    for obj_arg in ['hidden_num', 'lr', 'drop_rate']:

        if obj_arg == 'hidden_num':
            obj_data = data[(data['lr'] == lr) & (data['drop_rate'] == drop_rate) & (data['ori'] == ori)]
        elif obj_arg == 'lr':
            obj_data = data[(data['hidden_num'] == hidden_num) & (data['drop_rate'] == drop_rate) & (data['ori'] == ori)]
        elif obj_arg == 'drop_rate':
            obj_data = data[(data['hidden_num'] == hidden_num) & (data['lr'] == lr) & (data['ori'] == ori)]

        obj_data = obj_data.iloc[obj_data[obj_arg].argsort()]

        # plt

        # lung
        if dataset=='lung' and obj_arg == 'drop_rate':
            y = obj_data.loc[:, dataset].values.tolist()
            x = obj_data.loc[:, obj_arg].values.tolist()

            plt.figure(figsize=(8, 5))  # 设置图形大小
            plt.ylim(0.48, 0.52)

            plt.plot(x, y, marker='o', linestyle='-', color='b')  # 折线图，带圆圈标记

            plt.xticks(x, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

            # 添加标题和标签
            plt.title(f'{dataset}_op_{obj_arg}')
            plt.xlabel(obj_arg)  # 使用第一列的列名作为x轴标签
            plt.ylabel('f1')  # 使用第二列的列名作为y轴标签

            # 显示网格
            plt.grid(True, linestyle='--', alpha=0.6)

            # 显示图形
            plt.savefig(save_dir / f'{dataset}_op_{obj_arg}.png')
            plt.show()

        elif dataset == 'lung' and obj_arg == 'hidden_num':
            y = obj_data.loc[:, dataset].values.tolist()
            x = [int(i) for i in obj_data.loc[:, obj_arg].values.tolist()]

            obj_x_index = [500, 1000, 1500, 2000, 2200 ,2500, 3000]
            y = [y[x.index(i)] for i in obj_x_index]
            x = obj_x_index

            plt.figure(figsize=(8, 5))  # 设置图形大小
            plt.plot(x, y, marker='o', linestyle='-', color='b')  # 折线图，带圆圈标记

            # 添加标题和标签
            plt.title(f'{dataset}_op_{obj_arg}')
            plt.xlabel(obj_arg)  # 使用第一列的列名作为x轴标签
            plt.ylabel('f1')  # 使用第二列的列名作为y轴标签

            # 显示网格
            plt.grid(True, linestyle='--', alpha=0.6)

            # 显示图形
            plt.savefig(save_dir / f'{dataset}_op_{obj_arg}.png')
            plt.show()

        elif  dataset == 'lung' and obj_arg == 'lr':
            y = obj_data.loc[:, dataset].values.tolist()
            x = obj_data.loc[:, obj_arg].values.tolist()
            plt.figure(figsize=(8, 5))  # 设置图形大小


            obj_x_index = [0.0002,0.002, 0.01, 0.02, 0.03]
            y = [y[x.index(i)] for i in obj_x_index]
            x = obj_x_index

            x = [1,2,3,4,5]
            plt.xticks(x, [0.0002,0.002, 0.01, 0.02, 0.03])


            plt.plot(x, y, marker='o', linestyle='-', color='b')  # 折线图，带圆圈标记
            # 添加标题和标签
            plt.title(f'{dataset}_op_{obj_arg}')
            plt.xlabel(obj_arg)  # 使用第一列的列名作为x轴标签
            plt.ylabel('f1')  # 使用第二列的列名作为y轴标签

            # 显示网格
            plt.grid(True, linestyle='--', alpha=0.6)

            # 显示图形
            plt.savefig(save_dir / f'{dataset}_op_{obj_arg}.png')
            plt.show()

        else:
            y = obj_data.loc[:, dataset].values.tolist()
            x = obj_data.loc[:, obj_arg].values.tolist()
            plt.figure(figsize=(8, 5))  # 设置图形大小
            plt.plot(x, y, marker='o', linestyle='-', color='b')  # 折线图，带圆圈标记


            # 添加标题和标签
            plt.title(f'{dataset}_op_{obj_arg}')
            plt.xlabel(obj_arg)  # 使用第一列的列名作为x轴标签
            plt.ylabel('f1')  # 使用第二列的列名作为y轴标签

            # 显示网格
            plt.grid(True, linestyle='--', alpha=0.6)

            # 显示图形
            plt.savefig(save_dir / f'{dataset}_op_{obj_arg}.png')
            plt.show()










if __name__ == '__main__':
    read_results(dataset='lung',metric_name='f1', op_args=[2200, 0.01, 'False', 0.6])