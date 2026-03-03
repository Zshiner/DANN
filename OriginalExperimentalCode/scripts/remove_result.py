import pandas as pd
import pathlib
import os

def remove_result(dataset, model_name):
    base_dir = pathlib.Path(__file__).parent.parent / 'out'

    if dataset == 'lung':
        raise UserWarning('are you ok and u')
        data_dir_list = ['lung']
    elif 'sub' in dataset:
        data_dir_list = [f'{dataset}_repeat_{i}' for i in range(10)]

    remove_list = []

    for data_dir in data_dir_list:
        data_dir = base_dir / data_dir

        for root, dirs, files in os.walk(data_dir):
            for file in files:

                if model_name in file:
                    remove_list.append(data_dir / file)

    remove_list = set(list(remove_list))
    for remove_path in remove_list:
        try:
            os.remove(remove_path)
            print(f"文件 {remove_path} 已成功删除")
        except FileNotFoundError:
            print(f"文件 {remove_path} 不存在")
        except PermissionError:
            print(f"没有权限删除文件 {remove_path}")
        except Exception as e:
            print(f"删除文件时出错: {e}")



if __name__ == '__main__':

    # for i in range(1, 16):
    #     remove_result(f'sub_{i}', 'sann')

    remove_result(f'sub_{15}', 'sann')