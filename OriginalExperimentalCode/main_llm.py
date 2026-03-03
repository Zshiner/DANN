from scripts.llm import ResultGenerator
from scripts.get_data import gd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--llm_name', type=str, default='zp')
args = parser.parse_args()
llm_name = args.llm_name
from multiprocessing import Pool
import tqdm
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
import warnings
# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run(data_tuple):
    dataset_name, split_list = data_tuple

    # 根据数据集确定任务类型
    if 'sub' in dataset_name:
        type = 'multi_class'
    elif dataset_name == 'lung':
        type = "multi_label"
    else:
        raise UserWarning("任务类型未指定")

    if 'lung' not in dataset_name:
        print('sub stop')
        return None

    print('start:', dataset_name)

    llm_rg = ResultGenerator(
        data=split_list,
        dataset_name=dataset_name,
        process_num=300,
        type=type,
        llm_name=llm_name,
        top_k=2,
    )

    # 调用接口获取结果
    # llm_rg.run()

    # 获取评估结果
    llm_rg.evaluate()

if __name__ == '__main__':

    with Pool(processes=66) as pool:
        results = list(tqdm.tqdm(pool.imap(run, gd.get_all()), total=152))


    # debug
    # for (dataset_name, split_list) in tqdm.tqdm(gd.get_all()):
    #     run((dataset_name, split_list))


