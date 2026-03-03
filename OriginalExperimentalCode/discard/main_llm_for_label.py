from imaplib import IMAP4

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

# llm
# 每个llm维护一个结果表，后期从结果表里进行格式化和评价指标计算。相同样本直接取答案即可

import abc
import pandas
import pathlib
import os
from langchain_openai import ChatOpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from functools import partial
import tqdm
import hashlib
from utils.evaluator import MultiLabelEvaluator
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import tqdm


class ResultGenerator(abc.ABC):
    def __init__(self, data, dataset_name, process_num, type, llm_name):

        self.data = data
        self.root_save_dir = pathlib.Path(__file__).parent.absolute() / 'out'
        self.dataset_name = dataset_name
        self.process_num = process_num
        self.type = type
        self.llm_name = llm_name

        if self.type == 'multi_class':

            self.prompt = """
                你是一位资深的中医临床专家，请基于患者四诊信息，从指定的候选证型诊断中预测出最合适的一个。
                注意，你需要以json格式返回。

                示例：
                四诊信息：腻苔,舌胖,腹满,黄白相兼,滑,肥胖,疲乏无力,肢体困重,齿痕,舌色淡红,弦,水肿
                候选证型诊断：['3肥胖_肝郁气滞证', '3肥胖_脾肾阳虚证', '3肥胖_阴虚内热证', '3肥胖_脾虚湿阻证', '3肥胖_胃热湿阻证']
                输出：{'out': '3肥胖_肝郁气滞证'}

                请进行预测：
                四诊信息：symptoms
                候选证型诊断：syndrome
            """
        else:
            self.prompt = """
                你是一位资深的中医临床专家，请基于患者四诊信息，从指定的候选证型诊断中预测出最合适的一个或多个。
                注意，这是一个多标签分类诊断，诊断数量可能是一个或多个
                注意，你需要以json格式返回。

                示例：
                四诊信息：发热,口干,口黏腻,可闻及湿性啰音,咳嗽,咳痰,咽干,哮鸣,怕冷,慢性浅表性胃炎,易感冒,有气短重,有自汗中,气喘,气短,汗出,肺部呼吸音低,胸闷,鼻塞,鼻流涕
                候选证型诊断：['冷哮证', '外寒内热证', '外寒内饮证', '寒热错杂证', '心肺气虚证', '气虚血瘀证', '气血两虚证', '气阴两虚证', '湿热内蕴证', '湿热阻肺证', '热哮证', '痰浊内阻证', '痰浊壅肺证', '痰湿内阻证', '痰湿蕴肺证', '痰热内蕴证', '痰热郁肺证', '痰瘀内阻证', '痰瘀化火证', '痰瘀阻肺证', '痰蒙清窍证', '痰饮内停证', '瘀血阻肺证', '络气不和证', '肺气虚证', '肺肾两虚证', '肺肾气虚证', '肺肾气阴两虚证', '肺肾阳虚证', '肺肾阴虚证', '肺脾两虚证', '肺脾气虚证', '肺阴虚证', '脾肾两虚证', '脾肾气阴两虚证', '脾肾阳虚证', '脾胃湿热证', '脾虚湿困证', '血瘀证', '阳虚水泛证', '阴虚内热证', '阴虚津亏证', '阴虚肺热证', '阴阳两虚证', '风寒束肺证', '风热壅盛证', '风热袭肺证', '风痰哮证', '风邪犯肺证']
                输出：{'out': '痰热郁肺证'}

                请进行预测：
                四诊信息：symptoms
                候选证型诊断：syndrome
            """

        self.llm_info = [
            {
                'llm_name': 'zp',
                'client':
                    ChatOpenAI(
                        base_url='https://open.bigmodel.cn/api/paas/v4',
                        api_key='20b01302e183782e8c52f952ddbb570a.7VYqSvuekj98kRZu',
                        model='glm-4-plus',
                        streaming=False,
                        extra_body={
                            "response_format": {'type': 'json_object'},
                        },
                    )
            },

            {
                'llm_name': 'qwen',
                'client':
                    ChatOpenAI(
                        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        api_key='sk-f66d70a952fe41dbaf38057a37983609',
                        model='qwen-plus',
                        streaming=False,
                        extra_body={
                            "response_format": {'type': 'json_object'},
                        },
                    )
            },

            {
                'llm_name': 'deepseek',
                'client':
                    ChatOpenAI(
                        base_url='https://api.deepseek.com',
                        api_key='sk-257f25d28389433fbaa7923dadaf1677',
                        model='deepseek-reasoner',
                        streaming=False,
                        extra_body={
                            "response_format": {'type': 'json_object'},
                        },
                    )
            },

        ]

    def deal_data(self, train_x, train_y, test_x, test_y):
        test_label_list = test_y.apply(lambda row: row.index[row == 1].tolist(), axis=1).tolist()
        test_label_list_ = test_label_list.copy()
        train_label_list = train_y.apply(lambda row: row.index[row == 1].tolist(), axis=1).tolist()

        test_label_list = [i[0] for i in test_label_list]
        train_label_list = [i[0] for i in train_label_list]

        all_label_list = list(set(test_label_list + train_label_list))
        all_label_list = sorted(all_label_list)

        test_prompt_list = test_x.apply(lambda row: row.index[row == 1].tolist(), axis=1).tolist()
        test_prompt_list = [
            sorted(i) for i in test_prompt_list
        ]

        test_prompt_list = [
            self.prompt.replace('symptoms', ','.join(i)).replace('syndrome', str(all_label_list))
            for i in test_prompt_list
        ]

        return test_prompt_list, test_label_list, test_label_list_

    def get_result(self, obj_row=-1, llm_name=None, prompt_list=None, true_list=None, result=None):

        out_list = []

        llm_client = None
        for llm_info in self.llm_info:
            if llm_info['llm_name'] == llm_name:
                llm_client = llm_info['client']
                break

        for i in range(len(prompt_list)):

            # 原始解决方案，但效率慢
            if i != obj_row and obj_row != -1:
                continue

            prompt = prompt_list[i]

            hash_object = hashlib.md5(prompt.encode())
            md5_hash = hash_object.hexdigest()

            if md5_hash in result['md5_hash'].values:
                continue
            else:
                try:
                    # debug
                    # out = 'test12'
                    out = llm_client.invoke(prompt).content
                    out_list.append([self.dataset_name, md5_hash, prompt, true_list[i], out])
                except Exception:
                    pass

        return out_list

    def run(self):
        """

        :return:
        """




        # 确保所有测试样本都是跑一遍结果
        for llm_info in self.llm_info:
            llm_name = llm_info['llm_name']

            # 检查存储记录
            result_path = self.root_save_dir / f'{llm_name}.xlsx'
            result = pd.read_excel(result_path)

            for fold_data in self.data:
                train_x = fold_data[0]
                train_y = fold_data[1]
                test_x = fold_data[2]
                test_y = fold_data[3]

                #
                prompt_list, true_list, true_list_ = self.deal_data(train_x, train_y, test_x, test_y)

                # 更改label
                for i in tqdm.tqdm(range(len(prompt_list))):

                    if len(true_list_[i]) > 1:

                        hash_object = hashlib.md5(prompt_list[i].encode())
                        md5_hash = hash_object.hexdigest()
                        print(result.loc[result['md5_hash'] == md5_hash, 'label'].values[0], true_list_[i])
                        result.loc[result['md5_hash'] == md5_hash, 'label'] = str(true_list_[i])




            result.to_excel(result_path, index=False)




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
        return None

    print('start:', dataset_name)

    llm_rg = ResultGenerator(
        data=split_list,
        dataset_name=dataset_name,
        process_num=300,
        type=type,
        llm_name=llm_name,
    )

    # 调用接口获取结果
    llm_rg.run()

    # 获取评估结果
    # llm_rg.evaluate()

if __name__ == '__main__':

    # with Pool(processes=1) as pool:
    #     results = list(tqdm.tqdm(pool.imap(run, gd.get_all()), total=152))


    # debug
    for (dataset_name, split_list) in tqdm.tqdm(gd.get_all()):
        run((dataset_name, split_list))
