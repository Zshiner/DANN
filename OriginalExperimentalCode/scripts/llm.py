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
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import torch


class ResultGenerator(abc.ABC):
    def __init__(self, data, dataset_name, process_num, type, llm_name, top_k):

        self.data = data
        self.root_save_dir = pathlib.Path(__file__).parent.parent.absolute() / 'out'
        self.dataset_name = dataset_name
        self.process_num = process_num
        self.type = type
        self.llm_name = llm_name
        self.top_k = top_k

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

        all_label_list = list(set(test_label_list+train_label_list))
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

            #
            if llm_name == self.llm_name:
                pass
            else:
                continue

            for fold_data in self.data:
                train_x = fold_data[0]
                train_y = fold_data[1]
                test_x = fold_data[2]
                test_y = fold_data[3]

                #
                prompt_list, true_list = self.deal_data(train_x, train_y, test_x, test_y)

                # 检查存储记录
                result_path = self.root_save_dir / f'{llm_name}.xlsx'

                if os.path.exists(result_path):
                    result = pd.read_excel(result_path)
                else:
                    result = pd.DataFrame(
                        columns=['dataset_name', 'md5_hash','content', 'label', 'llm_out']
                    )
                    result.to_excel(result_path, index=False)

                # 调用api
                if self.process_num == 1:
                    out_list = self.get_result(
                        obj_row=-1,
                        llm_name=llm_name,
                        prompt_list=prompt_list,
                        true_list=true_list,
                        result=result
                    )
                else:
                    worker_partial = partial(
                        self.get_result,
                        llm_name=llm_name,
                        prompt_list=prompt_list,
                        true_list=true_list,
                        result=result
                    )
                    out_list = []
                    with ThreadPoolExecutor(max_workers=self.process_num) as executor:
                        bar = tqdm.tqdm(total=len(prompt_list))
                        for cur_out_list in executor.map(worker_partial, range(len(prompt_list))):
                            bar.update(len(cur_out_list))
                            out_list += cur_out_list
                    # with Pool(processes=self.process_num) as pool:
                    #     out_list = list(tqdm.tqdm(pool.imap(worker_partial, range(len(prompt_list))), total=len(prompt_list)))

                for i in out_list:
                    result.loc[len(result)] = i
                result.to_excel(result_path, index=False)

    def get_metrics(self, pre_list, true_list, type_):
        """

        :param pre_list:
        :param true_list:
        :param type:
        :return:
        """

        def multi_label_one_hot(nested_list, all_ids):
            # 创建一个映射：id -> 索引
            id_to_index = {id_: idx for idx, id_ in enumerate(all_ids)}

            # 初始化一个全零的 tensor
            num_samples = len(nested_list)
            num_classes = len(all_ids)
            one_hot_tensor = torch.zeros((num_samples, num_classes), dtype=torch.float32)

            # 填充 tensor
            for i, sublist in enumerate(nested_list):
                for id_ in sublist:
                    if id_ in id_to_index:
                        one_hot_tensor[i, id_to_index[id_]] = 1.0

            return one_hot_tensor

        if type_ == 'multi_label':

            pre_list = [i if isinstance(i, list) else [i] for i in pre_list]
            # 获取所有id
            all_id = []
            for i in pre_list + true_list:
                all_id.extend(i)

            all_id = list(set(all_id))
            all_id.pop(-1)

            #
            pre_list = multi_label_one_hot(pre_list, all_id)
            true_list = multi_label_one_hot(true_list, all_id)


            evaluator = MultiLabelEvaluator(top_k=self.top_k)

            return {
                "precision": evaluator.precision(true_list, pre_list),
                "recall": evaluator.recall(true_list, pre_list),
                "f1": evaluator.f1(true_list, pre_list),
            }
        else:
            return {
                "precision": precision_score(true_list, pre_list, average="macro"),
                "recall": recall_score(true_list, pre_list, average="macro"),
                "f1": f1_score(true_list, pre_list, average="macro"),
                "accuracy": accuracy_score(true_list, pre_list),

            }



    def evaluate(self):
        """

        :ret所有测试样本都是跑一遍结urn:
        """
        for llm_info in self.llm_info:
            llm_name = llm_info['llm_name']

            metrics = []

            save_path = pathlib.Path(__file__).parent.parent / 'out' / self.dataset_name / f'{llm_name}.xlsx'

            # if os.path.exists(save_path):
            #     return None

            for i, fold_data in enumerate(self.data):
                train_x = fold_data[0]
                train_y = fold_data[1]
                test_x = fold_data[2]
                test_y = fold_data[3]

                #
                prompt_list, true_list, true_list_ = self.deal_data(train_x, train_y, test_x, test_y)

                # 检查存储记录
                result_path = self.root_save_dir / f'{llm_name}.xlsx'

                if os.path.exists(result_path):
                    result = pd.read_excel(result_path)
                else:
                    raise UserWarning(f'{llm_name} result error......')

                #
                all_labels = None
                pre_list = []
                for prompt in prompt_list:

                    #
                    if not all_labels:
                        all_labels = eval(prompt.split('候选证型诊断：')[-1].strip())

                    hash_object = hashlib.md5(prompt.encode())
                    md5_hash = hash_object.hexdigest()

                    pre = result.query('md5_hash == @md5_hash')['llm_out'].values[0]
                    try:
                        pre = eval(pre)['out']
                    except Exception:
                        pre = ('error_out'
                               '')

                    # modify
                    if (',' in pre or '，' in pre) and not isinstance(pre, list):
                        pre = pre.replace('，', ',').split(',')



                    # onehot
                    if isinstance(pre, str):
                        # llm的输出不在标签里直接为错
                        try:
                            pre = all_labels.index(pre)
                        except Exception:
                            pre = -1
                    else:
                        pre = [all_labels.index(i) if i in all_labels else -1 for i in pre]

                    pre_list.append(pre)


                # 转onehot
                true_list = [all_labels.index(i) if isinstance(i,str) else [all_labels.index(ii) for ii in i] for i in true_list]
                true_list_ = [[all_labels.index(ii) if ii in all_labels else -1 for ii in i] for i in
                             true_list_]

                metrics.append(
                    self.get_metrics(pre_list, true_list if self.type=='multi_class' else true_list_, self.type)
                )

            metrics_data = pd.DataFrame(metrics)
            metrics_data.index= [1,2,3,4,5]
            metrics_data = metrics_data.T
            metrics_data['mean'] = metrics_data.mean(axis=1)
            metrics_data.to_excel(save_path)
            print(f'{self.dataset_name} {llm_name} save')








if __name__ == '__main__':
    pass