# 基于保存的五折最优模型，复现五折交叉验证现场。
# 用于案例分析，top_k等更换重新计算评价指标
from copyreg import pickle

from model.sann import SANN
from model.bert.Bert import TCMBert, ZYBert
from model.bert.BertCNN import BertCNN
import torch
import pathlib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from scripts.get_data import gd
from utils.tokenizer import totokenid_bert, padbatch2tokenid
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
import pickle

class Reproduction():
    def __init__(self, dataset_name, model_name, device):

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device=device

        self.base_dir = pathlib.Path(__file__).parent.parent / 'out'
        self.save_dir = pathlib.Path(__file__).parent.parent / 'result'

    def get_model(self):
        try:
            model = torch.load(
                self.base_dir / self.dataset_name / f'{self.model_name}_opt.pkl', map_location=torch.device('cpu'),
                weights_only=False

            )
        except Exception:
            with open((self.base_dir / self.dataset_name / f'{self.model_name}_opt.pkl').absolute(), 'rb') as f:
                model = pickle.load(f)

        return model

    def top_1(self):
        pass

    def get_result(self, model, dataloader):
        """

        :return:
        """
        model.eval()
        model = model.to(self.device)

        all_predict_y = None
        all_true_y = None
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                if isinstance(batch_x, torch.Tensor):
                    batch_x = batch_x.to(self.device)
                else:
                    batch_x = (i.to(self.device) for i in batch_x)
                predict_y = model.forward(batch_x)

                if all_predict_y is None:
                    all_predict_y = predict_y
                    all_true_y = batch_y
                else:
                    all_predict_y = torch.concat((all_predict_y, predict_y), dim=0)
                    all_true_y = torch.concat((all_true_y, batch_y), dim=0)

        return all_predict_y, all_true_y

    def get_case_result(self, x, y, predict_y):
        """

        :param x:
        :param y:
        :param predict_y:
        :return:
        """

        x_str =  [x.columns[x.loc[row] == 1].tolist() for row in x.index]
        x_str = [','.join(i) for i in x_str]

        y_str = [y.columns[y.loc[row] == 1].tolist() for row in y.index]

        if isinstance(predict_y, list):
            result = pd.DataFrame(
                data={
                    '症状': x_str,
                    '真实标签': y_str,
                    f'{self.model_name}': predict_y
                }
            )
            result.to_excel(
                self.base_dir / 'case' / f'case_{self.dataset_name}_{self.model_name}.xlsx'
            )
            return


        if self.dataset_name == 'lung':
            predict_y = torch.nn.Softmax(dim=-1)(predict_y)
            predict_y = pd.DataFrame(data=predict_y.detach().cpu(), columns=y.columns)
            predict_result = []

            # 遍历每一行
            for _, row in predict_y.iterrows():
                # 对行值进行排序，降序排列
                sorted_row = row.sort_values(ascending=False)
                # 获取前三个列名
                top_three_columns = sorted_row.head(3).index.tolist()
                # 添加到结果列表
                predict_result.append(top_three_columns)

        else:
            predict_y = torch.argmax(torch.nn.Sigmoid()(predict_y), dim=-1).detach().cpu().tolist()
            predict_result = [list(y.columns)[i] for i in predict_y]


        result = pd.DataFrame(
            data={
                '症状':x_str,
                '真实标签':y_str,
                f'{self.model_name}':predict_result
            }
        )

        result.to_excel(
            self.base_dir / 'case' /  f'case_{self.dataset_name}_{self.model_name}.xlsx'
        )

    def get_data(self):

        mark = False
        for (dataset_name, split_list) in gd.get_all():
            if self.dataset_name == dataset_name:
                mark = True
                break
            else:
                continue

        if mark:
            pass
        else:
            raise UserWarning('dataset error')

        # 或许需要适配不同模型，所以留下处理的接口。

        return split_list[0][2], split_list[0][3]

    def get_dataloader(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """


        class CustomDataset(Dataset):
            def __init__(self, x, y, device, bert=False, bert_path=None, max_len=None):
                super().__init__()
                self.max_len = max_len
                self.bert = bert
                if bert:
                    self.x = x.apply(lambda row: ','.join([col for col in x.columns if row[col] == 1]),
                                     axis=1).reset_index(drop=True)
                    self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
                else:
                    self.x = torch.Tensor(np.array(x))
                self.y = torch.Tensor(np.array(y))

            def __len__(self):
                return self.x.shape[0]

            def __getitem__(self, index):
                if self.bert:
                    cur_x = self.x[index]
                    input_ids, attention_mask, _ = totokenid_bert(self.tokenizer, [cur_x], max_len=self.max_len)
                    return (input_ids, attention_mask), self.y[index].to(torch.long)
                else:
                    return self.x[index], self.y[index].to(torch.long)

        data_loader = DataLoader(
            dataset=CustomDataset(
                x, y, self.device, bert=True if 'Bert' in model_name else False,
                bert_path=pathlib.Path(__file__).parent.absolute() / 'model' / 'bert' / 'zybert' if 'Bert' in model_name else None,
                max_len=512 if 'Bert' in model_name else None
            ),
            shuffle=False,
            batch_size=int(512),
            num_workers=0,
            collate_fn=padbatch2tokenid if 'Bert' in model_name else None,
        )

        return data_loader

    def get_result_llm(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        prompt = """
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

        x_prompt_list = x.apply(lambda row: row.index[row == 1].tolist(), axis=1).tolist()
        x_prompt_list = [
            sorted(i) for i in x_prompt_list
        ]

        x_prompt_list = [
            ','.join(i)
            for i in x_prompt_list
        ]


        data = pd.read_excel(pathlib.Path(__file__).parent.parent / 'out' / f'{self.model_name}.xlsx')
        data = data[data['dataset_name'] == self.dataset_name]

        predict_y = []
        for substring in x_prompt_list:
            # 查找A列中包含当前子字符串的行
            matched_rows = data[data['content'].str.contains(substring, regex=False)]

            if not matched_rows.empty:
                # 取出第一个匹配行的C列值
                predict_y.append(matched_rows.iloc[0]['llm_out'])
            else:
                # 如果没有匹配，可以添加None或其他你想要的默认值
                predict_y.append('error')

        return predict_y



    def run(self):

        x, y = self.get_data()

        if self.model_name in ['deepseek']:
            pass
        else:
            dataloader = self.get_dataloader(x, y)

            model = self.get_model()

        #
        if self.model_name in ['ZYBert', 'sann']:
            predict_y, _ = self.get_result(model, dataloader)

        elif self.model_name in ['deepseek']:
            predict_y = self.get_result_llm(x, y)

        else:
            predict_y = model.predict_proba(x)
            predict_y = np.column_stack([i[:, 1] for i in predict_y])
            predict_y = torch.tensor(predict_y)

        self.get_case_result(x, y, predict_y)







if __name__ == '__main__':

    # for model_name in ['deepseek', 'ZYBert', 'gbdt', 'sann']:
    #     for dataset in ['lung', 'sub_15_repeat_0']:
    #         rd = Reproduction(dataset, model_name, 'cuda')
    #         rd.run()

    for model_name in [ 'gbdt', 'sann']:
        for dataset in ['sub_15_repeat_0']:
            rd = Reproduction(dataset, model_name, 'cpu')
            rd.run()