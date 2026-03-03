# 基于参数优化的多分类对比实验框架。
import os.path
import abc
from model.sann import SANN
from model.bert.Bert import ZYBert, TCMBert
from model.bert.BertCNN import BertCNN
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score, roc_auc_score
import numpy as np
import pandas as pd
import pathlib
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import torch
from torch.optim import AdamW
import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from multiprocessing import Pool
from utils.evaluator import MultiLabelEvaluator
from utils.tokenizer import totokenid_bert, padbatch2tokenid
from transformers import AutoTokenizer
import pathlib
import multiprocessing

# 忽略 UndefinedMetricWarning 警告
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def get_para_example(parameters:dict):
    """
    从参数字典中组合出所有情况，以列表返回
    :param parameters:
    :return:
    """
    example_num = 1
    for value in parameters.values():
        example_num *= len(value)

    columns = list(parameters.keys())

    data = pd.DataFrame(np.zeros(shape=(example_num, len(columns))), columns=columns)

    split_num_ed = 1

    for key in parameters.keys():
        key_cor_para_num = len(parameters[key])

        repeat_num = example_num // (key_cor_para_num * split_num_ed)
        split_num_ed *= key_cor_para_num

        value_list = []
        for value in parameters[key]:
            temp_value_list = [value for i in range(repeat_num)]
            value_list.extend(temp_value_list)

        value_list *= example_num // len(value_list)

        data[key] = value_list

    return [{para_name: data.iloc[row, list(data.columns).index(para_name)] for para_name in list(data.columns)} for row in range(data.shape[0])]

class CustomDataset(Dataset):
    def __init__(self, x, y, device, bert=False, bert_path=None, max_len=None):
        super().__init__()
        self.max_len = max_len
        self.bert = bert
        if bert:
            self.x = x.apply(lambda row: ','.join([col for col in x.columns if row[col] == 1]), axis=1).reset_index(drop=True)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        else:
            self.x = torch.Tensor(np.array(x))
        self.y = torch.Tensor(np.array(y))
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.bert:
            cur_x = self.x[index]
            input_ids, attention_mask,_ = totokenid_bert(self.tokenizer, [cur_x], max_len=self.max_len)
            return (input_ids, attention_mask), self.y[index].to(torch.long)
        else:
            return self.x[index], self.y[index].to(torch.long)

class Comparator(abc.ABC):
    def __init__(
            self,
            data,
            dataset_name,
            save_dir,
            default_args,
            metric_name='f1',
            seed=0,
            device='cuda',
            verbose=0,
            process_num=1,
            type='multi_class',   # multi_class multi_label
            top_k=None,
            only_model=None,
            save_model=True,
    ):
        if type == 'multi_label':
            if top_k is None:
                raise ValueError('top_k must be provided with type multi_label')

        self.save_model=save_model
        self.top_k = top_k
        self.type = type
        self.seed = seed
        np.random.seed(self.seed)
        self.device = device
        self.verbose = verbose
        self.save_dir = pathlib.Path(save_dir).absolute()
        self.metric_name = metric_name

        self.process_num = process_num

        self.default_args = default_args

        self.data = data
        self.dataset_name = dataset_name
        self.only_model = only_model


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        #
        self.metrics = ['f1', 'accuracy', 'recall', 'precision']
        self.epoch = 3000
        self.tor = 5


        # 在最后获取参数，以防有些参数会根据输入的x和y发生变化
        self.model_param_grid = self.get_model_param_grid()

    def get_model_param_grid(self):
        """
        用户自定义
        """
        model_param_grid = {}

        # sann
        model_name = 'sann'
        model_class = SANN
        param_grid = {
                'hidden_dim': [200, 300, 150, 250, 1000, 800, 500, 100, 80,50, 20],
                'drop_rate':[0.1],
                'device':[self.device],
                # 'cla_lr':[2e-3, 2e-2, 2e-4, 2e-5],
                # 'sann_lr':[2e-3, 2e-2, 2e-4, 2e-5],
                'lr': [2e-2, 2e-1, 1e-1, 3e-2, 2e-3, 2e-4, 1e-2, 2e-5],
                'batch_size':[1024],
                'tor':[5],
                'ori':[False, True]
            }
        model_param_grid[model_name] = (model_class, param_grid)

        model_name = 'TCMBert'
        model_class = TCMBert
        param_grid = {
            'drop_rate': [0.1],
            'lr': [2e-3, 2e-4, 2e-1, 1e-3],
            'bert_lr': [2e-5],
            'batch_size': [256],
            'tor': [5],
            'max_len': [512],
            'bert_dim': [768],
            'bert_path': [pathlib.Path(__file__).parent.parent.absolute() / 'model' / 'bert' / 'tcmbert'],
        }
        model_param_grid[model_name] = (model_class, param_grid)


        model_name = 'ZYBert'
        model_class = ZYBert
        param_grid = {
                'drop_rate': [0.1],
                'lr': [2e-3, 2e-4, 2e-1, 1e-3],
                'bert_lr': [2e-5],
                'batch_size':[128],
                'tor':[5],
                'max_len': [512],
                'bert_dim': [1024],
                'bert_path': [pathlib.Path(__file__).parent.parent.absolute() / 'model' / 'bert' / 'zybert'],
            }
        model_param_grid[model_name] = (model_class, param_grid)

        #
        model_name = 'BertCNN'
        model_class = BertCNN
        param_grid = {
                'drop_rate': [0.1],
                'lr': [2e-3, 2e-4, 2e-1, 1e-3],
                'bert_lr': [2e-5],
                'batch_size':[256],
                'tor':[5],
                'max_len': [512],
                'bert_dim': [768],
                'bert_path': [pathlib.Path(__file__).parent.parent.absolute() / 'model' / 'bert' / 'bert-base-chinese'],
            }
        model_param_grid[model_name] = (model_class, param_grid)

        return model_param_grid

    def opt_model(self, model_class, param_grid, model_name):
        """
        .
        """
        dataset_name_list = [str(i) for i in range(1,len(self.data)+1)] + ['mean']

        out = pd.DataFrame(
            data=np.ones(shape=[len(self.metrics)+1, len(dataset_name_list)])*-1,
            columns=dataset_name_list,
            index=self.metrics+['best_params'],
            dtype='object'
        )


        # 迭代参数
        multi_list = []
        for params in get_para_example(param_grid):
            if self.default_args:
                multi_list.append((model_class, model_name, {}))
            else:
                multi_list.append((model_class, model_name, params))

        if self.process_num == 1:
            results = []
            for i in tqdm.tqdm(multi_list):
                results.append(self.get_metric(i))
        else:
            with Pool(processes=self.process_num) as pool:
                results = list(tqdm.tqdm(pool.imap(self.get_metric, multi_list), total=len(multi_list)))


        opt_result = sorted(results, key=lambda x: x[0], reverse=True)[0]

        out.iloc[-1, 0] = str(opt_result[3])

        for i, data in enumerate(opt_result[1]):
            try:
                for key, value in data.items():
                    out.iloc[self.metrics.index(key), i] = value
                    out.iloc[self.metrics.index(key), -1] = out.iloc[self.metrics.index(key), :-1].mean()
            except Exception:
                print('error')

        return out

    def get_metric(self, multi_data):
        """
        :return:
        """
        model_class, model_name, params = multi_data

        metrics = []
        opt_metric_cv = []
        opt_model = None
        opt_model_list = []

        for data_i, (train_x, train_y, test_x, test_y) in enumerate(self.data):


            params_ = params.copy()
            if 'sann' in model_name:
                # if 'lr' in params_.keys(): del params_['lr']
                # if 'batch_size' in params_.keys(): del params_['batch_size']
                # if 'tor' in params_.keys(): del params_['tor']
                params_['x'] = train_x
                params_['y'] = train_y
            elif 'Bert' in model_name:
                params_['output_dim'] = train_y.shape[-1]


            model = model_class(**params_)
            # model = torch.nn.DataParallel(model, device_ids=[0,1])
            if 'Bert' in model_name and 'CNN' not in model_name:
                optimizer = AdamW([
                    {'params': model.bert.parameters(), 'lr': params['bert_lr']},
                    {'params': model.classifier.parameters(), 'lr': params['lr']},
                ])
            elif 'BertCNN' == model_name:
                optimizer = AdamW([
                    {'params': model.bert.parameters(), 'lr': params['bert_lr']},
                    {'params': model.classifier.parameters(), 'lr': params['lr']},
                    {'params': model.convs.parameters(), 'lr': params['lr']}
                ])
            else:
                optimizer = AdamW([
                    {'params': model.parameters(), 'lr': params['lr']}
                ])

            train_loader = DataLoader(
                dataset=CustomDataset(
                    train_x, train_y, self.device, bert=True if 'Bert' in model_name else False,
                    bert_path=params_['bert_path'] if 'Bert' in model_name else None, max_len=params_['max_len']  if 'Bert' in model_name else None
                ),
                shuffle=False,
                batch_size=int(params['batch_size']),
                num_workers=0 if 'sann' in model_name else 50,
                collate_fn=padbatch2tokenid if 'Bert' in model_name else None,

            )

            test_loader = DataLoader(
                dataset=CustomDataset(
                    test_x, test_y, self.device, bert=True if 'Bert' in model_name else False,
                    bert_path=params_['bert_path'] if 'Bert' in model_name else None, max_len=params_['max_len'] if 'Bert' in model_name else None
                ),
                shuffle=False,
                batch_size=int(params['batch_size']),
                num_workers=0 if 'sann' in model_name else 50,
                collate_fn=padbatch2tokenid if 'Bert' in model_name else None,
            )


            opt_train_metrics = None
            opt_test_metrics = None
            opt_metric = 0
            self.tor = params['tor']
            opt_model_in_cv = None


            for e in range(self.epoch):
                model.train()
                model.to(self.device)
                loss_list = []
                for batch_x, batch_y in train_loader:
                    if isinstance(batch_x, torch.Tensor):
                        batch_x = batch_x.to(self.device)
                    else:
                        batch_x = (i.to(self.device) for i in batch_x)
                    optimizer.zero_grad()
                    predict_y =  model.forward(batch_x)

                    if self.device == 'cuda':
                        batch_y = batch_y.to('cuda')

                    if self.type == 'multi_class':
                        predict_y = torch.nn.Softmax(dim=-1)(predict_y)
                        batch_y = torch.argmax(batch_y, dim=-1).reshape([-1])
                        loss = torch.nn.CrossEntropyLoss(reduction='mean')(predict_y, batch_y)
                    elif self.type == 'multi_label':
                        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(predict_y, batch_y.float())
                    else:
                        raise UserWarning('waiting......')
                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())

                # if model_name != 'sann':
                if data_i == 0:
                    print(f'{e}: {sum(loss_list) / len(loss_list)}')

                if self.type == 'multi_class':
                    train_metrics = self.evaluate_cla(model, train_loader)
                    test_metrics = self.evaluate_cla(model, test_loader)
                elif self.type == 'multi_label':
                    train_metrics = self.evaluate_label(model, train_loader)
                    test_metrics = self.evaluate_label(model, test_loader)
                else:
                    raise UserWarning('waiting......')

                # if model_name != 'sann':
                #     print(f'{e}: {test_metrics}')

                cur_metric = test_metrics[self.metric_name]
                if cur_metric > opt_metric:
                    opt_metric = cur_metric
                    tor = 0
                    opt_train_metrics = train_metrics
                    opt_test_metrics = test_metrics
                    opt_model_in_cv = model
                else:
                    tor += 1
                    if tor >= self.tor:
                        break

            metrics.append(opt_test_metrics)
            opt_metric_cv.append(opt_metric)
            opt_model_list.append(opt_model_in_cv)

        # 基于五折评价指标，选出最好的一折作为最优模型并保存
        opt_model = opt_model_list[np.argmax(opt_metric_cv)]
        if self.save_model:
            torch.save(opt_model, self.save_dir / f'{model_name}_opt.pkl')
            for i in range(5):
                torch.save(opt_model_list[i], self.save_dir / f'{model_name}_{i}.pkl')

        print(model_name, metrics)

        return np.array(opt_metric_cv).mean(), metrics, model_name, params

    def run(self):
        outcome = {}
        for model_name, (model_class, param_grid) in self.model_param_grid.items():

            if self.only_model:
                if model_name == self.only_model:
                    pass
                else:
                    continue

            print(model_name)

            #
            if self.default_args:
                save_path = self.save_dir / f'{model_name}_default_{self.top_k}.xlsx'
            else:
                save_path = self.save_dir / f'{model_name}_opt_{self.top_k}.xlsx'
            if os.path.exists(save_path):
                print('结果已存在，修改实验请先移除原结果')
                continue
            else:
                out = self.opt_model(model_class, param_grid, model_name=model_name)
                outcome[model_name] = out
                out.to_excel(save_path)


        return outcome

    def evaluate_cla(self, model, dataloader):
        model.eval()
        model = model.to(self.device)
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                if isinstance(batch_x, torch.Tensor):
                    batch_x = batch_x.to(self.device)
                else:
                    batch_x = (i.to(self.device) for i in batch_x)
                predict_y = model.forward(batch_x)

                predict_y = torch.nn.Softmax(dim=-1)(predict_y)
                batch_y = torch.nn.Softmax(dim=-1)(batch_y.float())

                predict_y = torch.argmax(predict_y, dim=-1)
                batch_y = torch.argmax(batch_y, dim=-1)

                predictions.extend(predict_y.detach().cpu().tolist())
                true_labels.extend(batch_y.detach().cpu().tolist())

        return {
            "precision": precision_score(true_labels, predictions, average="macro"),
            "recall": recall_score(true_labels, predictions, average="macro"),
            "f1": f1_score(true_labels, predictions, average="macro"),
            "accuracy": accuracy_score(true_labels, predictions),

        }

    def evaluate_label(self, model, dataloader):

        evaluator = MultiLabelEvaluator(top_k=self.top_k)

        model.eval()
        model = model.to(self.device)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                if isinstance(batch_x, torch.Tensor):
                    batch_x = batch_x.to(self.device)
                else:
                    batch_x = (i.to(self.device) for i in batch_x)
                predict_y = model.forward(batch_x)
                predict_y = predict_y.detach().cpu()
        return {
            "precision": evaluator.precision(batch_y, predict_y),
            "recall": evaluator.recall(batch_y, predict_y),
            "f1": evaluator.f1(batch_y, predict_y),
            # "accuracy": accuracy_score(true_labels, predictions),

        }

if __name__ == '__main__':
    pass