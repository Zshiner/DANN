# 基于参数优化的多分类对比实验框架。
import os.path
import abc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import numpy as np
import pandas as pd
# import pathlib
import pickle
import torch
from torch.optim import AdamW
import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from multiprocessing import Pool
from utils.evaluator import MultiLabelEvaluator
import pathlib
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

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

class Comparator(abc.ABC):
    def __init__(
            self,
            data,
            dataset_name,
            save_dir,
            default_args,
            process_num=1,
            metric_name='f1',
            seed=0,
            device='cuda',
            verbose=0,
            type='multi_class',   # multi_class multi_label
            top_k=None,
            only_model=None,
            save_model=True
    ):
        if type == 'multi_label':
            if top_k is None:
                raise ValueError('top_k must be provided with type multi_label')

        self.save_model = save_model
        self.top_k = top_k
        self.type = type
        self.seed = seed
        np.random.seed(self.seed)
        self.device = device
        self.verbose = verbose
        self.save_dir = pathlib.Path(save_dir).absolute()
        self.metric_name = metric_name

        if process_num == 1:
            warnings.warn("机器学习模型并发进程为1，请确认......")
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

        # Logistic Regression(LR)(Cox, 1958)
        model_name = 'lr'
        model_class = LogisticRegression
        param_grid = {}
        model_param_grid[model_name] = (model_class, param_grid)

        # GBDT
        model_name = 'gbdt'
        model_class = GradientBoostingClassifier
        param_grid = {
            'n_estimators': [100, 200, 500],    # 移除极值(10/1000)和中间冗余值(50/300/800)
            'learning_rate': [0.1, 0.2, 0.3],   # 聚焦关键区间(0.1-0.3)，移除>0.3的高学习率
            'max_depth': [2, 3, 4],             # 移除深度1（通常效果最差）
            'random_state': [self.seed]
        }
        model_param_grid[model_name] = (model_class, param_grid)
        # #
        # # RandomForestClassifier
        model_name = 'rf'
        model_class = RandomForestClassifier
        param_grid = {
            'n_estimators': [10, 50, 100, 200, 300, 500, 800, 1000],
            'max_features': [None, 'sqrt', 'log2'],
            'random_state': [self.seed]

        }

        model_param_grid[model_name] = (model_class, param_grid)


        # # Extreme Learning Machine Huang et al.,2012
        # model_name = 'elm'
        # model_class = ELM
        # param_grid = {
        #     'num_hidden':[100, 200]
        # }
        # model_param_grid[model_name] = (model_class, param_grid)

        # MLP
        # model_name = 'MLP'
        # model_class = MLPClassifier
        # param_grid = {
        #     'hidden_layer_sizes': [
        #         (i,) for i in [10, 30, 50, 100, 200, 300, 500]
        #
        #     ],
        #     'learning_rate': ['invscaling', 'constant', 'adaptive']
        # }
        # model_param_grid[model_name] = (model_class, param_grid)

        # AdaBoost (AB) (Freund & Schapire, 1996)
        model_name = 'ad'
        model_class = AdaBoostClassifier
        param_grid = {
            'random_state': [self.seed],
        }
        model_param_grid[model_name] = (model_class, param_grid)

        # C4.5
        model_name = 'C4.5'
        model_class = DecisionTreeClassifier
        param_grid = {
            'criterion': ['entropy'],
            'random_state': [self.seed],
        }
        model_param_grid[model_name] = (model_class, param_grid)

        # KNN
        model_name = 'KNN'
        model_class = KNeighborsClassifier
        param_grid = {
            'n_neighbors': [i for i in range(1, 7)],
        }

        model_param_grid[model_name] = (model_class, param_grid)

        # Naive Bayes classifier (NB) (Duda et al., 2000)
        model_name = 'NB'
        model_class = GaussianNB
        param_grid = {
            'priors': [None],
        }

        model_param_grid[model_name] = (model_class, param_grid)

        # # sann
        # model_name = 'sann'
        # model_class = SANN
        # param_grid = [
        #     {
        #         'lr': [1e-2, 1e-3, 1e-4],
        #         'hidden_dim': [10, 20, 40, 80, 90, 100, 200],
        #         'random_state': [self.seed],
        #         'drop_rate':[0.1, 0.001, 0.0001],
        #         'tor_num':[2,3,4],
        #         'ori':[True, False],
        #         'device':[self.device],
        #         'batch_size':[512]
        #     }
        # ]

        # SVC
        model_name = 'svc'
        model_class = SVC
        param_grid = {
            'C': [1.0, 0.9, 0.8, 0.75],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'random_state': [self.seed]

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
            for key, value in data.items():
                out.iloc[self.metrics.index(key), i] = value
                out.iloc[self.metrics.index(key), -1] = out.iloc[self.metrics.index(key), :-1].mean()

        return out

    def get_metric(self, multi_data):
        """
        :return:
        """
        model_class, model_name, params = multi_data

        metrics = []
        opt_metric_cv = []
        opt_model_list = []

        for train_x, train_y, test_x, test_y in self.data:

            params_ = params.copy()

            model = MultiOutputClassifier(model_class(**params_))

            model.fit(train_x, train_y)

            if self.type == 'multi_class':
                train_metrics = self.evaluate_cla(model, train_x, train_y)
                test_metrics = self.evaluate_cla(model, test_x, test_y)
            elif self.type == 'multi_label':
                train_metrics = self.evaluate_label(model, train_x, train_y)
                test_metrics = self.evaluate_label(model, test_x, test_y)
            else:
                raise UserWarning('任务类型未指定')

            opt_train_metrics = train_metrics
            opt_test_metrics = test_metrics
            opt_metric = test_metrics[self.metric_name]

            metrics.append(opt_test_metrics)
            opt_metric_cv.append(opt_metric)
            opt_model_list.append(model)


        # 基于五折评价指标，选出最好的一折作为最优模型并保存
        opt_model = opt_model_list[np.argmax(opt_metric_cv)]
        if self.save_model:
            with open(self.save_dir / f'{model_name}_opt.pkl', 'wb') as f:
                pickle.dump(opt_model, f)
            for i in range(5):
                with open(self.save_dir / f'{model_name}_{i}.pkl', 'wb') as f:
                    pickle.dump(opt_model_list[i], f)


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

    def evaluate_cla(self, model, x, y):

        pre_y = model.predict(x)

        return {
            "precision": precision_score(y, pre_y, average="macro"),
            "recall": recall_score(y, pre_y, average="macro"),
            "f1": f1_score(y, pre_y, average="macro"),
            "accuracy": accuracy_score(y, pre_y),
        }

    def evaluate_label(self, model, x, y):

        evaluator = MultiLabelEvaluator(top_k=self.top_k)

        pre_y = model.predict(x)
        pre_y = torch.tensor(pre_y)
        y = torch.tensor(np.array(y))

        return {
            "precision": evaluator.precision(y, pre_y),
            "recall": evaluator.recall(y, pre_y),
            "f1": evaluator.f1(y, pre_y),
            # "accuracy": accuracy_score(true_labels, predictions),

        }

if __name__ == '__main__':
    pass