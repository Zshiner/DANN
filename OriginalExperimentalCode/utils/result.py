import os
from operator import index

import numpy as np
import pandas as pd
import pathlib

class Result():
    def __init__(self, outcome_dir='./out', save_dir='./result'):
        self.outcome = None
        self.metrics_name = ['f1', 'accuracy', 'recall', 'precision']

        self.outcome_dir = pathlib.Path(outcome_dir)
        self.save_dir = pathlib.Path(save_dir)

    def read_outcome(self):
        outcome = []
        for root, dirs, files in os.walk(self.outcome_dir):
            for dataset_name in dirs:
                for _, _, files in os.walk(pathlib.Path(root) / dataset_name):
                    for model_file in files:
                        model_path = pathlib.Path(root) / dataset_name / model_file
                        model_name = model_path.stem
                        data = pd.read_excel(model_path)

                        #
                        cur_outcome = {}
                        cur_outcome['dataset'] = dataset_name
                        cur_outcome['model'] = model_name
                        cur_outcome['parameters'] = data.iloc[-1, 1]

                        #
                        metrics = data.iloc[:-1, 1:]
                        metrics.index = pd.Index(list(data.iloc[:,0])[:-1])
                        cur_outcome['metrics'] = metrics
                        outcome.append(cur_outcome)

        self.outcome = outcome

    def get_all_mean(self):
        """
        获取在所有数据集中，不同模型的平均指标
        :return:
        """
        model_list = set([i['model'] for i in self.outcome])
        result = []
        result_index = []

        for model_name in model_list:
            model_outcome = [i['metrics'].iloc[:,-1] for i in self.outcome if i['model']==model_name]
            model_outcome = np.array(model_outcome).mean(axis=0)
            result.append(model_outcome)
            result_index.append(model_name)

        result = pd.DataFrame(np.array(result), index=result_index, columns=self.metrics_name)

        result.to_excel(self.save_dir / 'all_datasets_mean.xlsx')


    def run(self):
        self.read_outcome()
        self.get_all_mean()


if __name__ == '__main__':
    res = Result()
    res.run()
