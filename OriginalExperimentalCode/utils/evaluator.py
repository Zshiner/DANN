import torch
import numpy as np
from config.config import Config

class MultiLabelEvaluator:
    def __init__(self, top_k=2):
        self.top_k = top_k

    def get_top_k(self, predicted_labels: torch.Tensor):
        """
        保留top，置零非top的方式对预测数据进行处理。
        :param predicted_labels:
        :return:
        """

        # 获取每行top的索引
        values, indices = torch.topk(predicted_labels, k=self.top_k, dim=-1)

        # 创建一个与输入形状相同的全零张量
        result = torch.zeros_like(predicted_labels)

        # 使用 scatter_ 将前两个最大值的位置设为1
        result.scatter_(-1, indices, 1)

        return result

    def precision(self, true_labels: torch.Tensor, predicted_labels: torch.Tensor):
        """

        :param true_labels:
        :param predicted_labels:
        :return:
        """

        predicted_labels = self.get_top_k(predicted_labels)

        predicted_total = predicted_labels.sum(dim=-1).sum(dim=-1).item()

        successful_labels = ((predicted_labels + true_labels) >= 2).sum().item()

        return successful_labels / predicted_total

    def recall(self, true_labels: torch.Tensor, predicted_labels: torch.Tensor):
        """"""
        predicted_labels = self.get_top_k(predicted_labels)

        true_total = true_labels.sum(dim=-1).sum(dim=-1).item()

        successful_labels = ((predicted_labels + true_labels) >= 2).sum().item()

        return successful_labels / true_total

    def f1(self, true_labels: torch.Tensor, predicted_labels: torch.Tensor):

        first = 2*(self.precision(true_labels, predicted_labels) * self.recall(true_labels, predicted_labels))
        second = self.precision(true_labels, predicted_labels) + self.recall(true_labels, predicted_labels)

        try:
            return first / second
        except ZeroDivisionError:
            return 0

    def accuracy(self, true_labels: torch.Tensor, predicted_labels: torch.Tensor):
        """

        :param true_labels:
        :param predicted_labels:
        :return:
        """
        # predicted_labels = self.get_top_k(predicted_labels)
        #
        #
        # successful_positive_labels = ((predicted_labels + true_labels) >= 2).sum().item()
        #
        # successful_negative_labels = predicted_labels + true_labels
        # successful_negative_labels = torch.where(successful_negative_labels > 0, torch.tensor(1000), successful_negative_labels)
        # successful_negative_labels = torch.where(successful_negative_labels == 0, torch.tensor(1), successful_negative_labels)
        # successful_negative_labels = torch.where(successful_negative_labels == 1000, torch.tensor(0),successful_negative_labels).sum().item()
        #
        # all_labels = predicted_labels.shape[0]*predicted_labels.shape[1]
        # return None
        # return (successful_positive_labels) / all_labels

if __name__ == '__main__':

    np.random.seed(Config.seed)

    num_samples = 10
    num_labels = 8

    y_pred = np.random.rand(num_samples, num_labels)

    y_true = np.zeros((num_samples, num_labels), dtype=int)
    for i in range(num_samples):
        # 随机选择 1~2 个正标签
        positive_labels = np.random.choice(num_labels, size=np.random.randint(1, 3), replace=False)
        y_true[i, positive_labels] = 1  # 选中的标签设为 1

    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    evaluator = MultiLabelEvaluator(top_k=2)
    print(evaluator.precision(y_true, y_pred))
    print(evaluator.recall(y_true, y_pred))
    print(evaluator.f1(y_true, y_pred))
    print(evaluator.accuracy(y_true, y_pred))