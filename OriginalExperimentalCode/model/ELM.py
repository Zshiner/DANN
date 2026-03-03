import numpy as np

class ELM():
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden

    def fit(self, X, y):
        # 随机初始化隐含层权重
        self.hidden_weights = np.random.randn(X.shape[1], self.num_hidden)
        # 计算隐含层输出
        self.hidden_output = np.dot(X, self.hidden_weights)
        # 计算输出层权重
        self.output_weights = np.linalg.pinv(self.hidden_output) @ y

    def predict(self, X):
        # 计算隐含层输出
        hidden_output = np.dot(X, self.hidden_weights)
        # 计算预测输出
        y_pred = hidden_output @ self.output_weights
        return y_pred

    def set_params(self, **kwargs):
        """
        .
        """
        for name, value in kwargs.items():
            setattr(self, name, value)

        return self

    def get_params(self, deep):
        return {
            'num_hidden': self.num_hidden,
        }