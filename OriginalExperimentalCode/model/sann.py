import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.decomposition import NMF
import random
import os
from sklearn.base import BaseEstimator, ClassifierMixin

class DANNModel(torch.nn.Module):
    def __init__(self, w1, w2, drop_rate, device='cpu'):
        super().__init__()
        self.device = device
        self.w1 = torch.tensor(w1, device=self.device, dtype=torch.float)
        self.w2 = torch.tensor(w2, device=self.device, dtype=torch.float)

        self.linear1 = torch.nn.Linear(*self.w1.shape, device=self.device)
        self.linear2 = torch.nn.Linear(*self.w2.shape, device=self.device)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(drop_rate)

        with torch.no_grad():
            self.linear1.weight.data = self.w1.T.clone()
            self.linear2.weight.data = self.w2.T.clone()


    def forward(self, x, label=None, require_prob=False):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.linear2(out)

        return out

class DANNModelOri(torch.nn.Module):
    def __init__(self, w, drop_rate=0., device='cpu'):
        super().__init__()
        self.device = device
        self.w = torch.tensor(w, device=self.device, dtype=torch.float)

        self.linear = torch.nn.Linear(*self.w.shape, device=self.device)

        with torch.no_grad():
            self.linear.weight.data = self.w.T.clone()

        self.drop = torch.nn.Dropout(drop_rate)


    def forward(self, x):
        out = self.drop(x)
        out = self.linear(out)

        return out

class DANN(torch.nn.Module):
    def __init__(self,
                 x,
                 y,    # 独热编码的数据，无论是多分类还是多标签
                 hidden_dim=None,
                 ori=True,
                 random_state=0,
                 drop_rate=0.1,
                 device='cpu',
                 features_name=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__()


        self.random_state = random_state
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.hidden_dim = hidden_dim
        self.seed = random_state
        self.drop_rate = drop_rate
        self.ori=ori

        self.ori_model = None
        self.trained_model = None
        self.features_name = []
        self.labels_name = []
        self.ori_weights = None
        self.device = device

        self.ori_weights = None
        self.trained_weights = None
        self.ori_weights_min = 0
        self.w1 = None
        self.w2 = None
        self.trained_w1 = None
        self.trained_w2 = None

        self.classes_ = None
        self.features_name = features_name

        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.int)

        # 分布感知初始化
        if self.ori:
            self.__fit_ori(x, y, features_name)
        else:
            self.__fit(x, y, features_name)

    def forward(self, x):
        return self.trained_model(x)

    def __MatrixFactorization(self, weights, hidden_dim):
        model = NMF(n_components=hidden_dim, init='random', max_iter=1000)
        W = model.fit_transform(weights)
        H = model.components_
        return W, H

    def __get_samples_by_label(self, x: torch.Tensor, y: torch.Tensor, label):
        """
        .
        """
        x_rows = y[:, label] == 1

        x_by_label = x[x_rows, :]
        return x_by_label

    def get_ori_weights(self, x: torch.Tensor, y: torch.Tensor):
        """
        .
        """
        # features*labels
        ori_weights = np.zeros(
            shape=[
                len(self.features_name),
                len(self.labels_name)
            ]
        )
        ori_weights = torch.tensor(ori_weights)

        # 将所有特征归一化为0-100
        x = 100*(x - x.min(dim=0)[0])/(x.max(dim=0)[0] - x.min(dim=0)[0])


        # 计算D
        x_ave = x.mean(dim=0)
        D = ori_weights
        for label_index in range(len(self.labels_name)):
            x_label = self.__get_samples_by_label(x, y, label=self.labels_name[label_index])
            D[:, label_index] = (x_label.mean(dim=0) - (0.5*x_ave)) / x_ave

        # 计算D_subtraction
        D_ave = D.mean(dim=1)
        D_ave = D_ave.reshape(shape=[-1, 1])
        D_ave = D_ave.repeat([1, D.shape[-1]])
        D_sub = D - D_ave

        #全局归一化到0-1
        temp = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )
        global_min = temp.min(dim=0)[0].min(dim=0)[0]
        global_max = temp.max(dim=0)[0].max(dim=0)[0]
        global_ave = temp.mean(dim=0).mean(dim=0)
        D_sub = (D_sub - global_ave) / (global_max-global_min)

        # 部分特征均值为0，在计算时权值赋值为nan。由于原始特征的均值为0，没有意义，因此将nan赋值为0
        D_sub = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )

        return D_sub

    def __fit(self, x, y, features_name=None):
        """
        .
        """
        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]
        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = self.get_ori_weights(x, y)
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )
        self.ori_weights_min = min(0., float(ori_weights.min(dim=0)[0].min(dim=0)[0]))

        # 矩阵分解
        weights = self.ori_weights - self.ori_weights_min
        if self.hidden_dim is not None:
            pass
        else:
            self.hidden_dim = weights.shape[0]*2
        w1, w2 = self.__MatrixFactorization(weights, hidden_dim=self.hidden_dim)
        self.w1 = w1
        self.w2 = w2


        # initialization and training
        self.ori_model = DANNModel(w1, w2, self.drop_rate, device=self.device)
        self.trained_model = DANNModel(w1, w2, self.drop_rate, device=self.device)

        self.trained_w1 = self.trained_model.linear1.weight.detach().cpu().numpy()
        self.trained_w2 = self.trained_model.linear2.weight.detach().cpu().numpy()
        self.trained_weights = pd.DataFrame(
            data=np.matmul(self.trained_w2, self.trained_w1).T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

    def __fit_ori(self, x, y, features_name=None):
        """
        .
        """

        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]

        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = np.array(self.get_ori_weights(x, y))
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )


        # initialization and training
        self.ori_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)
        self.trained_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)

        self.trained_weights = pd.DataFrame(
            data=self.trained_model.linear.weight.detach().cpu().numpy().T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

class DANN_without_D(torch.nn.Module):
    def __init__(self,
                 x,
                 y,    # 独热编码的数据，无论是多分类还是多标签
                 hidden_dim=None,
                 ori=True,
                 random_state=0,
                 drop_rate=0.1,
                 device='cpu',
                 features_name=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__()


        self.random_state = random_state
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.hidden_dim = hidden_dim
        self.seed = random_state
        self.drop_rate = drop_rate
        self.ori=ori

        self.ori_model = None
        self.trained_model = None
        self.features_name = []
        self.labels_name = []
        self.ori_weights = None
        self.device = device

        self.ori_weights = None
        self.trained_weights = None
        self.ori_weights_min = 0
        self.w1 = None
        self.w2 = None
        self.trained_w1 = None
        self.trained_w2 = None

        self.classes_ = None
        self.features_name = features_name

        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.int)

        # 分布感知初始化
        if self.ori:
            self.__fit_ori(x, y, features_name)
        else:
            self.__fit(x, y, features_name)

    def forward(self, x):
        return self.trained_model(x)

    def __MatrixFactorization(self, weights, hidden_dim):
        model = NMF(n_components=hidden_dim, init='random', max_iter=1000)
        W = model.fit_transform(weights)
        H = model.components_
        return W, H

    def __get_samples_by_label(self, x: torch.Tensor, y: torch.Tensor, label):
        """
        .
        """
        x_rows = y[:, label] == 1

        x_by_label = x[x_rows, :]
        return x_by_label

    def get_ori_weights(self, x: torch.Tensor, y: torch.Tensor):
        """
        .
        """
        # features*labels
        ori_weights = np.zeros(
            shape=[
                len(self.features_name),
                len(self.labels_name)
            ]
        )
        ori_weights = torch.tensor(ori_weights)

        # 将所有特征归一化为0-100
        x = 100*(x - x.min(dim=0)[0])/(x.max(dim=0)[0] - x.min(dim=0)[0])


        # 计算D
        x_ave = x.mean(dim=0)
        D = ori_weights
        for label_index in range(len(self.labels_name)):
            x_label = self.__get_samples_by_label(x, y, label=self.labels_name[label_index])
            D[:, label_index] = (x_label.mean(dim=0) - (0.5*x_ave)) / x_ave

        # drop D
        D = torch.randn_like(D)


        # 计算D_subtraction
        D_ave = D.mean(dim=1)
        D_ave = D_ave.reshape(shape=[-1, 1])
        D_ave = D_ave.repeat([1, D.shape[-1]])
        D_sub = D - D_ave

        #全局归一化到0-1
        temp = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )
        global_min = temp.min(dim=0)[0].min(dim=0)[0]
        global_max = temp.max(dim=0)[0].max(dim=0)[0]
        global_ave = temp.mean(dim=0).mean(dim=0)
        D_sub = (D_sub - global_ave) / (global_max-global_min)

        # 部分特征均值为0，在计算时权值赋值为nan。由于原始特征的均值为0，没有意义，因此将nan赋值为0
        D_sub = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )

        return D_sub

    def __fit(self, x, y, features_name=None):
        """
        .
        """
        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]
        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = self.get_ori_weights(x, y)
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )
        self.ori_weights_min = min(0., float(ori_weights.min(dim=0)[0].min(dim=0)[0]))

        # 矩阵分解
        weights = self.ori_weights - self.ori_weights_min
        if self.hidden_dim is not None:
            pass
        else:
            self.hidden_dim = weights.shape[0]*2
        w1, w2 = self.__MatrixFactorization(weights, hidden_dim=self.hidden_dim)
        self.w1 = w1
        self.w2 = w2


        # initialization and training
        self.ori_model = DANNModel(w1, w2, self.drop_rate, device=self.device)
        self.trained_model = DANNModel(w1, w2, self.drop_rate, device=self.device)

        self.trained_w1 = self.trained_model.linear1.weight.detach().cpu().numpy()
        self.trained_w2 = self.trained_model.linear2.weight.detach().cpu().numpy()
        self.trained_weights = pd.DataFrame(
            data=np.matmul(self.trained_w2, self.trained_w1).T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

    def __fit_ori(self, x, y, features_name=None):
        """
        .
        """

        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]

        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = np.array(self.get_ori_weights(x, y))
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )


        # initialization and training
        self.ori_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)
        self.trained_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)

        self.trained_weights = pd.DataFrame(
            data=self.trained_model.linear.weight.detach().cpu().numpy().T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

class DANN_without_F(torch.nn.Module):
    def __init__(self,
                 x,
                 y,    # 独热编码的数据，无论是多分类还是多标签
                 hidden_dim=None,
                 ori=True,
                 random_state=0,
                 drop_rate=0.1,
                 device='cpu',
                 features_name=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__()


        self.random_state = random_state
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.hidden_dim = hidden_dim
        self.seed = random_state
        self.drop_rate = drop_rate
        self.ori=ori

        self.ori_model = None
        self.trained_model = None
        self.features_name = []
        self.labels_name = []
        self.ori_weights = None
        self.device = device

        self.ori_weights = None
        self.trained_weights = None
        self.ori_weights_min = 0
        self.w1 = None
        self.w2 = None
        self.trained_w1 = None
        self.trained_w2 = None

        self.classes_ = None
        self.features_name = features_name

        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.int)

        # 分布感知初始化
        if self.ori:
            self.__fit_ori(x, y, features_name)
        else:
            self.__fit(x, y, features_name)

    def forward(self, x):
        return self.trained_model(x)

    def __MatrixFactorization(self, weights, hidden_dim):
        model = NMF(n_components=hidden_dim, init='random', max_iter=1000)
        W = model.fit_transform(weights)
        H = model.components_
        return W, H

    def __get_samples_by_label(self, x: torch.Tensor, y: torch.Tensor, label):
        """
        .
        """
        x_rows = y[:, label] == 1

        x_by_label = x[x_rows, :]
        return x_by_label

    def get_ori_weights(self, x: torch.Tensor, y: torch.Tensor):
        """
        .
        """
        # features*labels
        ori_weights = np.zeros(
            shape=[
                len(self.features_name),
                len(self.labels_name)
            ]
        )
        ori_weights = torch.tensor(ori_weights)

        # 将所有特征归一化为0-100
        x = 100*(x - x.min(dim=0)[0])/(x.max(dim=0)[0] - x.min(dim=0)[0])


        # 计算D
        x_ave = x.mean(dim=0)
        D = ori_weights
        for label_index in range(len(self.labels_name)):
            x_label = self.__get_samples_by_label(x, y, label=self.labels_name[label_index])
            D[:, label_index] = (x_label.mean(dim=0) - (0.5*x_ave)) / x_ave

        # 计算D_subtraction
        D_ave = D.mean(dim=1)
        D_ave = D_ave.reshape(shape=[-1, 1])
        D_ave = D_ave.repeat([1, D.shape[-1]])

        # drop F=D_ave
        D_ave = torch.randn_like(D_ave)

        D_sub = D - D_ave

        #全局归一化到0-1
        temp = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )
        global_min = temp.min(dim=0)[0].min(dim=0)[0]
        global_max = temp.max(dim=0)[0].max(dim=0)[0]
        global_ave = temp.mean(dim=0).mean(dim=0)
        D_sub = (D_sub - global_ave) / (global_max-global_min)

        # 部分特征均值为0，在计算时权值赋值为nan。由于原始特征的均值为0，没有意义，因此将nan赋值为0
        D_sub = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )

        return D_sub

    def __fit(self, x, y, features_name=None):
        """
        .
        """
        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]
        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = self.get_ori_weights(x, y)
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )
        self.ori_weights_min = min(0., float(ori_weights.min(dim=0)[0].min(dim=0)[0]))

        # 矩阵分解
        weights = self.ori_weights - self.ori_weights_min
        if self.hidden_dim is not None:
            pass
        else:
            self.hidden_dim = weights.shape[0]*2
        w1, w2 = self.__MatrixFactorization(weights, hidden_dim=self.hidden_dim)
        self.w1 = w1
        self.w2 = w2


        # initialization and training
        self.ori_model = DANNModel(w1, w2, self.drop_rate, device=self.device)
        self.trained_model = DANNModel(w1, w2, self.drop_rate, device=self.device)

        self.trained_w1 = self.trained_model.linear1.weight.detach().cpu().numpy()
        self.trained_w2 = self.trained_model.linear2.weight.detach().cpu().numpy()
        self.trained_weights = pd.DataFrame(
            data=np.matmul(self.trained_w2, self.trained_w1).T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

    def __fit_ori(self, x, y, features_name=None):
        """
        .
        """

        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]

        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = np.array(self.get_ori_weights(x, y))
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )


        # initialization and training
        self.ori_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)
        self.trained_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)

        self.trained_weights = pd.DataFrame(
            data=self.trained_model.linear.weight.detach().cpu().numpy().T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

class DANN_without_DF(torch.nn.Module):
    def __init__(self,
                 x,
                 y,    # 独热编码的数据，无论是多分类还是多标签
                 hidden_dim=None,
                 ori=True,
                 random_state=0,
                 drop_rate=0.1,
                 device='cpu',
                 features_name=None,
                 *args,
                 **kwargs,
                 ):
        super().__init__()


        self.random_state = random_state
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.hidden_dim = hidden_dim
        self.seed = random_state
        self.drop_rate = drop_rate
        self.ori=ori

        self.ori_model = None
        self.trained_model = None
        self.features_name = []
        self.labels_name = []
        self.ori_weights = None
        self.device = device

        self.ori_weights = None
        self.trained_weights = None
        self.ori_weights_min = 0
        self.w1 = None
        self.w2 = None
        self.trained_w1 = None
        self.trained_w2 = None

        self.classes_ = None
        self.features_name = features_name

        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.int)

        # 分布感知初始化
        if self.ori:
            self.__fit_ori(x, y, features_name)
        else:
            self.__fit(x, y, features_name)

    def forward(self, x):
        return self.trained_model(x)

    def __MatrixFactorization(self, weights, hidden_dim):
        model = NMF(n_components=hidden_dim, init='random', max_iter=1000)
        W = model.fit_transform(weights)
        H = model.components_
        return W, H

    def __get_samples_by_label(self, x: torch.Tensor, y: torch.Tensor, label):
        """
        .
        """
        x_rows = y[:, label] == 1

        x_by_label = x[x_rows, :]
        return x_by_label

    def get_ori_weights(self, x: torch.Tensor, y: torch.Tensor):
        """
        .
        """
        # features*labels
        ori_weights = np.zeros(
            shape=[
                len(self.features_name),
                len(self.labels_name)
            ]
        )
        ori_weights = torch.tensor(ori_weights)

        # 将所有特征归一化为0-100
        x = 100*(x - x.min(dim=0)[0])/(x.max(dim=0)[0] - x.min(dim=0)[0])


        # 计算D
        x_ave = x.mean(dim=0)
        D = ori_weights
        for label_index in range(len(self.labels_name)):
            x_label = self.__get_samples_by_label(x, y, label=self.labels_name[label_index])
            D[:, label_index] = (x_label.mean(dim=0) - (0.5*x_ave)) / x_ave

        # 计算D_subtraction
        D_ave = D.mean(dim=1)
        D_ave = D_ave.reshape(shape=[-1, 1])
        D_ave = D_ave.repeat([1, D.shape[-1]])
        D_sub = D - D_ave

        # drop D+F
        D_sub = torch.randn_like(D_sub) * (D_sub.max() - D_sub.min()) + D_sub.min()

        #全局归一化到0-1
        temp = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )
        global_min = temp.min(dim=0)[0].min(dim=0)[0]
        global_max = temp.max(dim=0)[0].max(dim=0)[0]
        global_ave = temp.mean(dim=0).mean(dim=0)
        D_sub = (D_sub - global_ave) / (global_max-global_min)

        # 部分特征均值为0，在计算时权值赋值为nan。由于原始特征的均值为0，没有意义，因此将nan赋值为0
        D_sub = torch.where(
            torch.isnan(D_sub),
            torch.full_like(D_sub, 0),
            D_sub,
        )

        return D_sub

    def __fit(self, x, y, features_name=None):
        """
        .
        """
        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]
        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = self.get_ori_weights(x, y)
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )
        self.ori_weights_min = min(0., float(ori_weights.min(dim=0)[0].min(dim=0)[0]))

        # 矩阵分解
        weights = self.ori_weights - self.ori_weights_min
        if self.hidden_dim is not None:
            pass
        else:
            self.hidden_dim = weights.shape[0]*2
        w1, w2 = self.__MatrixFactorization(weights, hidden_dim=self.hidden_dim)
        self.w1 = w1
        self.w2 = w2


        # initialization and training
        self.ori_model = DANNModel(w1, w2, self.drop_rate, device=self.device)
        self.trained_model = DANNModel(w1, w2, self.drop_rate, device=self.device)

        self.trained_w1 = self.trained_model.linear1.weight.detach().cpu().numpy()
        self.trained_w2 = self.trained_model.linear2.weight.detach().cpu().numpy()
        self.trained_weights = pd.DataFrame(
            data=np.matmul(self.trained_w2, self.trained_w1).T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self

    def __fit_ori(self, x, y, features_name=None):
        """
        .
        """

        if features_name is not None:
            self.features_name = list(features_name)
        else:
            self.features_name = [f'feature_{i}' for i in range(x.shape[1])]

        self.labels_name = [i for i in range(y.shape[1])]
        self.classes_ = [i for i in range(y.shape[1])]


        # get ori weights
        ori_weights = np.array(self.get_ori_weights(x, y))
        self.ori_weights = pd.DataFrame(
            data=ori_weights,
            index=self.features_name,
            columns=self.labels_name
        )


        # initialization and training
        self.ori_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)
        self.trained_model = DANNModelOri(ori_weights, drop_rate=self.drop_rate, device=self.device)

        self.trained_weights = pd.DataFrame(
            data=self.trained_model.linear.weight.detach().cpu().numpy().T,
            index=self.features_name,
            columns=self.labels_name
        )

        return self




if __name__ == '__main__':
    # DANN = DANN(device='cpu', lr=1e-4, tor_num=5, batch_size=5000, epoch=1000, seed=0)
    # DANN.fit(x, y, features_name)
    # pre = DANN.predict(x)
    # print(1)
    pass