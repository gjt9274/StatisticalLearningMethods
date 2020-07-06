"""
@File:    svm
@Author:  GongJintao
@Create:  7/5/2020 4:04 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
import pandas as pd
from load_data import load_data, normalize
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, max_epochs=100, kernel='Linear',C = 1.0 ,epsilon=1e-8,sigma=1):
        self.max_epochs = max_epochs  #最大迭代次数
        self._kernel = kernel   #核函数类型
        self.epsilon = epsilon  #精度epsilon
        self.sigma = sigma      #高斯核函数的sigma
        self.C = C              #松弛变量

    def init_args(self, features, labels):
        """
        初始化参数
        """

        # n指输入样本个数，m是特征维数
        self.n, self.m = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.K = self.construct_kernel_matrix()
        self.alpha = np.ones(self.n)
        self.E = [self._E(i) for i in range(self.n)]  # 差值 E_i,保存到一个表里

    def construct_kernel_matrix(self):
        """
        计算样本的核函数矩阵
        Returns:
            K (array): 核函数矩阵，(n x n)
        """
        K = np.zeros((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                K[i][j] = self.kernel(self.X[i],self.X[j])
                K[j][i] = K[i][j]

        return K

    def kernel(self, x1, x2):
        """
        核函数
        Args:
            x1 (array): 第一个样本 (m x 1)
            x2 (array): 第二个样本 (m x 1)

        Returns:
            'Linear': 未使用核函数
            'Poly'  : 多项式核
            'Gaussian': 高斯核

        """
        x1 = np.array(x1).reshape(-1, 1)
        x2 = np.array(x2).reshape(-1, 1)
        assert(x1.shape == (self.m, 1))
        assert(x2.shape == (self.m, 1))
        if self._kernel == 'Poly':
            return (np.dot(x1.T, x2) + 1)**2
        elif self._kernel == 'Gaussian':
            return np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * self.sigma**2))
        else:
            return np.dot(x1.T, x2)

    def _KKT(self, i):
        """
        停机条件
        Args:
            i (int): 索引号

        Returns:
            (bool) 是否满足停机条件
        """
        y_g = self._g(i)*self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1-self.epsilon
        elif 0 < self.alpha[i] < self.C:
            return 1-self.epsilon <= y_g <=1+self.epsilon
        else:
            return y_g <= 1+self.epsilon


    def _g(self, i):
        """
        g(x) = \sum_{i}^{N} \alpha_{i} y_{i} K(x_i,x) + b
        Args:
            i (int): 索引号

        Returns:
            r (float): g(x_i)
        """
        r = self.b
        for j in range(self.n):
            r += self.alpha[j] * self.Y[j] * self.K[j][i]
        return r

    def _E(self, i):
        """
        函数 g(x) 对输入x_i 的预测值与真实输出 y_i 之差
        Args:
            i (int): 索引号

        Returns:
            g(x_i) - y_i
        """
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        """
        初始化alpha变量
        Returns:
            i,j (int,int): 表示alpha1,alpha2所对应的索引
        """
        # 外层循环首先遍历所有满足条件 0<alpha_i<C的样本点
        index_list = [i for i in range(self.n) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.n) if i not in index_list]
        index_list.extend(non_satisfy_list)

        for i in index_list:
            if self._KKT(i):
                continue    # 如果满足停机条件，则继续

            # 选取第二个变量
            E1 = self.E[i]
            if E1 >= 0:  # 如果E1是正数，则选择最小的E_i作为E2
                j = np.argmin(self.E)
            else:  # 如果E1是负数，则选择最大的E_i 作为E2
                j = np.argmax(self.E)
            return i, j

    def clip_alpha(self, alpha_unc, L, H):
        """
        对未经剪辑的alpha的解进行剪辑
        Args:
            alpha_unc (float): 未经剪辑的alpha
            L (float): alpha不等式约束的左端点
            H (float): alpha不等式约束的右端点

        Returns:
            经过剪辑后的解 alpha
        """
        if alpha_unc > H:
            return H
        elif alpha_unc < L:
            return L
        else:
            return alpha_unc

    def fit(self, features, labels):
        """
        训练
        Args:
            features (array): 特征 (n x m)
            labels (array): 标签 (n x m)

        Returns:

        """
        # 1. 初始化参数
        self.init_args(features, labels)

        for epoch in range(self.max_epochs):
            # 2. 选择alpha
            i1, i2 = self._init_alpha()

            # 3. 求边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]

            # 4. 求解析解
            # eta = K11+K22-2k12
            eta = self.K[i1][i1] + self.K[i2][i2] -2*self.K[i1][i2]

            if eta <= 0:
                continue

            # 未剪辑的alpha2的解析解
            alpha2_new_unc = self.alpha[i2] + self.Y[i2]*(E1-E2)/eta
            # 剪辑后的alpha2的解析解
            alpha2_new = self.clip_alpha(alpha2_new_unc,L,H)
            # alpha1的解析解
            alpha1_new = self.alpha[i1] + self.Y[i1]*self.Y[i2]*(self.alpha[i2] - alpha2_new)
            # 阈值
            b1_new = -E1 - self.Y[i1]*self.K[i1][i1]*(alpha1_new - self.alpha[i1]) - self.Y[i2]*self.K[i2][i1]*(alpha2_new-self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1]*self.K[i1][i2]*(alpha1_new - self.alpha[i1]) - self.Y[i2]*self.K[i2][i2]*(alpha2_new-self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new= (b1_new + b2_new) / 2

            # 5. 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            # self.E[i1] = self._E(i1)
            # self.E[i2] = self._E(i2)
            self.E = [self._E(i) for i in range(self.n)]
        print('train over!')

    # def predict(self,data):
    #     """
    #     预测
    #     Args:
    #         data (array): 测试数据 (num_sample,m)
    #
    #     Returns:
    #         predictions (array): 预测标签 (num_sample,1)
    #     """
    #     r = self.b
    #     predictions = []
    #     for i in range(len(data)):
    #         for j in range(self.n):
    #             r += self.alpha[j]*self.Y[j]*self.kernel(data[i],self.X[j])
    #         predictions.append(1 if r >= 0 else -1)
    #     return predictions
    #
    # def score(self,X_test,Y_test):
    #     """
    #     计算测试得分
    #     Args:
    #         X_test (array): 测试样本 (num_sample,m)
    #         Y_test (array): 测试标签 (num_sample,1)
    #
    #     Returns:
    #         score (float): 测试准确率
    #     """
    #     predictions = self.predict(X_test)
    #     right_count = np.sum(predictions == Y_test)
    #     score = right_count/ len(X_test)
    #     return score

    def predict(self, data):
        r = self.b
        for i in range(self.n):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)

    def _weight(self):
        """
        求训练后SVM的参数 w
        Returns:
            w (array): 超平面 w (m,1)
        """
        yx = self.Y.reshape(-1,1) * self.X
        self.w = np.dot(yx.T,self.alpha)

        return self.w


def test(svm,test_file_path):
    """
    测试
    :param test_file_path: 测试文件所在路径
    :return:
    """
    # 1.准备数据
    test_data,passenger_id = load_data(test_file_path,training=False)

    # 2. 归一化
    normal_test_data = normalize(test_data)

    # 3. 预测
    predictions = svm.predict(normal_test_data)

    # 4. 转换成符合要求的数，并保存结果
    predictions = pd.Series(predictions) #输入必须是1维数据，要拉平

    result = pd.DataFrame({'PassengerId':passenger_id,'Survived':predictions})
    result.to_csv('./my_submission.csv',index=False)

def create_data():
    iris = load_iris()
    #选取前两类作为数据
    Data=np.array(iris["data"])[:100]
    Label=np.array(iris["target"])[:100]
    Label=Label*2-1
    print("dara shape:",Data.shape)
    print("label shape:",Label.shape)
    return Data, Label

if __name__ == "__main__":
    data, labels = load_data('./data/Titanic/train.csv')
    normal_data = normalize(data)
    svm = SVM(max_epochs=200)
    svm.fit(normal_data,labels)
    print(svm.score(normal_data,labels.flatten()))
    # test(svm,'./data/Titanic/train.csv')

    clf = SVC()
    clf.fit(normal_data, labels)
    print(clf.score(normal_data, labels))
