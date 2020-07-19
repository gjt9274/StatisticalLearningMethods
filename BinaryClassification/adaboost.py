"""
@File:    adaboost
@Author:  GongJintao
@Create:  7/6/2020 8:11 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""
from load_data import *
import numpy as np
import pandas as pd


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.clf_nums = n_estimators
        self.learning_rate = learning_rate

    def init_args(self, datasets, labels):
        self.X = datasets
        self.Y = labels
        self.N, self.M = datasets.shape

        self.clf_sets = []  # 弱分类器集合

        self.weights = [1.0 / self.N] * self.N  # 初始化权值矩阵

        self.alpha = []  # 分类器G(x) 的系数

    def build_stump(self, X, Y, weights):
        """
        构建单层决策树桩
        Args:
            X (array): 样本特征 (N, M)
            Y (array): 样本标签 (N, 1)
            weights (list): 权值矩阵 len = N

        Returns:
            axis (int): 最终使用的特征维度编号
            best_v (float): 最终判断阈值
            direct (string): 'positive' 表示大于阈值则预测结果为1，‘negative'表示大于阈值预测结果为-1
            clr_error (float): 最终的分类误差率
            compare_array: (array): 最终的预测结果 (N, 1)
        """

        clf_error = np.inf  # 初始化错误率为无穷大
        best_v = 0.0  # 初始化阈值
        direct = None  # 判断方向
        compare_array = None  # 最终的预测结果
        axis = 0  # 特征维度
        # 遍历每个特征
        for j in range(self.M):
            features = X[:, j]
            features_min = min(features)
            features_max = max(features)
            n_step = (features_max - features_min +
                      self.learning_rate) // self.learning_rate

            for i in range(1, int(n_step)):
                v = features_min + self.learning_rate * i
                if v not in features:
                    # 计算分类误差率
                    # 以 v 为阈值将特征分为两部分，一部分的标签为1，另一部分为-1
                    # 再计算分类误差率，即分错样本的权值和
                    compare_array_positive = np.array(
                        [1 if features[k] > v else -1 for k in range(self.N)]
                    )
                    weight_error_positive = sum([
                        weights[k] for k in range(self.N)
                        if compare_array_positive[k] != Y[k]
                    ])

                    compare_array_negative = np.array(
                        [-1 if features[k] > v else 1 for k in range(self.N)]
                    )
                    weight_error_negative = sum([
                        weights[k] for k in range(self.N)
                        if compare_array_negative[k] != Y[k]
                    ])

                    if weight_error_positive < weight_error_negative:
                        weight_error = weight_error_positive
                        _compare_array = compare_array_positive
                        _direct = 'positive'
                    else:
                        weight_error = weight_error_negative
                        _compare_array = compare_array_negative
                        _direct = 'negative'

                    if weight_error < clf_error:
                        clf_error = weight_error
                        compare_array = _compare_array
                        best_v = v
                        direct = _direct
                        axis = j

        return axis, best_v, direct, clf_error, compare_array

    def _alpha(self, error):
        """
        计算分类器的系数 alpha
        Args:
            error (float): 分类错误率

        Returns:

        """
        return 0.5 * np.log((1 - error) / error)

    def _Z(self, weights, a, clf):
        """
        计算规范化因子
        Args:
            weights (list): 权值向量
            a (float): 第m个分类器系数
            clf (): 第m个分类器

        Returns:

        """
        return sum([
            weights[i] * np.exp(-1 * a * self.Y[i] * clf[i])
            for i in range(self.N)
        ])

    def _update_weights(self, a, clf, Z):
        """
        更新权值向量
        Args:
            a (float): 分类器系数
            clf (): 分类器
            Z (float): 规范化因子

        Returns:

        """
        for i in range(self.N):
            self.weights[i] = self.weights[i] * np.exp(
                -1 * a * self.Y[i] * clf[i]
            ) / Z

    def classify(self, x, v, direct):
        """
        分类
        Args:
            x (float): 样本特征
            v (float): 阈值
            direct (string): 判断方向

        Returns:

        """
        if direct == 'positive':
            return 1 if x > v else -1
        else:
            return -1 if x > v else 1

    def fit(self, X, Y):
        self.init_args(X, Y)

        for epoch in range(self.clf_nums):
            axis, best_v, direct, clf_error, clf_result = self.build_stump(
                self.X, self.Y, self.weights
            )

            a = self._alpha(clf_error)
            self.alpha.append(a)

            self.clf_sets.append((axis, best_v, direct))

            Z = self._Z(self.weights, a, clf_result)

            self._update_weights(a, clf_result, Z)

    def predict(self, feature):
        """
        预测结果
        Args:
            feature (array): 一个样本 (1, M)

        Returns:

        """
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.classify(f_input, clf_v, direct)

        return 1 if result > 0 else -1

    def score(self, X_test, Y_test):
        """
        计算分类正确率
        Args:
            X_test (array): 测试样本特征 (num_test_sample, M)
            Y_test (array): 测试样本标签 (num_test_sample, 1)

        Returns:

        """
        right_count = 0
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == Y_test[i]:
                right_count += 1

        return right_count / len(X_test)


def test(model, test_file_path):
    """
    测试
    :param test_file_path: 测试文件所在路径
    :return:
    """
    # 1.准备数据
    test_data, passenger_id = load_data(test_file_path, training=False)

    # 2. 归一化
    normal_test_data = normalize(test_data)

    # 3. 预测
    predictions = []
    for i in range(len(normal_test_data)):
        predictions.append(model.predict(normal_test_data[i]))

    # 4. 转换成符合要求的数，并保存结果
    predictions = pd.Series(predictions)  # 输入必须是1维数据，要拉平
    predictions = predictions.apply(lambda x: 1 if x >= 0 else 0)

    result = pd.DataFrame(
        {'PassengerId': passenger_id, 'Survived': predictions})
    result.to_csv('./my_submission.csv', index=False)


if __name__ == "__main__":
    data, label = load_data('../data/Titanic/train.csv')
    normal_data = normalize(data)
    model = AdaBoost(n_estimators=200, learning_rate=0.1)
    model.fit(normal_data, label)
    print(model.score(normal_data, label))
    test(model, '../data/Titanic/test.csv')

    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
    clf.fit(normal_data, label)
    print(clf.score(normal_data, label))
