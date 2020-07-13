"""
@File:    knn
@Author:  GongJintao
@Create:  7/13/2020 8:25 AM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
from load_data import load_data
from collections import Counter


class KNN:
    def __init__(self, X, Y, n_neighbors=3, p=2):
        """
        KNN
        Args:
            X (array): 训练样本 (N, M)
            Y (array): 训练样本标签 (N, 1)
            n_neighbors (int): 临近点个数
            p (int): 距离度量，默认 2 表示 2 范数
        """
        self.n = n_neighbors
        self.p = p
        self.X = X
        self.Y = Y

    def predict(self, x):
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(x - self.X[i], ord=self.p)
            knn_list.append((dist, self.Y[i]))

        for i in range(self.n, len(self.X)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(x - self.X[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.Y[i])

        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)  # 统计各个标签出现的次数
        # 按出现次数排序，然后取最多的那一个
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]

        return max_count

    def score(self, x_test, y_test):
        right_count = 0
        for x, y in zip(x_test, y_test):
            label = self.predict(x)
            if label == y:
                right_count += 1
        return right_count / len(x_test)


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data('../data/MNIST/')
    knn = KNN(train_images,train_labels)
    score = knn.score(test_images,test_labels)
    print(score)
