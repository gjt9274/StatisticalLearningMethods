"""
@File:    k_means
@Author:  GongJintao
@Create:  7/24/2020 8:35 AM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, datasets, k=3):
        self.datasets = datasets
        self.N = len(datasets)
        self.k = k
        self.belong_list = np.zeros(self.N)  # 聚类结果
        self.cluster_center = [[] for _ in range(self.k)]  # 聚类中心

    def cal_dis(self, x1, x2):
        return sum(np.square(x1 - x2))

    def init_center(self):
        """
        先随机选择一个点，然后找离其最远的点作为第二个中心点，依次类推
        """
        center = []
        c0 = np.random.randint(0,self.N)
        center.append(c0)
        max_dis = float(-np.inf)
        index = -1

        while len(center) < self.k:
            remainder_list = [i for i in range(self.N) if i not in center] #
            for i in remainder_list:
                dis = self.get_farthest(i,center)
                if dis > max_dis:
                    max_dis = dis
                    index = i
            center.append(index)
        return center




    def get_farthest(self,i,center_list):
        """计算第i个样本到当前聚类中心列表的最远距离"""
        x = self.datasets[i]
        max_dis = float(-np.inf)
        for j in center_list:
            dis = self.cal_dis(x,self.datasets[j])
            if dis > max_dis:
                max_dis = dis
        return max_dis




    def get_belongs(self, i):
        x = self.datasets[i]
        min_dis = float(np.inf)
        belong_index = -1

        for index in range(self.k):
            dis = self.cal_dis(x, self.cluster_center[index])
            if dis < min_dis:
                min_dis = dis
                belong_index = index
        return belong_index

    def fit(self):

        # 1. 随机初始化聚类中心
        center = self.init_center()
        self.cluster_center = self.datasets[center]
        # self.cluster_center = self.datasets[np.random.choice(self.N,3)]  #随机初始化聚类中心

        # 2. 计算每个样本到聚类中心的距离，将其划分到距离最近的类，直到满足条件
        while True:
            belong_new_list = np.zeros(self.N)
            # 3. 计算每个样本的类，并记录
            for i in range(self.N):
                belong_index = self.get_belongs(i)
                belong_new_list[i] = belong_index

            # 4. 如果聚类结果不再变化，则结束
            if (self.belong_list == belong_new_list).all():
                break
            # 5. 否则更新聚类中心
            self.belong_list = belong_new_list

            for index in range(self.k):
                self.cluster_center[index] = self.datasets[self.belong_list == index].mean(
                    axis=0)

        result = {}
        for i in range(self.k):
            result[i] = np.where(self.belong_list == i)[0]

        return result


if __name__ == "__main__":
    datasets = pd.read_csv('../data/iris/iris.csv')
    data = datasets.iloc[:, :-1].values
    # np.random.shuffle(data)
    model = KMeans(data)
    results = model.fit()

    fig = plt.figure(figsize=(10, 10))

    color = ['r', 'y', 'b']

    plt.subplot(121)
    for key, c in zip(results.keys(), color):
        plt.scatter(datasets.loc[results[key], 'Petal.Width'],
                    datasets.loc[results[key], 'Petal.Length'], c=c)

    plt.subplot(122)
    labels = set(datasets.iloc[:, -1])
    for label, c in zip(labels, color):
        plt.scatter(datasets.loc[datasets.iloc[:, -1] == label, 'Petal.Width'],
                    datasets.loc[datasets.iloc[:, -1] == label, 'Petal.Length'], c=c)

    plt.show()
