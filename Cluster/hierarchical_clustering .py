"""
@File:    hierarchical_clustering
@Author:  GongJintao
@Create:  7/23/2020 8:21 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HierarchicalClustering :
    def __init__(self,num_categories=3):
        self.num_categories = num_categories
        self.results = {}

    def cal_dis(self, x1, x2):
        return sum(np.square(x1 - x2))

    def cal_culster_dis(self,culs1,culs2):
        min_dis = float(np.inf)
        for i in culs1:
            for j in culs2:
                if self.D[i][j] < min_dis:
                    min_dis = self.D[i][j]
        return min_dis

    def get_min_culster(self):
        min_dis = float(np.inf)
        min_index = (-1,-1)
        for i in self.results.keys():
            for j in self.results.keys():
                if i==j : continue
                dis = self.cal_culster_dis(self.results[i],self.results[j])
                if dis < min_dis:
                    min_dis = dis
                    min_index = (i,j)

        return min_index

    def fit(self,datasets):
        n = datasets.shape[0]
        self.D = np.array([[self.cal_dis(datasets[i],datasets[j]) for j in range(n)] for i in range(n)])

        for i in range(n):
            self.results[i] = []
            self.results[i].append(i)

        while len(self.results) > self.num_categories:
            index1,index2 = self.get_min_culster()
            self.results[index1].extend(self.results[index2])
            self.results.pop(index2)

        return self.results

if __name__ == "__main__":
    datasets = pd.read_csv('../data/iris/iris.csv')
    model = HierarchicalClustering()
    results = model.fit(datasets.iloc[:,:-1].values)

    fig = plt.figure(figsize=(10,10))

    color = ['r','y','b']


    plt.subplot(121)
    for key,c in zip(results.keys(),color):
        plt.scatter(datasets.loc[results[key],'Sepal.Width'],datasets.loc[results[key],'Sepal.Length'],c=c)

    plt.subplot(122)
    labels = set(datasets.iloc[:,-1])
    for label,c in zip(labels,color):
        plt.scatter(datasets.loc[datasets.iloc[:,-1]==label,'Sepal.Width'],datasets.loc[datasets.iloc[:,-1]==label,'Sepal.Length'],c=c)

    plt.show()

