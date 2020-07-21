"""
@File:    decision_tree
@Author:  GongJintao
@Create:  7/19/2020 3:01 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import pandas as pd
import numpy as np
from collections import Counter
from load_data import load_data
import time


class Node:
    def __init__(self, leaf=True, feature_axis=None, label=None):
        self.leaf = leaf    #是否是叶子节点
        self.feature_axis = feature_axis    #当前结点所用的特征所在的列
        self.label = label    #当前结点的标签
        self.sub_tree = {}    #子树

    # 添加子结点
    def add_node(self, val, node):
        self.sub_tree[val] = node

    # 预测，如果当前结点时叶子结点直接返回类标签
    # 否则在其子结点中预测
    def predict(self, features):
        if self.leaf is True:
            return self.label
        return self.sub_tree[features[:,self.feature_axis].item()].predict(features)


class DecisionTree:
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon

    def cal_ent(self, datasets):
        """
        计算数据集的经验熵
        Args:
            datasets (array): 数据集 (n, m+1)，最后一列是标签

        Returns:
            ent (float): 数据集的经验熵
        """
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = - sum([(p / data_length) * np.log2(p / data_length)
                     for p in label_count.values()])

        return ent

    def cal_cond_ent(self, datasets, axis=0):
        """
        计算在某个特征下的经验条件熵
        Args:
            datasets (array): 数据集 (n, m+1) 最后一列是标签
            axis (int): 选取的特征所在的维度

        Returns:
            cond_ent (float): 经验条件熵

        """
        data_length = len(datasets)
        feature_sets = {}  # 用来存储当前特征，每个维度所含的样本
        for i in range(data_length):
            feature = datasets[i][axis]  # 当前特征的某个值
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.cal_ent(p) for p in
                        feature_sets.values()])
        return cond_ent

    def info_gain(self, ent, cond_ent):
        """信息增益"""
        return ent - cond_ent

    def get_max_info_gain(self, datasets):
        """
        求最大信息增益所在的特征
        Args:
            datasets (array): 数据集

        Returns:
            best (tuple): best[0]表示当前数据集，最大信息增益的特征的维度，best[1]表示最大信息增益
        """
        count = len(datasets[0])
        ent = self.cal_ent(datasets)  # 计算数据集的经验熵
        best_feature = []
        for c in range(count - 1):  # 最后一维是标签
            c_info_gain = self.info_gain(
                ent, self.cal_cond_ent(
                    datasets, axis=c))  # 计算每个特征下的信息增益
            best_feature.append((c, c_info_gain))  # 将特征所在维度，和信息增益添加进去
        # 求最大信息增益的特征
        best = max(best_feature, key=lambda x: x[-1])
        return best

    def create_tree(self, datasets):
        """
        创建决策树
        Args:
            datasets (array): 数据集 (n, m+1) 最后一列为标签
        Returns:
            tree : 决策树

        """
        features = datasets[:, :-1]  # 特征集
        labels = datasets[:, -1]  # 标签集

        # 1. 如果 D 中所有实例都属于同一类 C_k，则 T 为单节点树
        # 并将类 C_K 作为该节点的类标记，返回 T
        if len(np.unique(labels)) == 1:
            return Node(leaf=True, label=labels[0])  # 因为标签相同，所以随便选一个即可

        # 2. 如果特征集 A 为空，则 T 为单节点树，将 D 中实例数最大的类 C_k作为该结点的类标记，返回 T
        if len(features) == 0:
            return Node(
                leaf=True, label=Counter(
                    labels).most_common()[0][0])

        # 3. 计算最大信息增益
        # max_features_axis: 最大信息熵特征所在维度
        # max_info_gain:  该特征的最大信息熵
        max_feature_axis, max_info_gain = self.get_max_info_gain(datasets)

        # 4. 如果 A_g 的信息增益小于阈值，则 T 为单结点树
        # 并将 D 中实例数最大的类 C_k 作为该结点的类标签，返回 T
        if max_info_gain < self.epsilon:
            return Node(leaf=True,
                        label=Counter(labels.flatten()).most_common()[0][0])

        # 5. 构建 A_g 子集
        node_tree = Node(leaf=False,
                         feature_axis=max_feature_axis)

        # 6. 按特征 A_g 的取值将数据集进行划分
        feature_list = Counter(features[:, max_feature_axis]).keys()
        for f in feature_list:
            # 保留datasets中第max_featur_axis列满足条件的所有行
            sub_datasets = datasets[datasets[:, max_feature_axis] == f, :]
            # 删除当前最大信息增益特征所在的列
            sub_datasets = np.delete(sub_datasets, max_feature_axis, axis=1)

            # 递归生成树
            sub_tree = self.create_tree(sub_datasets)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self,train_data,train_labels):
        """
        对外训练接口
        Args:
            train_data (array): 训练数据
            train_labels (array): 训练标签

        Returns:

        """
        datasets = np.concatenate((train_data,train_labels.reshape(-1,1)),axis=1)
        self._tree = self.create_tree(datasets)
        return self._tree

    def predict(self,x):
        """
        预测函数
        Args:
            x (): 单个样本特征

        Returns:

        """
        return self._tree.predict(x)

    def score(self,test_data,test_labels):
        total_count = len(test_data)
        correct = 0
        for i in range(total_count):
            predict = self.predict(test_data[i])
            if predict == test_labels[i]:
                correct += 1
        return correct/total_count




if __name__ == "__main__":
    (train_images, train_labels), (test_images,
                                   test_labels) = load_data('../data/MNIST/')

    # 将数据二值化，并拉平
    train_images = (train_images > 0).astype('uint8').reshape(train_images.shape[0], -1)
    test_images = (test_images > 0).astype('uint8').reshape(test_images.shape[0], -1)

    # datasets = np.array([['青年', '否', '否', '一般', '否'],
    #                      ['青年', '否', '否', '好', '否'],
    #                      ['青年', '是', '否', '好', '是'],
    #                      ['青年', '是', '是', '一般', '是'],
    #                      ['青年', '否', '否', '一般', '否'],
    #                      ['中年', '否', '否', '一般', '否'],
    #                      ['中年', '否', '否', '好', '否'],
    #                      ['中年', '是', '是', '好', '是'],
    #                      ['中年', '否', '是', '非常好', '是'],
    #                      ['中年', '否', '是', '非常好', '是'],
    #                      ['老年', '否', '是', '非常好', '是'],
    #                      ['老年', '否', '是', '好', '是'],
    #                      ['老年', '是', '否', '好', '是'],
    #                      ['老年', '是', '否', '非常好', '是'],
    #                      ['老年', '否', '否', '一般', '否'],
    #                      ])

    model = DecisionTree()
    # decision_tree = model.fit(datasets[:,:-1],datasets[:,-1])
    # pred = model.predict(np.array(['老年', '否', '否', '一般']).reshape(1,-1))
    # print(pred)
    decision_tree = model.fit(train_images,train_labels)
    score = model.score(test_images,test_labels)
    print(score)
