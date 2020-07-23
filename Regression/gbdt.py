"""
@File:    gbdt
@Author:  GongJintao
@Create:  7/22/2020 4:27 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""
import numpy as np
import pandas as pd
from  load_data import load_data

class Node:
    def __init__(self,feature_index=None,split_value=None,leaf=False,c=0.0):
        self.leaf = leaf  # 判断是否是叶子结点，即最终的划分
        self.feature = feature_index # 最佳切分特征所
        self.value = split_value # 最佳切分点
        self.left_node = None # 左节点
        self.right_node = None # 右节点
        self.c = c # 当前划分区域的系数

    def predict(self,x):
        """
        预测函数
        Args:
            x (): 单个特征

        Returns:

        """
        if self.leaf is True:
            return self.c
        elif x[self.feature] <= self.value:
            return self.left_node.predict(x)
        elif x[self.feature] > self.value:
            return self.right_node.predict(x)


class GBDT:
    def __init__(self,n_estimators=50,alpha=1.0):
        self.tree_nums = n_estimators    # 回归树的个数
        self.alpha = alpha    # 为了防止过拟合，每颗回归树的惩罚系数
        self.tree_list = [] # 回归树列表


    def get_best_partition(self,datasets):
        n,m = datasets.shape

        min_loss = float(np.inf)
        best_feature_index = -1
        best_split_value = -1

        for feature_index in range(m-1):
            for split_val in set(datasets.iloc[:,feature_index]):
                left_datasets = datasets.loc[datasets.iloc[:,feature_index] <= split_val]
                right_datasets = datasets.loc[datasets.iloc[:,feature_index] > split_val]

                y_left = left_datasets.iloc[:,-1]
                y_right = right_datasets.iloc[:,-1]

                c_left = y_left.mean()
                c_right = y_right.mean()

                loss = sum(np.square(y_left-c_left)) + sum(np.square(y_right-c_right))
                if loss < min_loss:
                    min_loss=loss
                    best_feature_index = feature_index
                    best_split_value = split_val

        return best_feature_index,best_split_value



    def create_cart_tree(self,datasets,depth,max_depth):
        """
        创建回归树，基于CART算法
        Args:
            datasets (array): 数据集
            depth (): 深度
            max_depth ():  最大深度

        Returns:
            tree (): 回归树

        """

        if len(datasets)< 2 or depth ==max_depth:
            return Node(leaf=True,c=datasets.iloc[:,-1].mean())

        feature_axis,split_value = self.get_best_partition(datasets)
        current_node = Node(feature_axis,split_value)

        # 划分数据集
        left_datasets = datasets.loc[datasets.iloc[:,feature_axis] <= split_value]
        right_datasets = datasets.loc[datasets.iloc[:,feature_axis] > split_value]

        current_node.left_node = self.create_cart_tree(left_datasets,depth+1,max_depth)
        current_node.right_node = self.create_cart_tree(right_datasets,depth+1,max_depth)

        return current_node

    def fit(self,datasets,max_depth=2):
        """
        梯度提升过程
        Args:
            datasets (): 数据集

        Returns:

        """

        # 1. 初始化 f_0
        f_0 = Node(leaf=True,c=datasets.iloc[:,-1].mean())
        self.tree_list.append(f_0)


        # 2. 梯度提升
        for i in range(1,self.tree_nums+1):
            # 对每个样本求残差
            for j in range(len(datasets)):
                datasets.iloc[j,-1] = datasets.iloc[j,-1] - self.alpha * self.tree_list[-1].predict(datasets.iloc[j,:-1])

            f = self.create_cart_tree(datasets,depth=0,max_depth=max_depth)
            self.tree_list.append(f)


    def predict(self,x):
        pred = 0.0
        for tree in self.tree_list:
            pred += tree.predict(x)
        return pred

    def score(self,test_data,test_label):
        total_num = len(test_data)
        correct = 0
        for i in range(total_num):
            pred = self.predict(test_data.iloc[i])
            # 如果预测值和真实值相差不超过20%，则表示正确
            if (abs(pred-test_label[i])/test_label[i]) < 0.2:
                correct += 1
        return correct/total_num

if __name__ == "__main__":
    data_path = '../data/boston_housing/housing.csv'
    train_data,test_data = load_data(data_path)

    model = GBDT()
    model.fit(train_data)
    score = model.score(test_data.iloc[:,:-1],test_data.iloc[:,-1])
    print(score)






