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
from math import sqrt
from collections import Counter,namedtuple


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

class KdNode(object):
    def __init__(self,dom_elt,split,left,right):
        """
        创建KD树的节点
        Args:
            dom_elt (): k 维向量节点，即 k 维空间中的一个样本点
            split (): 进行分割的维度所在的序号
            left ():  左子树
            right (): 右子树
        """
        self.dom_elt = dom_elt
        self.split = split
        self.left = left
        self.right = right

class KdTree(object):
    def __init__(self,data):
        self.k = len(data[0]) #特征维度
        self.root = self._build_tree(0,data)

    def _build_tree(self,split,data_set):
        """
        构建kd树
        Args:
            split (int): 分割维度所在的序号
            data_set (array): 数据集 (N, k)

        Returns:

        """
        if not data_set: #递归边界，数据集为空返回None
            return None

        # 按要进行分割的那一维特征进行排序
        data_set.sort(key = lambda x:x[split])
        split_pos = len(data_set) // 2
        median = data_set[split_pos] #中位数分割点
        split_next = (split+1) % self.k #更新维度

        # 递归创建KD树
        return KdNode(
            median,
            split,
            self._build_tree(split_next,data_set[:split_pos]), #左子树
            self._build_tree(split_next,data_set[split_pos+1:]) #右子树
        )

#定义一个namedtuple，用来存储最近坐标点，最近距离，访问过的节点数
result = namedtuple("Result_tuple",
                    "nearest_point nearest_dist nodes_visited")

def find_nearest(tree,point):
    """
    找到样本点point的最近临近点
    Args:
        tree (KdTree): kd树
        point (array): 样本点 (1,k)

    Returns:

    """
    k = len(point) #样本维度

    def travel(kd_node,target,max_dist):
        if kd_node is None:
            return result([0]*k, float("inf"),0)

        nodes_visited = 1

        s = kd_node.split # 获取分割的维度
        pivot = kd_node.dom_elt # 进行分割的轴

        if target[s] <= pivot[s]: #如果目标点第 s 维小于分割点的对应维度值，说明目标点离左子树更近
            nearer_node = kd_node.left
            further_node = kd_node.right
        else:
            nearer_node = kd_node.right
            further_node = kd_node.left

        temp1 = travel(nearer_node,target,max_dist)

        nearest = temp1.nearest_point #以此叶子节点为”当前最近点“
        dist = temp1.nearest_dist #更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist # 最近点在以目标点为球心，目标点到“当前最近点”的距离为半径的超球体内

        temp_dist = abs(pivot[s] - target[s]) # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist: #判断超球体是否与超平面 相交
            return result(nearest,dist,nodes_visited) #不相交则直接返回

        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1-p2)**2 for p1,p2 in zip(pivot,target)))

        if temp_dist < dist: #如果更近
            nearest = pivot #更新最近点
            dist = temp_dist #更新最近距离
            max_dist = dist #更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node,target,max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist: #如果另一个子结点内存在更近的点
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return result(nearest,dist,nodes_visited)

    return travel(tree.root,point,float("inf"))


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data('../data/MNIST/')
    knn = KNN(train_images,train_labels)
    score = knn.score(test_images[:100],test_labels[:100])
    print(score)


