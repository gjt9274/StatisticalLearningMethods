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
from matplotlib import pyplot as plt
import time


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

    def get_attr(self):
        return self.dom_elt,self.split,self.left,self.right

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
        if not len(data_set): #递归边界，数据集为空返回None
            return None

        # 按要进行分割的那一维特征进行排序
        # data_set.sort(key = lambda x:x[split])
        sorted(data_set,key=lambda x:x[split])
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

class KnnBasedOnKdTree(object):
    def __init__(self,data,label,p=2):
        self.x = data
        self.y = label
        self.p = p
        self.kd_tree = KdTree(data)
        # 定义一个namedtuple，用来存储最近坐标点，最近距离，标签
        self.nearest = []

    def _search_leaf(self,stack,tree,target):
        travel_tree = tree
        while travel_tree:
            dom_elt,axes,lchile,rchild = travel_tree.get_attr()
            if target[axes] >= dom_elt[axes]:
                next_node = rchild
                next_direction = 'right' #记录哪个方向被访问过了
            else:
                next_node = lchile
                next_direction = 'left'

            stack.append((travel_tree,next_direction))
            travel_tree = next_node

    def _get_farthest(self):
        """获取k个点中最远的元素"""
        farthest = None
        for item  in self.nearest:
            if not farthest:
                farthest = item
            elif farthest[1] < item[1]:
                farthest = item
        return farthest


    def _check_nearest(self,current,target,k):
        d = np.linalg.norm(current-target,ord=self.p)
        l = len(self.nearest)

        if l < k: #说明还没有满k个点，继续添加
            self.nearest.append((current,d))
        else:
            farthest_d = self._get_farthest()[1]
            if farthest_d > d:
                # 将旧的最远点移除
                new_nearest = [item for item in self.nearest if item[1] < farthest_d]
                # 添加新的点
                new_nearest.append((current,d))
                self.nearest = new_nearest



    def find_nearest(self,target,k=1):
        tree = self.kd_tree.root
        stack = []
        self._search_leaf(stack,tree,target) #一直搜索直到叶子结点，并将路径压入栈
        while stack:
            node,next_direction = stack.pop() #弹出栈顶为当前最近点
            dom_elt,axes,lchild,rchild = node.get_attr()
            # 因为要添加k个临近点，所以需要检查当前点到目标点的距离
            self._check_nearest(dom_elt,target,k)
            if lchild and rchild is None: #当前结点为叶子结点，则回退
                continue
            #获取最远点距离，如果k=1，即获取当前最近点与目标点的距离
            farthest,distance = self._get_farthest()
            if abs(dom_elt[axes] - farthest[axes]) < distance: #判断是否与另一半区域相交
                if next_direction == 'right':
                    try_node = lchild
                else:
                    try_node = rchild
                self._search_leaf(stack,try_node,target)
        return self.nearest


if __name__ == "__main__":
    # start = time.time()
    # (train_images, train_labels), (test_images, test_labels) = load_data('../data/MNIST/')
    # knn = KNN(train_images,train_labels)
    # score = knn.score(test_images[:100],test_labels[:100])
    # end = time.time()
    # print(score)
    # print("time:{:.4f}s".format(end-start))

    data = np.array([[6.27, 5.50],
                     [1.24, -2.86],
                     [17.05, -12.79],
                     [-6.88, -5.40],
                     [-2.96, -0.50],
                     [7.75, -22.68],
                     [10.80, -5.03],
                     [-4.60, -10.55],
                     [-4.96, 12.61],
                     [1.75, 12.26],
                     [15.31, -13.16],
                     [7.83, 15.70],
                     [14.63, -0.35]])

    labels = np.array([1, 2, 3, 2, 2, 3, 3, 2, 1, 1, 3, 1, 3])

    knn = KnnBasedOnKdTree(data,labels)
    target = np.array([4,10])
    nearest = knn.find_nearest(target,1)
    print(nearest)
    plt.figure(dpi=100)  # 指定像素
    for i in range(len(labels)):
        if labels[i] == 1:
            color = 'c'
        elif labels[i] == 2:
            color = 'b'
        else:
            color = 'g'
        plt.scatter(data[i][0], data[i][1], c=color, s=30, alpha=0.8)

    plt.scatter(target[0], target[1], c='r', s=30, alpha=0.8)

    for point,_ in nearest:
        plt.plot([target[0], point[0]], [target[1], point[1]], 'r--')

    plt.show()



