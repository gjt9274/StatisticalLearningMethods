"""
@File:    naive_bayes
@Author:  GongJintao
@Create:  7/14/2020 6:18 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from load_data import load_data
import time

class NaiveBayes:
    def __init__(self, data, label, lamda=1.0):
        self.data = data
        self.X = (data > 0).astype('uint8').reshape(data.shape[0], -1)
        self.Y = label
        self.lamda = lamda
        self.class_num = 10
        self.feature_num = len(self.X[0])
        self.total_num = len(data)
        self.prior_prob, self.conditional_prob= self._calculate_prob(self.X, self.Y, self.lamda)

    def _calculate_prob(self, x, y, lamda):
        """
        计算先验概率
        Args:
            x (array):  样本特征 (n, 28x28)
            y (array): 样本标签数组 (n, 1)
            lamda (float): 拉普拉斯平滑系数

        Returns:
            prior_prob (array): 先验概率 (class_num(=10), 1)
            conditional_prob (array): 条件概率 (class_num, feature_num),因为图片已经二值化，此处返回的是像素值为1的条件概率，求 0 的只需 1减去该值即可
        """
        # 计算先验概率
        counter = Counter(sorted(y))
        prior_prob = np.array(
            [(counter[i] + lamda) / (self.total_num + self.class_num * lamda) for i in range(self.class_num)])

        # 计算条件概率
        conditional_num = np.zeros((self.class_num, self.feature_num))
        conditional_prob = np.zeros((self.class_num, self.feature_num))
        for i in range(self.class_num):
            conditional_num[i] = x[y.reshape(-1) == i].sum(axis=0)
            # 因为二值化的图片像素值只有0，1，所以是 2 x lamda
            conditional_prob[i] = (
                conditional_num[i] + lamda) / (counter[i] + 2 * lamda)

        return prior_prob,conditional_prob

    def show(self, binarization=False):
        image = self.data

        if binarization:
            image = (self.data > 0).astype('uint8')

        fig = plt.figure(figsize=(28, 28))
        for i in range(9):
            ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
            ax.imshow(image[i], cmap='gray')
            ax.set_title(str(self.Y[i]))

        plt.show()

    def predict(self, x_test):
        x_test = (x_test > 0).astype('uint8').reshape(-1)
        prob = np.zeros(10)
        for i in range(self.class_num):
            temp =  sum([np.log(self.conditional_prob[i][x]) if x_test[x] == 1 else np.log(1-self.conditional_prob[i][x]) for x in range(len(x_test))])
            prob[i] = np.log(self.prior_prob[i]) + temp

        return np.argmax(prob)

    def score(self,test_inputs,test_labels):
        total_num = len(test_inputs)
        correct = 0
        for i in range(total_num):
            pred = self.predict(test_inputs[i])
            if pred == test_labels[i]:
                correct += 1

        return correct/total_num


if __name__ == "__main__":
    start = time.time()
    (train_images, train_labels), (test_images,
                                   test_labels)=load_data('../data/MNIST/')
    model=NaiveBayes(train_images, train_labels)
    # model.show(binarization=True)
    score = model.score(test_images[:100],test_labels[:100])
    end = time.time()
    print(score)
    print("time:{:.4f}s".format(end-start))