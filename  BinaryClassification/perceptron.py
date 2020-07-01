from  load_data import load_data,normalize
import numpy as np
import pandas as pd

def perceptron(data, label):
    """
    感知机算法 y = w*x + b
    :param x: 输入特征 [n, features_nums]
    :param y: 标签 [n,1]
    :return:
    """
    # 1. 初始化权重
    w = np.zeros((data.shape[1], 1))  # [num_features,1]
    b = 0

    # 2. 初始化超参
    lr = 0.001  # 学习率
    epochs = 500  # 训练次数
    total_num = data.shape[0] #样本总个数
    # 3. 训练
    for epoch in range(epochs):
        count = 0 #用来记录每一轮误分类点的个数
        # 随机梯度下降法
        for i, x in enumerate(data):
            y = label[i]
            if y * (np.dot(x, w) + b) <= 0:
                w += (lr * y * x.T).reshape(-1,1) #(7,)->(7,1)
                b += lr * y
                count += 1 #误分类个数+1

        acc = (total_num-count) / total_num
        print("Epoch {0:d}, acc={1:.3f}".format(epoch,acc))

    return w,b

def test(w,b,test_file_path):
    """
    测试
    :param w:
    :param b:
    :param test_file_path: 测试文件所在路径
    :return:
    """
    # 1.准备数据
    test_data,passenger_id = load_data(test_file_path,training=False)

    # 2. 归一化
    normal_test_data = normalize(test_data)

    # 3. 预测
    predictions = np.dot(normal_test_data,w) + b #[n,1]

    # 4. 转换成符合要求的数，并保存结果
    predictions = pd.Series(predictions.flatten()) #输入必须是1维数据，要拉平
    predictions = predictions.apply(lambda x: 1 if x>=0 else 0)

    result = pd.DataFrame({'PassengerId':passenger_id,'Survived':predictions})
    result.to_csv('my_submission.csv',index=False)


if __name__ == "__main__":
    data,label = load_data('./data/Titanic/train.csv')
    normal_data = normalize(data)
    w,b = perceptron(normal_data,label)
    test(w, b, './data/Titanic/test.csv')

