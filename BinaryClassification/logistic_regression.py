import numpy as np
import pandas as pd
from load_data import  load_data, normalize

def  logistic_regrssion(data,label):
    """
    逻辑斯蒂回归，P(Y=1|x) = exp(wx+b)/(1+exp(wx+b))
    用极大似然估计来估计
    :param data: [n,num_features]
    :param label: [n,1]
    :return w,b: 返回参数
    """
    # 1. 初始化参数
    w = np.zeros([data.shape[1],1])
    b = 0

    # 2. 设置超参数
    lr = 0.001 # 学习率
    epochs = 500 # 训练轮数

    # 3. 训练
    for epoch in range(epochs):
        loss = 0
        for i, x in enumerate(data):
            # 计算P(Y=0|x)
            px = 1/ (1+np.exp(np.dot(x,w)+b))

            # 计算损失函数(负对数似然函数)
            # loss = -log(P(Y|X) = -(ylog(px) +(1-y)log(1-px))
            y = label[i]
            loss = - (y*np.log(px) + (1-y)*np.log(1-px))

            # 更新参数
            w += lr*(px-y)*x.reshape(-1,1)
            b += lr*(px-y)


        print("Epoch {0:d}, loss={1:.3f}".format(epoch, loss.item()))

    return w,b

def test(w,b,test_file_path):
    """
    测试
    :param w:
    :param b:
    :param test_file_path: 测试文件所在路径
    :return:
    """
    # 1.加载数据
    test_data, passenger_id = load_data(test_file_path,training=False)

    # 2. 归一化数据
    normal_test_data = normalize(test_data)

    # 3. 预测
    predictions = 1/(1+np.exp(np.dot(normal_test_data,w)+b))
    predictions = pd.Series(predictions.flatten())
    predictions = predictions.apply(lambda x: 1 if x>=0.5 else 0)

    result = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predictions})
    result.to_csv('my_submission_1.csv', index=False)

if __name__ == "__main__":
    data,label = load_data('./data/Titanic/train.csv')
    normal_data = normalize(data)
    # 需要将label还原成0，1
    label= list(map(lambda x:1 if x==1 else 0,label))
    w,b = logistic_regrssion(normal_data,label)
    test(w,b,'./data/Titanic/test.csv')

