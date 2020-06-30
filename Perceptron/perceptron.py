import numpy as np
import pandas as pd


"""
以泰坦尼克数据集为例，实现感知机算法
任务介绍：根据数据提供的乘客信息，预测乘客是否生存
训练集特征：
PassengerId: 乘客编号(number, 与预测无关，可以舍去)
Survived: 是否存活(bool number, 1:是, 2:否)
Pclass: 船舱等级(number, 1:1st, 2:2nd, 3:3rd)
Name: 姓名(str, 与预测无关，可以舍去)
Sex: 性别(str, male,female, 可以用1来表示male，0表示female)
Age: 年龄(number,存在少量缺失，可以考虑用均值来填充)
SibSp: 乘客在船上的兄弟姐妹/配偶人数(number)
Parch: 乘客在船上的父母、孩子人数(number)
Ticket: 船票号码(str)
Fare: 船票价格(number)
Cabin: 船舱号(str, 存在大量缺失，可以舍去该特征)
Embarked: 登船港口(str c=Cherbourg, q=Queenstown, s=Southampton， 可以考虑用数字(1,2,3)代替)
"""


def load_data(data_path,training=True):
    '''
    :param data_path: csv数据的路径
    :return:  返回(x,y)
    '''
    data = pd.read_csv(data_path)

    # 0. 可以先判断数据是否存在空值
    # train_data.isna().any()

    # 1. 选择用于训练的特征
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # 2. 处理缺失值
    # 'Age'
    # train_data['Age'].isnull().sum()  #计算Age这一列空值个数：177
    # 用均值来填充
    data['Age'].fillna(value=data['Age'].mean(), inplace=True)

    # 'Embarked'
    # train_data['Embarked'].isnull().sum() #计算Embarked这一列空值个数：2
    # 使用众数来填充
    data['Embarked'].fillna(
        value=data['Embarked'].mode(),
        inplace=True)

    # ‘Fare'
    # 只有测试数据有
    if not training:
        data['Fare'].fillna(value=data['Fare'].mean(),inplace=True)

    # 3. 处理非数值
    data['Sex'] = data['Sex'].apply(
        lambda x: 1 if x == 'male' else 0)
    data['Embarked'] = data['Embarked'].apply(
        lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))

    # 4. 选取特征
    x = data[features].values

    # 因为感知机的标签为 -1，1，所以需要转换0为-1
    # 需要区分测试集，如果是训练阶段，y为标签
    # 如果是测试阶段 ，y返回乘客编号
    if training:
        y = data['Survived'].apply(lambda x: -1 if x == 0 else 1)
        y = y.values.reshape(-1, 1)  # 为了避免出错，最好将其加一个维度
    else:
        y = data['PassengerId']

    return x, y


def normlize(x):
    """
    归一化特征空间，将数据转换成均值为0，标准差为1的数据
    :param x: [n,features_nums] 每一列代表一个特征
    :return: 返回按列归一化后的X
    """
    # 均值
    miu = np.mean(x, axis=0)

    # 标准差
    sigma = np.std(x, axis=0)

    normal_x = (x - miu)/sigma

    return normal_x


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
    # 1.准备数据
    test_data,passenger_id = load_data(test_file_path,training=False)

    # 2. 归一化
    normal_test_data = normlize(test_data)

    # 3. 预测
    predictions = np.dot(normal_test_data,w) + b #[n,1]

    # 4. 转换成符合要求的数，并保存结果
    predictions = pd.Series(predictions.flatten()) #输入必须是1维数据，要拉平
    predictions = predictions.apply(lambda x: 1 if x>=0 else 0)

    result = pd.DataFrame({'PassengerId':passenger_id,'Survived':predictions})
    result.to_csv('my_submission.csv',index=False)


if __name__ == "__main__":
    data,label = load_data('../data/Titanic/train.csv')
    normal_data = normlize(data)
    w,b = perceptron(normal_data,label)
    test(w,b,'../data/Titanic/test.csv')

