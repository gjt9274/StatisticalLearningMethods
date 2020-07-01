import pandas as pd
import numpy as np

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
    # 只有测试数据存在缺失值
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
