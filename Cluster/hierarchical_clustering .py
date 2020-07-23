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


class HierarchicalClustering :
    def __init__(self, datasets, num_categories=3):
        self.datesets = datasets
        self.num_categories = num_categories

    def cal_dis(self, x1, x2):
        return sum(np.square(x1 - x2))

    def cal_culster_dis(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def score(self):
        pass