"""
@File:    hmm
@Author:  GongJintao
@Create:  7/27/2020 11:08 AM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
import pandas as pd
import json
import pickle
import os

STATE = {'B', 'M', 'E', 'S'}


class HiddenMarkov:
    def __init__(self):
        # B: 词的开头字
        # M: 词的中间字
        # E: 词的结尾字
        # S: 单个字
        self.states = {}  # 状态标记
        self.init_vec = {}  # 初始化状态分布向量 len(state) x 1
        self.emit_mat = {}  # 发射矩阵，即每个状态到每个字的概率，len(state) x len(observe)
        self.trans_mat = {}  # 状态转移矩阵，即每个状态之间相互转移的概率，len(state) x len(state)

    # 初始化参数
    def _init_args(self):
        for state in self.states:
            # 1. 初始化 初始状态分布向量
            self.init_vec[state] = 0
            # 2. 初始化发射矩阵
            self.emit_mat[state] = {}
            # 3. 初始化状态转移矩阵
            self.trans_mat[state] = {}
            for target in self.states:
                self.trans_mat[state][target] = 0.0

    def train_epoch(self, observes, status):
        for i in range(len(status)):
            if i == 0:
                self.init_vec[status[i]] += 1  # 统计初始状态的频数
            else:
                self.trans_mat[status[i - 1]][status[i]] += 1

            if observes[i] not in self.emit_mat[status[i]]:
                self.emit_mat[status[i]][observes[i]] = 1
            else:
                self.emit_mat[status[i]][observes[i]] += 1

    def get_tags(self, word):
        if len(word) == 1:
            return 'S'
        else:
            return 'B' + (len(word) - 2) * 'M' + 'E'

    def get_prob(self):
        init_vec = {}
        trans_mat = {}
        emit_mat = {}
        for key in self.init_vec:
            if self.init_vec[key] == 0:  # ‘M','E'不可能出现在开头，所以为0，需要设置一个极小值
                init_vec[key] = -3.14e+100
            else:
                init_vec[key] = float(self.init_vec[key]) / \
                    sum(self.init_vec.values())

        for key1 in self.trans_mat:
            trans_mat[key1] = {}
            for key2 in self.trans_mat[key1]:
                if self.trans_mat[key1][key2] == 0:
                    trans_mat[key1][key2] = -3.14e+100
                else:
                    trans_mat[key1][key2] = float(
                        self.trans_mat[key1][key2]) / sum(self.trans_mat[key1].values())

        for key1 in self.emit_mat:
            emit_mat[key1] = {}
            for key2 in self.emit_mat[key1]:
                if self.emit_mat[key1][key2] == 0:
                    emit_mat[key1][key2] = -3.14e+100
                else:
                    emit_mat[key1][key2] = float(
                        self.emit_mat[key1][key2]) / sum(self.emit_mat[key1].values())

        return init_vec, trans_mat, emit_mat

    # 维特比算法
    def viterbi(self, sequence):
        init_vec, trans_mat, emit_mat = self.get_prob()
        tab = [{}]
        path = {}

        # 初始化
        for state in self.states:
            tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0])
            path[state] = [state]

        for t in range(1, len(sequence)):
            tab.append({})
            new_path = {}
            for state1 in self.states:
                items = []
                for state2 in self.states:
                    prob = tab[t - 1][state2] * trans_mat[state2].get(state1)
                    items.append((prob, state2))
                best = max(items)
                tab[t][state1] = best[0] * emit_mat[state1].get(sequence[t])
                new_path[state1] = path[best[1]] + [state1]
            path = new_path
        prob, state = max([(tab[len(sequence) - 1][state], state)
                           for state in self.states])
        return path[state]

    def cut_sent(self, src, tags):
        word_list = []
        start = 0
        started = False

        if len(tags) != len(src):
            return None

        if tags[0] not in {'S', 'B'}:
            if tags[1] in {'E', 'M'}:
                tags[0] = 'B'
            else:
                tags[0] = 'S'

        if tags[-1] not in {'S', 'E'}:  # 结尾不是 ‘S'或者'E’，说明结尾错误
            if tags[-2] in {'S', 'E'}:  # 如果倒数第二个已经是词结尾，说明最后一个是单独一个字，为’S‘标志
                tags[-1] = 'S'
            else:
                tags[-1] = 'E'  # 如果倒数第二个不是结尾标志，则最后一个是词结尾’E‘标志

        for i in range(len(tags)):
            if tags[i] == 'S':
                if started:
                    started = False
                    word_list.append(src[start:i])
                word_list.append(src[i])
            elif tags[i] == 'B':
                if started:
                    word_list.append(src[start:i])
                start = i
                started = True
            elif tags[i] == 'E':
                started = False
                word = src[start:i + 1]
                word_list.append(word)
            elif tags[i] == 'M':
                continue
        return word_list

    def save(self, file_name="hmm.json", code='json'):
        fw = open(file_name, 'w', encoding='utf-8')
        data = {
            "trans_mat": self.trans_mat,
            "emit_mat": self.emit_mat,
            "init_vec": self.init_vec
        }
        if code == "json":
            txt = json.dumps(data)
            txt = txt.encode('utf-8').decode('unicode-escape')
            fw.write(txt)
        elif code == "pickle":
            pickle.dump(data, fw)

    def load(self, file_name='hmm.json', code="json"):
        fr = open(file_name, 'r', encoding='utf-8')
        if code == "json":
            txt = fr.read()
            model = json.loads(txt)
        elif code == "pickle":
            model = pickle.load(fr)

        self.trans_mat = model["trans_mat"]
        self.emit_mat = model["emit_mat"]
        self.init_vec = model["init_vec"]


class HMMSegger(HiddenMarkov):
    def __init__(self):
        super(HMMSegger, self).__init__()
        self.states = STATE
        self.data = None

    def train(self):
        if os.path.exists("hmm.json"):
            self.load("hmm.json")
        else:
            if self.data is None:
                return "请加载数据"

            self._init_args()

            for line in self.data:
                line = line.strip()

                observes = ""
                for i in range(len(line)):
                    if line[i] == " ":
                        continue
                    observes += line[i]

                words = line.split()
                status = ""
                for word in words:
                    status += self.get_tags(word)

                assert len(observes) == len(status)
                self.train_epoch(observes, status)

            self.save("hmm.json")

    def loat_data(self, file_path):
        self.data = open(file_path, 'r', encoding='utf-8')

    def cut(self, sentence):
        try:
            tags = self.viterbi(sentence)
            return self.cut_sent(sentence, tags)
        except BaseException:
            return sentence


if __name__ == "__main__":
    file_path = "../data/PKU/pku_training.utf8"
    segger = HMMSegger()
    segger.loat_data(file_path)
    segger.train()
    res = segger.cut("武汉市长江大桥")
    print(res)
