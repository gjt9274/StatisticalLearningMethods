"""
@File:    hmm
@Author:  GongJintao
@Create:  7/29/2020 9:25 PM
@Software:Pycharm
@Blog:    https://gongjintao.com
@Email:   gjt9274@gmail.com
"""

import numpy as np
import json
import pickle
import os

STATES = {'B', 'M', 'E', 'S'}
EPS = -3.14e+100


class HiddenMarkov:
    def __init__(self):
        self.init_vec = {}  # 初始化状态概率  N(len(state))
        self.trans_mat = {}  # 状态转移矩阵，N X N
        self.emit_mat = {}  # 发射矩阵 ， N X M
        self.states = {}

    def _init_args(self):
        """初始化参数"""
        for state in self.states:
            self.init_vec[state] = 0
            self.emit_mat[state] = {}
            self.trans_mat[state] = {}
            for target in self.states:
                self.trans_mat[state][target] = 0

    def train_epoch(self, observers, status):
        """
        根据每个句子来统计信息
        Args:
            observers (str): 观测序列，即句子
            status (str): 状态序列，与状态序列等长

        Returns:

        """
        for i in range(len(status)):
            if i == 0:  # 即该状态序列的开头状态
                self.init_vec[status[i]] += 1
            else:
                # 转移矩阵，前一个状态和后一个状态的统计频数
                self.trans_mat[status[i - 1]][status[i]] += 1

            if observers[i] not in self.emit_mat[status[i]]:
                self.emit_mat[status[i]][observers[i]] = 0
            else:
                self.emit_mat[status[i]][observers[i]] += 1

    def get_prob(self):
        """
        得到概率化后的三个参数
        """
        init_vec = {}
        trans_mat = {}
        emit_mat = {}

        for key in self.init_vec:
            if self.init_vec[key] == 0:
                init_vec[key] = -3.14e+100
            else:
                # 为了防止句子过长，概率相乘，值下溢，用log来计算
                init_vec[key] = np.log(
                    float(self.init_vec[key]) / sum(self.init_vec.values()))

        for key1 in self.trans_mat:
            total = sum(self.trans_mat[key1].values())
            trans_mat[key1] = {}
            for key2 in self.trans_mat[key1]:
                if self.trans_mat[key1][key2] == 0:
                    trans_mat[key1][key2] = -3.14e+100
                else:
                    trans_mat[key1][key2] = np.log(
                        float(self.trans_mat[key1][key2]) / total)

        for key1 in self.emit_mat:
            total = sum(self.emit_mat[key1].values())
            emit_mat[key1] = {}
            for key2 in self.emit_mat[key1]:
                if self.emit_mat[key1][key2] == 0:
                    emit_mat[key1][key2] = -3.14e+100
                else:
                    emit_mat[key1][key2] = np.log(
                        float(self.emit_mat[key1][key2]) / total)

        return init_vec, trans_mat, emit_mat

    def get_tag(self, word):
        """得到每个词语的状态序列"""
        if len(word) == 1:
            return 'S'  # 说明是单个字
        else:
            return 'B' + (len(word) - 2) * 'M' + 'E'

    def viterbi(self, sentence):
        init_vec, trans_mat, emit_mat = self.get_prob()

        delta = [{}]
        psi = [{}]

        # 初始化
        for state in self.states:
            delta[0][state] = init_vec[state] + \
                emit_mat[state].get(sentence[0], EPS) #如果出现训练集中没有见过的词，则返回一个极小值
            psi[0][state] = 0

        for t in range(1, len(sentence)):
            delta.append({})
            psi.append({})
            for state1 in self.states:
                item = []
                for state2 in self.states:
                    prob = delta[t - 1][state2] + trans_mat[state2][state1]
                    item.append((prob, state2))
                best = max(item)
                delta[t][state1] = best[0] + \
                    emit_mat[state1].get(sentence[t], EPS)
                psi[t][state1] = best[1]

        # 回溯最佳路径
        path = []
        path.append(max(delta[-1],key=delta[-1].get))

        for t in range(len(psi)-2,-1,-1):
            path.append(psi[t+1][path[-1]])  # 回溯最佳路径

        return path[::-1]

    def cut_sent(self, src, tags):
        word_list = []
        start = 0
        started = False

        if len(tags) != len(src):
            return None

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
        model = {}
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
        self.states = STATES
        self.data = None

    def train(self):
        if os.path.exists("hmm.json"):
            self.load("hmm.json")
        else:
            if self.data is None:
                return "请加载数据"

            self._init_args()

            for line in self.data:
                words = line.strip().split()

                observes = ""
                status = ""
                for word in words:
                    observes += word
                    status += self.get_tag(word)

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

def test(segger,test_file):
    with open(test_file,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            print(segger.cut(line.strip()))

if __name__ == "__main__":
    file_path = "../data/PKU/pku_training.utf8"
    segger = HMMSegger()
    segger.loat_data(file_path)
    segger.train()
    sentence = "隐马尔可夫模型进行分词任务"
    print(segger.cut(sentence)) #['隐马尔可夫', '模型', '进行', '分词', '任务']
    # test(segger,"../data/PKU/pku_test.utf8")

