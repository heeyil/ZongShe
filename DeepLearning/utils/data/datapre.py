import pandas as pd
import numpy as np
import re
import collections


# ont_hot编码，用于处理离散特征
def one_hot(targets, n_classes):
    return np.eye(n_classes, dtype=np.float32)[np.array(targets).reshape(-1)]


'''
特征缩放
归一化和标准化都是特征缩放的手段
'''


class feature_scaling:
    def __init__(self, datas):
        self.datas = datas
        # 检查变量类型是否为dataframe，不是则转换
        if isinstance(self.datas, pd.DataFrame):
            pass
        else:
            self.datas = pd.DataFrame(self.datas)
        self.numeric_features = datas.dtypes[datas.dtypes != 'object'].index

    # 数据标准化
    # 将特征缩放到均值为0、方差为1
    def standardizing(self):
        self.datas[self.numeric_features] = self.datas[self.numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        return self.datas

    # 数据归一化
    # 使用最小-最大缩放
    def normalization(self):
        self.datas[self.numeric_features] = self.datas[self.numeric_features].apply(
            lambda x: (x - x.min()) / (x.max() - x.min()))

    def fillnan(self, pad_value=None, method=None, axis=None):
        self.datas = self.datas.fillna(value=pad_value, method=method, axis=axis)

    def re(self):
        return self.datas


'''
自然语言处理中常用的文本清洗手段
对文本序列的一系列处理
'''


def clean_text(text):
    # 去除特殊字符和标点符号
    text = re.sub(r"[^a-zA-Z]", ' ', text)
    # 将文本转换为小写
    text = text.lower()
    # 去除多余的空格
    text = re.sub(r"\s+", " ", text)

    return text


# 根据空格分词，得到词表
def tokenizer(text):
    return [st.split() for st in text]


# 去除停用词
def clear_stop_words(text, stop_words):
    # 用户自定义停用词表并作为参数传入函数
    filtered_tokens = [word for word in text if word in stop_words]
    return filtered_tokens


# 总的文本预处理函数，包含了上述单独的函数，功能有待加强
def create_corpus(text, stop_words):
    text = [clean_text(st) for st in text]
    text = tokenizer(text)
    text = clear_stop_words(text, stop_words)

    return text


'''
在NLP中要将每一个词元(token)与其索引对应(idx)
经过预处理后还存在着缺失项，需要补齐
为了做词嵌入，还需要将每一行截断成相同的长度
'''


def set_word2id(text, pad, pad_value):
    # 保留在数据集中至少出现5次的词
    counter = collections.Counter([tk for st in text for tk in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

    # 获得初始的word2id
    idx_to_token = [tk for tk, _ in counter.items()]
    word2id1 = {tk: idx for idx, tk in enumerate(idx_to_token, 1)}

    # 补齐文本
    # 补齐所用单词
    PAD_TEXT = pad
    # 补齐所用值
    PAD_INDEX = pad_value

    # 用所有行长度的均值作为截断长度
    num = [len(st) for st in text]
    seq_len = int(np.mean(num))

    # 放入补齐单词
    word2id1[PAD_TEXT] = PAD_INDEX
    word2id2 = [[word2id1[tk] for tk in st if tk in word2id1] for st in text]

    # 得到经过补齐和截断后的列表，各元素为该token第一次出现在text中的index
    word2id = [st[:seq_len] + [PAD_INDEX] * (seq_len - len(st)) for st in word2id2]

    return word2id