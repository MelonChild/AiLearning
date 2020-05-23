#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from annoy import AnnoyIndex

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words = line.split(' ')

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words

def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1

        # 按照字典里的词频进行排序，出现次数多的排在前面
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)

        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    """
    vocab = [(w,i) for i,w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]
    return vocab,reverse_vocab

def timeit(f):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = f(*args, **kwargs)
        end_time = time.time()
        print("%s函数运行时间为：%.8f" %(f.__name__, end_time - start_time))
        return res
    return wrapper

def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))

if __name__ == '__main__':
    lines = read_data('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                      '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                      '{}/datasets/test_set.seg_x.txt'.format(BASE_DIR))
    vocab,reverse_vocab = build_vocab(lines)
    # print(vocab)
    save_word_dict(vocab, '{}/datasets/vocab.txt'.format(BASE_DIR))