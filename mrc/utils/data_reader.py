#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_data(qpath, ppath):
    """
    读取数据，组合数据
    :param qpath:
    :param ppath:
    :return:
    """
    with open(qpath, 'r', encoding='utf-8') as f1, \
            open(ppath, 'r', encoding='utf-8') as f2:
        words = []
        for line in f1:
            words = line.split()

        for line in f2:
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
        # 统计词频
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
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

    vocab = [(w, i) for i, w in enumerate(result)]
    reverse_vocab = [(i, w) for i, w in enumerate(result)]

    return vocab, reverse_vocab,dic

def save_data(data, path):
    """
    保存数据
    :param data:
    :param path:
    :return:
    """
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            w, i = line
            f.write("%s\t%s\n" % (w, i))

def reader(filenames=['title.search.demo.dev.txt',
                      'para.search.demo.dev.txt',
                      'search.demo.dev.txt']):
    """
    构建词频词典
    :return:
    """
    lines = read_data('{}/datasets/devsets/{}'.format(BASE_DIR,filenames[0]),
                      '{}/datasets/devsets/{}'.format(BASE_DIR,filenames[1]))
    vocab, reverse_vocab,words = build_vocab(lines)

    save_data(vocab, '{}/datasets/devsets/vocab.{}'.format(BASE_DIR,filenames[2]))
    save_data(reverse_vocab, '{}/datasets/devsets/re-vocab.{}'.format(BASE_DIR,filenames[2]))
    save_data(words, '{}/datasets/devsets/sort-vocab.{}'.format(BASE_DIR,filenames[2]))