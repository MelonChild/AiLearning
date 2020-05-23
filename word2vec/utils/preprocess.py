#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from tokenizer import segment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOVE_WORDS = ['|','[',']','语音','图片',' ']


def parse_data(train_path,test_path):
    '''
    :param train_path: 训练集数据路径
    :param test_path: 测试集数据路径
    :return:
    '''
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    train_x = train_df.Question.str.cat(train_df.Dialogue)
    print('train_x is ', len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    # if 'Report' in train_df.columns:
    # train_y = train_df.Report
    # print('train_y is ', len(train_y))

    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    test_y = []
    train_x.to_csv('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)


def preprocess_sentence(sentence):
    '''

    :param sentence: 要预处理的句子
    :return:
    '''
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line

def remove_words(words_list):
    '''
    :param words_list: 所有词集
    :return: 清洗后的词集
    '''
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list

def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines

if __name__ == '__main__':

    # 转换训练数据
    parse_data('{}/datasets/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR))