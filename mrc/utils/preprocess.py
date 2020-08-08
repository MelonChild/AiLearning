#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 预处理原始数据

import numpy as np
from mrc.utils.segment import segment
import json
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 需要清除的无意义词汇
REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']

def remove_words(words_list):
    """
    关键词剔除
    :param words_list:
    :return:
    """
    words_list = [worditem for worditem in words_list if worditem not in REMOVE_WORDS]
    return words_list

def read_lines(path, col_sep=None):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if col_sep:
                if col_sep in line:
                    lines.append(line)
            else:
                lines.append(line)
    return lines

def extract_sentence(qpath, ppath):
    ret = []
    lines = read_lines(qpath)
    lines += read_lines(ppath)
    for line in lines:
        ret.append(line)
    return ret

def parse_data(path,qpath,ppath,apath):
    """
    数据转换
    :param path: 元数据路径
    :param qpath: 问题数据存储路径
    :param ppath: 文章数据存储路径
    :param apath: 所有数据存储路径
    :return:
    """
    with open(path,'rb') as fp:
        # print(fp.readline())

        # 清理数据
        remove_file(qpath)
        remove_file(ppath)
        remove_file(apath)

        maxline = 100
        for i in range(maxline):
            line = fp.readline()
            if line:
                data = json.loads(line)
                for document in data['documents']:
                    data =preprocess_document(document)
                    print("Q:",document['title'])
                    print(data['title'])
                    save_data(qpath,data['title'])
                    save_data(ppath,data['paragraph'])
            else:
                break

        #集合数据集
        sentences = extract_sentence(qpath, ppath)
        save_data(apath, sentences,False)

def save_data(qpath,data,line=True):
    """
    存储数据
    :param qpath: 数据存储路径
    :param data: 数据
    :return:
    """
    with open(qpath, 'a', encoding='utf-8') as f1:
        if line:
            f1.write('%s' % data)
            f1.write('\n')
        else:
            f1.write('%s' % '\n'.join(data))

def remove_file(path):
    os.path.exists(path) and os.remove(path)

def preprocess_document(document,is_segment=False):
    """
    分词处理句子
    :param document:
    :param is_segment: 是否直接分词
    :return:
    """
    segment_data={}

    # 问题分词
    if is_segment:
        seg_list = segment(document['title'].strip())
        seg_list = remove_words(seg_list)
    else:
        seg_list = document['segmented_title']
    segment_data['title'] = ' '.join(seg_list)

    # 段落分词
    if is_segment:
        par_list_temp=[]
        for paragraph in document['paragraphs']:
            par_list_temp.append(segment(paragraph.strip()))
    else:
        par_list_temp = document['segmented_paragraphs']


    par_list = []
    for temp in par_list_temp:
        par_list.extend(temp)
    segment_data['paragraph'] = ' '.join(par_list)

    return segment_data

def process(filenames=['search.demo.dev.json',
                      'title.search.demo.dev.txt',
                      'para.search.demo.dev.txt',
                      'all.search.demo.dev.txt']):
    """
    数据集预处理
    :return:
    """
    parse_data('{}/datasets/devsets/{}'.format(BASE_DIR,filenames[0]),
               '{}/datasets/devsets/{}'.format(BASE_DIR,filenames[1]),
               '{}/datasets/devsets/{}'.format(BASE_DIR,filenames[2]),
               '{}/datasets/devsets/{}'.format(BASE_DIR,filenames[3]))