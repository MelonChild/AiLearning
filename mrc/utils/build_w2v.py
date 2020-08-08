#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 构建词向量

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from mrc.utils.data_utils import dump_pkl
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build(out_path=None, sentence_path='',
          w2v_bin_path="w2v.bin", min_count=1):
    print('train w2v model...')
    # train model
    w2v = Word2Vec(sg=1, sentences=LineSentence(sentence_path),
                   size=256, window=5, min_count=min_count, iter=40)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print("save %s ok." % w2v_bin_path)

    # load model
    model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    dump_pkl(word_dict, out_path, overwrite=True)

def w2v():
    """
    构建词向量
    :return:
    """
    build('{}/datasets/devsets/word2vec.search.demo.dev.txt'.format(BASE_DIR),
          '{}/datasets/devsets/all.search.demo.dev.txt'.format(BASE_DIR),
          '{}/datasets/devsets/w2v.search.demo.dev.bin'.format(BASE_DIR),)