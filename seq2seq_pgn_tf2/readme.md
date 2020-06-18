# Homework-week4
2020-06-06 OOV和Word-repetition问题的改进

## GitHub

    https://github.com/MelonChild/AiLearning    
    
    tag: T-Week-4

## 目录文件说明

+ datasets
  
  数据文件，训练数据，分词数据
  
+ encoders decoders

  rnn 编解码
  
+ utils

  工具集

+ models

  PGN

## 要求

### week4

- 在BahdanauAttentionCoverage类中完成与coverage相关的代码定义

- 完成Pointer类中Pgen系数的定义

- 在PGN model中补充代码，完成整个模型的搭建

- 在loss中完成_coverage_loss的代码

- 通过上述重要模型代码补全，可以顺利跑通PGN网络

- 测试代码不要调整可以参考使用

## 更新日志

### seq2seq_pgn_tf2

OOV和Word-repetition问题的改进

+ 2020-06-11

  创建项目，熟悉项目代码，添加注释

+ 2020-06-12

  补全代码，测试程序；
  回顾课程，复习实现原理及编程处理

+ 2020-06-13
 
  补全代码，测试更新
  
  Building the model ...
  0%|          | 0/20000 [00:00<?, ?it/s]Creating the vocab ...
max_size of vocab was specified as 30; we now have 30 words. Stopping reading.
Finished constructing vocabulary of 30 total words. Last word added: 故障
Creating the batcher ...
2020-06-13 17:29:21.443853: I tensorflow/core/platform/cpu_feature_guard.cc:145] This TensorFlow binary is optimized with Intel(R) MKL-DNN to use the following CPU instructions in performance critical operations:  SSE4.1 SSE4.2 AVX AVX2 FMA
To enable them in non-MKL-DNN operations, rebuild TensorFlow with the appropriate compiler flags.
2020-06-13 17:29:21.446092: I tensorflow/core/common_runtime/process_util.cc:115] Creating new thread pool with default inter op setting: 4. Tune using inter_op_parallelism_threads for best performance.
Creating the checkpoint manager
Model restored
  0%|          | 3/20000 [19:49<2261:21:47, 407.11s/it]

+ 2020-06-14
  
  完善代码，了解基础名词，知识点，重新梳理代码逻辑

