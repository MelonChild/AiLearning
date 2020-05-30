# Homework-week1
2020-05-23 基于Seq2Seq架构的模型搭建

## GitHub

    https://github.com/MelonChild/AiLearning    
    
    tag: T-Week-2

## 目录文件说明

+ datasets
  
  数据文件，训练数据，分词数据
  
+ encoders decoders

  rnn 编解码
  
+ utils

  工具集

## 要求

1. 补全rnn_encoder.py，完成Encoder的结构

2. 补全rnn_decoder.py，完成Attention和Decoder的结构

3. 完成sequence_to_sequence.py

4. python ./seq2seq_tf2/bin/main.py看看能不能把整个模型的训练跑通哟！

## 更新日志

### seq2seq_tf2

Week2-基于Seq2Seq架构的模型搭建

+ 2020-05-25

  创建项目，更新数据集，跑通代码。

+ 2020-05-26

  查看tensorflow 资料，熟练基本用法

+ 2020-05-28
 
  完善补全代码，测试更新
  修改部分数据训练
  
        max_size of vocab was specified as 300; we now have 300 words. Stopping reading.
        Finished constructing vocabulary of 300 total words. Last word added: 问
        true vocab is  <seq2seq_tf2.batcher.Vocab object at 0x135fd40b8>
        Creating the batcher ...
        2020-05-28 23:12:53.793799: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
        2020-05-28 23:12:53.814585: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f80e7852c80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
        2020-05-28 23:12:53.814612: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
        Building the model ...
        Creating the checkpoint manager
        Initializing from scratch.
        Starting the training ...
        Epoch 1 Batch 100 Loss 7.8928
        Epoch 1 Batch 200 Loss 7.5014

+ 2020-05-29

  使用华为云GPU,调试代码