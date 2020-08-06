## 思考题   
> MATCHSUM里面pearl-summary是什么？为什么要找到pearl-summary？        
+ 原文、候选摘要、标签，评估候选集中和原文或者标签在语义空间上是最近的，可认为是最好的摘要；
+ sentence-level extractors 将文章分成句子，然后根据得分高低，和gold summary进行计算得到得分Rouge
+ summary-level extractors 直接拿候选摘要计算
+ pearl-summary summary-level的得分高，但是sentence-level的得分低，这种类型的摘要
+ 两种抽取方式会漏掉一些可能是最佳的摘要
  

  
> 知识蒸馏里参数T（temperature）的意义？    
+ 知识蒸馏的目的是将一个高精度且笨重的teacher转换为一个更加紧凑的student；
+ 首先提高teacher模型softmax层的temperature参数获得一个合适的soft target集合；
+ 然后对要训练的student模型，使用同样的temperature参数值匹配teacher模型的soft target集合，作为student模型总目标函数的一部分，以诱导student模型的训练，实现知识的迁移；
+ 较大的T来训练模型，这时候复杂的神经网络能产生更均匀分布的软目标；
+ 相同T值的学习soft target,可以学习到数据的结构分布特征；
+ 实际应用中，将T值恢复到1，让类别概率偏向正确概率；
+ T值的使用能够将关键的分布信息从原有的数据中分离出来。


> TAPT（任务自适应预训练）是如何操作的？  
+ TAPT：Task-Adaptive Pretraining，指的就是在具体任务数据上继续预训练；
+ 使用了比较小的预训练语料库，假设训练集很好地代表了任务的各个方面；
+ 采取RoBERTa作为基准的预训练语言模型，
+ 在任务相关的无标注语料继续进行预训练，然后再对特定任务进行finetune

 
> 从模型优化的角度，在推理阶段，如何更改MATCHSUM的孪生网络结构？  
+ 构建摘要候选集的时候 Cmn个，可以优先剔除无意义的，计算之后肯定得分很低的部分
+ 两个loss函数，候选摘要之间的差异排名也可以做优化

 