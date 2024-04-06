# coding: utf-8

import torch
import torch.nn as nn
from transformers import BertModel

# 模型实例类
# 总结：定义模型 》 定义分类器 》 softmax函数归化
# Bert
# Bert
class BertClassifier(nn.Module):
    # 模型必备的初始化方法
    def __init__(self, bert_config, num_labels):
        super().__init__()
        # 定义BERT模型，transformer提供了成熟的bert调用
        self.bert = BertModel(config=bert_config)  # 指定的超参数配置文件位置，这样相当于就拿到了一个bert
        # 定义分类器
        # 把Bert的输出做一个线性变换,bert_config.hidden_size:维度(512,768),num_labels:映射成多少维度的向量/类别
        # 不管模型输出是多少维度，统一通过线性变换的维度变到比如有十个分类，就变成有十个元素的向量
        # 再通过softmax（激活函数）做一个归化，最后哪个归化的概率最高就判定这条数据是哪个类别
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    # 分类器过程
    # input_ids:文本的id集合,attention_mask:位置padding集合,token_type_ids:句子顺序集合
    # 模型必备的功能方法，包含了一个模型实例内部的具体逻辑
    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT的输出
        # 分为两个部分，第一个元素是输入序列所有 token 的 Embedding 向量层，第二个变量是[CLS]位的隐层信息
        # [CLS]id[SEP]    [4 768]  [[1_CLS], [2_CLS], [], []]
        # [CLS]的意义：代表一段文本的第一个token，可以作为分类任务的依据，
        # [CLS]涉及了语义是否有偏的问题，在对token进行attention计算时，[CLS]不包含任务语义，如果使用具体的token作为输出第一个位置信息会混入语义信息
        # [CLS]在经过encoder处理后，它里面也包含了整段文本的语义信息
        # bert_output:last_hidden_state:最后一层隐层信息,pooler_output:最后一层每个句子的[CLS]位的数量和模型输出的维度(768)的隐层信息,适用于分类
        # 'last_hidden_state':每个句子(4)的每个字(512)的隐层信息(768维),适用于NER
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取[CLS]位置的pooled output  [4, 768]
        pooled = bert_output[1]
        # 分类  [CLS] [] [] []
        # [4, 512]  这是4句话的512个字的id
        # [4, 512, 768] 要把每个字映射成768维的向量
        # [4, 512, 768] 模型输出格式也是一致
        # [CLS] 我们只取第一个字的向量
        # [4, 768]  * [768, 10] = [4, 10] 矩阵变化
        # [CLS]没有语义信息，是没有任何含义的字符
        # 相当于取了四个[CLS]的768维向量信息，再通过线性变换把768变成10个类别
        logits = self.classifier(pooled)
        # 返回softmax后结果，把向量进行归一化，转换成每个类别的权重
        return torch.softmax(logits, dim=1)
