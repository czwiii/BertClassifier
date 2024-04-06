# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from common import constants


# 训练数据初始化类（核心）
class CNewsDataset(Dataset):  # 标签（类别），文本，初始化加载训练和测试数据。
    def __init__(self, filename):
        # 数据集初始化
        self.labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']  # 设置十个类别
        self.labels_id = list(range(len(self.labels)))  # 把分类转成id
        # 加载路径下的tokenizer，决定了通过切字还是切词并生成以下参数
        self.tokenizer = BertTokenizer.from_pretrained(constants.BERT_PATH)
        # List<List<文本id>>，把每段文本每个字转换为词表（vocab.txt）中的id，并映射成embedding向量
        self.input_ids = []
        # List<List<句子位置id>>，标识每个字是文本AB句的顺序的集合的集合，对句子进行分类处理，每个字用0表示在第一句
        self.token_type_ids = []
        # List<List<padding_id>>，每个句子padding的最大的长度(1024)，超过的部分截断，不足的用[PAD]补齐，严格保持为一个矩阵的格式
        self.attention_mask = []
        self.label_id = []  # List<List<文本id>>，标签的id信息集合，如果是每个字都有一个标签，则是标签id的集合的集合
        self.load_data(filename)  # 初始化时，加载参数文件名指定的数据集

    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        # 读取文件
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        # 逐行读取，tqdm:迭代遍历的进度条可视化工具
        for line in tqdm(lines, ncols=100):
            label, text = line.strip().split('\t')  # strip:过滤回车,split:按'\t'进行分隔为标签和文本
            label_id = self.labels.index(label)  # 把标签转成id
            # 利用tokenizer对文本进行切字转化成token，指定token,padding,max_length:最大输入长度
            # tokenizer:模型的分词器,切分文本
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            # 记录文本id，np.array:把python的list封装为np类型的list,更加兼容pytorch
            self.input_ids.append(np.array(token['input_ids']))  # size=512的list,存放切字后的字的id值
            # 记录句子位置的标识，如果不涉及多句子，则都是0
            self.token_type_ids.append(np.array(token['token_type_ids']))  # size=512的list,区分文字在句子位置的标识,如果只有一句则值都为0
            # 记录位置padding，对句子进行mask,标注为哪个位置为padding(用1表示原始文本0表示填充),该位置的词不需要进行attention
            self.attention_mask.append(np.array(token['attention_mask']))  # size=512的list,存放字的padding值
            # 记录标签id，如果是NER任务，一个字对应一个标签，那么这里append就不是一个label_id，而是也是一个label_id集合
            self.label_id.append(label_id)  # 把label_id放到list中，一个文本/句子代表一个标签

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)
