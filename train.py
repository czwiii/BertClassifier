# coding: utf-8

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
from sklearn import metrics
from common import constants


# 基础训练过程总结：
# 前置准备：配置超参数和模型参数》设置设备》读取训练数据集和评估数据集》把数据封装到DataLoader进行sampler切分batch》加载模型》初始化模型》定义优化器和损失函数
# 训练过程：迭代epoch》迭代batch》把数据输入模型训练》计算损失》计算准确率》反向传播》更新参数》使用验证集（模型不认识的数据集）进行评估》选出最优模型
# 数据加载过程：文本转换id_list 》 类别转换label_list
# 执行训练类
def main():
    # ----------------获取参数设置----------------
    batch_size = 4
    epochs = 10  # 数据全量训练次数
    learning_rate = 5e-6  # 学习率

    device = 'cpu'

    # ----------------数据读取与封装----------------
    # 获取到dataset，数据加载
    train_dataset = CNewsDataset('data/cnew.train_debug.txt')
    valid_dataset = CNewsDataset('data/cnew.val_debug.txt')

    # 封装DataLoader，用于生成Batch，dataset=数据集，batch_size=指定大小，shuffle=是否打乱数据
    # DataLoader:对每一个epoch先自动进行sampler,把完整的数据集切分成设置的batch_size，再一个一个进行读取
    # train_dataloader:batch_sampler:采样器,按照batch_size的大小分割数据，4000条数据batch_size是4batch_sampler是1000
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # ----------------模型加载----------------
    # 读取BERT的配置文件，模型目录里必须指定config.json并且不建议修改，否则会报错
    bert_config = BertConfig.from_pretrained(constants.BERT_PATH)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    # num_labels:Bert输出的768维的数据转成多少维度，多少个label就转成多少维
    model = BertClassifier(bert_config, num_labels).to(device)

    # ----------------优化器与损失函数配置----------------
    # 优化器，根据损失计算梯度，然后更新参数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数，交叉熵损失（分类任务时一般使用）
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0  # 只保存最好的模型，每迭代一个epoch后衡量当前模型对于测试集的fi是多少，f1是均衡衡量精准率和召回率，浮点数值越大模型越好

    # ----------------模型训练过程----------------
    for epoch in range(1, epochs + 1):  # 每个epoch全量数据循环一次
        losses = 0  # 当前整个epoch总的损失之和，一个epoch对应多个batch，每个batch会计算一次损失再进行反向传播更新参数
        accuracy = 0  # 当前整个epoch的精确率之和

        model.train()  # 模型的启动模式，train:把模型设置到训练状态，后面还需要再设置成评估状态
        train_bar = tqdm(train_dataloader, ncols=100)  # 迭代每个batch，可视化当前epoch迭代到第几个batch
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:  # 当前batch的数据
            # 梯度清零
            # 手动清零，优点是模型太大batch_size上不去，因为size越大占用显存越大
            # 多个batch进行梯度清零，就是多个batch_size进行反向传播，变相的扩大batch_size，提高灵活度
            # 控制多少个batch进行一次反向传播更新参数
            # 正常每次清零一次
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)  # 打印描述信息

            # output:一个十维（十个类别长度的集合）的向量，每个维度的概率代表是该类别的概率
            # 传入数据，调用model.forward()  [4, 10]
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()  # 累计所有损失

            # output [0.1, 0.05, ..., 0.23]
            pred_labels = torch.argmax(output, dim=1)  # 返回集合中数值最大的label
            # 通过预测的标签是否等于传入的真实标签，计算当前batch的精确率
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)
            accuracy += acc

            loss.backward()  # 进行损失的反向传播,计算梯度
            optimizer.step()  # 执行优化器的优化，实施梯度的更新
            train_bar.set_postfix(loss=loss.item(), acc=acc)  # 设置打印信息的值

        average_loss = losses / len(train_dataloader)  # 计算一个epoch的损失
        average_acc = accuracy / len(train_dataloader)  # 计算一个epoch的精确率

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # ----------------模型验证过程----------------
        model.eval()  # 每个epoch执行完之后，把model切换成评估模式评估模型的泛化能力，代码与训练模式类似
        losses = 0
        pred_labels = []
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)  # 封装评估数据集
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:  # 遍历每个评估batch
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            loss = criterion(output, label_id.to(device))  # 计算损失
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)  # 计算预测值
            acc = torch.sum(pred_label == label_id.to(device)).item() / len(pred_label)
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss = losses / len(valid_dataloader)
        print('\tLoss:', average_loss)  # 打印平均损失

        # 分类报告
        # classification_report:根据真实标签和预测标签输出每个类别的prf信息
        # 当前epoch执行完之后，当前模型对于评估测试集整体每个类别的prf信息
        report = metrics.classification_report(true_labels, pred_labels, labels=valid_dataset.labels_id,
                                               target_names=valid_dataset.labels)
        print('* Classification Report:')
        print(report)

        # f1 用来判断最优模型
        # f1_score:根据当前的epoch模型的真实标签和预测标签的f1值，判断当前模型是否是最优的
        f1 = metrics.f1_score(true_labels, pred_labels, labels=valid_dataset.labels_id, average='micro')

        if not os.path.exists('models'):
            os.makedirs('models')

        # 判断并保存验证集上表现最好的模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'models/best_model.pkl')


if __name__ == '__main__':
    main()
