# coding: utf-8

import os
import json
import torch
import random
import torch.nn as nn
import numpy as np
import argparse

from transformers import BertTokenizer, AdamW, BertConfig
from model import BertClassifier
from dataset import CNewsDataset
from tqdm import tqdm
from sklearn import metrics
from torch.utils.data.distributed import DistributedSampler


def train():
    # ----------------获取参数设置----------------
    model_config = json.load(open(args.model_config_file))
    model_name_or_path = model_config['model_name_or_path']
    train_data_path = model_config['train_data_path']
    eval_data_path = model_config['eval_data_path']
    batch_size = model_config['batch_size']
    learning_rate = model_config['learning_rate']
    epochs = model_config['num_epochs']
    output_dir = model_config['output_dir']

    # ----------------GPU配置与GPU通讯方式配置----------------
    # rank是分配给分布式组中每个进程的唯一标识符，是连续的整数，范围从0到world_size
    local_rank = int(os.environ.get("LOCAL_RANK"))
    # 根据local_rank指定device
    torch.cuda.set_device(local_rank)
    # 在调用任何其他方法之前，需要使用该函数初始化distribute包
    # 指定pytorch分布式训练通讯后端（GPU -> nccl / CPU -> gloo）
    torch.distributed.init_process_group(backend='nccl')

    # ----------------固定随机种子（种子确定，模型训练结果可复现）----------------
    seed = 1234
    # 为python设置随机种子
    random.seed(seed)
    # 为numpy设置随机种子
    np.random.seed(seed)
    # 为CPU设置随机种子
    torch.manual_seed(seed)
    # 为所有GPU设置随机种子
    torch.cuda.manual_seed_all(seed)

    # ----------------数据读取与封装----------------
    # 数据读取
    train_dataset = CNewsDataset(train_data_path)
    valid_dataset = CNewsDataset(eval_data_path)

    # 数据封装
    # DistributedSampler向不同GPU分发训练数据
    train_sampler = DistributedSampler(train_dataset)
    # DataLoader根据数据集和采样器，完成batch的划分
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    valid_sampler = DistributedSampler(valid_dataset)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)

    # ----------------模型加载与DDP配置----------------
    # 读取BERT的配置文件
    bert_config = BertConfig.from_pretrained(model_name_or_path)
    num_labels = len(train_dataset.labels)

    # 初始化模型
    model = BertClassifier(bert_config, num_labels).cuda()

    # 创建分布式并行模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    # ----------------优化器与损失函数配置----------------
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_f1 = 0

    # ----------------模型训练过程----------------
    for epoch in range(1, epochs + 1):
        losses = 0
        accuracy = 0

        model.train()
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 模型参数置零
            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            # 传入数据，调用model.forward()
            output = model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                token_type_ids=token_type_ids.cuda(),
            )

            # 计算loss
            loss = criterion(output, label_id.cuda())
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.cuda()).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        # ----------------模型验证过程----------------
        model.eval()
        losses = 0
        # eval数据集所有的预测标签
        pred_labels = []
        # eval数据集所有的真实标签
        true_labels = []
        valid_bar = tqdm(valid_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda(),
                token_type_ids=token_type_ids.cuda(),
            )

            # 计算当前batch的loss
            loss = criterion(output, label_id.cuda())
            losses += loss.item()

            # 计算当前batch的acc
            pred_label = torch.argmax(output, dim=1)
            acc = torch.sum(pred_label == label_id.cuda()).item() / len(pred_label)
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss = losses / len(valid_dataloader)
        print('\tLoss:', average_loss)

        # 统计各个类别的PRF值
        report = metrics.classification_report(true_labels, pred_labels, labels=valid_dataset.labels_id,
                                               target_names=valid_dataset.labels)
        print('* Classification Report:')
        print(report)

        # f1 用来判断最优模型
        f1 = metrics.f1_score(true_labels, pred_labels, labels=valid_dataset.labels_id, average='micro')

        if not os.path.exists('models'):
            os.makedirs('models')

        # 判断并保存验证集上表现最好的模型
        if local_rank == 0 and f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output_dir)


def main():
    train()


if __name__ == '__main__':
    # 读取配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--cached_data', type=str, help='the cached data file')
    args = parser.parse_args()
    main()


