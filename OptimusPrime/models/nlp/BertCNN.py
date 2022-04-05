#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : BertCNN.py
# @Author  : Max
# @Time    : 2022/4/4
# @Desc    : 使用bert编码的CNN分类模型

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from gensim.models import KeyedVectors

class Config(object):
    def __init__(self, name):
        self.set_name(name)
        self.dropout = 0.5
        self.class_num = 2
        self.bert_path = 'bert-base-chinese'

        # 不可通过参数传入修改
        self.filter_size = (2, 5, 6)
        self.filter_num = 256
        self.hidden_size = 768  # 由预训练模型决定

    def set_name(self, name):
        name = '_'.join([name, 'config'])
        self.__name__ = name

    def update_config(self,conf):
        if 'vocab_size' in conf:
            self.vocab_size = conf['vocab_size']
        if 'dropout' in conf:
            self.dropout = conf['dropout']
        if 'class_num' in conf:
            self.class_num = conf['class_num']
        if 'pretrained_bert_path' in conf:
            self.bert_path = conf['pretrained_bert_path']
        if 'bert_path' in conf:
            self.bert_path = conf['bert_path']



class BertCNN(nn.Module):
    __name__ = 'BertCNN'
    def __init__(self, conf=None):
        super(BertCNN, self).__init__()
        self.config = Config(self.__name__)
        # 更新参数
        self.config.update_config(conf)

        # 获取模型相关配置信息
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.filter_num, (k, self.config.hidden_size)) for k in self.config.filter_size])
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.filter_num * len(self.config.filter_size), self.config.class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        tokens = x[0]  # 输入的句子
        mask = x[2]  # 填充部分的mask标记，0为填充
        last_hidden_state, pooler_output = self.bert(tokens, attention_mask=mask, return_dict=False)
        encode = last_hidden_state.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(encode, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def init_embedding(self, embed_path):
        if embed_path is not None:
            if embed_path.endswith('.txt'):
                embed = KeyedVectors.load_word2vec_format(embed_path, binary=False)
            else:
                embed = KeyedVectors.load_word2vec_format(embed_path, binary=True)
            weights = torch.FloatTensor(embed.vectors)
            embeddings = nn.Embedding.from_pretrained(weights)
        else:
            embeddings = nn.Embedding(self.config.vocab_size, self.config.embed_dim)
        return embeddings

