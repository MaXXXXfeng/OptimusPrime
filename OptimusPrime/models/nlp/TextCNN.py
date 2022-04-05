#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : TextCNN.py
# @Author  : Max
# @Time    : 2022/2/14
# @Desc    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors

class Config(object):
    def __init__(self, name):
        self.set_name(name)
        self.pretrained_embedding_path = None
        self.vocab_size = 10000
        self.embed_dim = 100
        self.dropout = 0.5
        self.class_num = 2

        # 不可通过参数传入修改
        self.filter_size = (2, 5, 6)
        self.filter_num = 256

    def set_name(self, name):
        name = '_'.join([name, 'config'])
        self.__name__ = name

    def update_config(self,conf):
        if 'pretrained_path' in conf:
            self.pretrained_embedding_path = conf['pretrained_path']
        if 'embed_dim' in conf:
            self.embed_dim = conf['embed']
        if 'vocab_size' in conf:
            self.vocab_size = conf['vocab_size']
        if 'dropout' in conf:
            self.dropout = conf['dropout']
        if 'class_num' in conf:
            self.class_num = conf['class_num']



class TextCNN(nn.Module):
    __name__ = 'TextCNN'
    def __init__(self, conf=None):
        super(TextCNN, self).__init__()
        self.config = Config(self.__name__)
        # 更新参数
        self.config.update_config(conf)

        # 获取模型相关配置信息
        self.embedding = self.init_embedding(self.config.pretrained_embedding_path)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.filter_num, (k, self.embedding.embedding_dim)) for k in self.config.filter_size])
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.filter_num * len(self.config.filter_size), self.config.class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
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



