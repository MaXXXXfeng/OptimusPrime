#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : LSTM.py
# @Author  : Max
# @Time    : 2022/3/21
# @Desc    :

import torch
import torch.nn as nn
import numpy as np
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
        self.hidden_size = 128
        self.num_layers = 2

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
            self.dropout = nn.Dropout(conf['dropout'])
        if 'class_num' in conf:
            self.class_num = conf['class_num']


class LSTM(nn.Module):
    __name__ = 'LSTM'
    def __init__(self, conf=None):
        super(LSTM, self).__init__()
        self.config = Config(self.__name__)
        # 更新参数
        self.config.update_config(conf)

        self.embedding = self.init_embedding(self.config.pretrained_embedding_path)


        self.lstm = nn.LSTM(self.config.embed_dim, self.config.hidden_size, self.config.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.config.dropout)
        self.fc = nn.Linear(self.config.hidden_size * 2, self.config.class_num)

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

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
