#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : BERT.py
# @Author  : Max
# @Time    : 2022/3/27
# @Desc    : bert + 全连接层

import torch.nn as nn
from transformers import BertModel


class Config(object):
    def __init__(self, name):
        self.set_name(name)
        self.class_num = 2

        self.bert_path = 'bert-base-chinese'

        # 不可通过参数传入修改
        self.hidden_size = 768  # 由预训练模型决定
    def set_name(self, name):
        name = '_'.join([name, 'config'])
        self.__name__ = name

    def update_config(self, conf):
        if conf is None:
            return

        if 'pretrained_bert_path' in conf:
            self.bert_path = conf['pretrained_bert_path']
        if 'bert_path' in conf:
            self.bert_path = conf['bert_path']
        if 'class_num' in conf:
            self.class_num = conf['class_num']



class BertBase(nn.Module):
    __name__ = 'BertBase'
    def __init__(self, conf=None):
        super(BertBase, self).__init__()
        self.config = Config(self.__name__)
        self.config.update_config(conf)
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.config.hidden_size, self.config.class_num)
    def forward(self, x):
        tokens = x[0]  # 输入的句子
        mask = x[2]  # 填充部分的mask标记，0为填充
        last_hidden_state, pooler_output = self.bert(tokens, attention_mask=mask,return_dict=False)
        out = self.fc(pooler_output)
        return out