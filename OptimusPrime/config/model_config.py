#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : model_config.py
# @Author  : Max
# @Time    : 2022/2/14
# @Desc    : 模型训练参数

import torch


class ModelConfig(object):
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 64
        self.EPOCH = 5
        self.LR = 1e-3 # learning rate
        self.PRINT_ITER = 100 # 每隔X个iter 打印训练结果
        self.LOSS_FUNC = torch.nn.CrossEntropyLoss()
    def update_params(self,conf):
        if 'epoch' in conf:
            self.EPOCH = conf['epoch']
        if 'batch_size' in conf:
            self.BATCH_SIZE = conf['batch_size']
        if 'learning_rate' in conf:
            self.LR = conf['learning_rate']
        if 'loss_func' in conf:
            self.LOSS_FUNC = conf['loss_func']
        if 'print_iter' in conf:
            self.PRINT_ITER = conf['print_iter']


MODEL_CONF = ModelConfig()

if __name__ == '__main__':
    print(MODEL_CONF.DEVICE)

