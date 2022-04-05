#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Max
# @Time    : 2022/1/31
# @Desc    : pytorch 模型记录
from OptimusPrime.models.nlp.TextCNN import TextCNN
from OptimusPrime.models.nlp.LSTM import LSTM
from OptimusPrime.models.nlp.BERT import BertBase
from OptimusPrime.models.nlp.BertCNN import BertCNN

models = {
    'TextCNN': TextCNN,
    'LSTM': LSTM,
    'Bert': BertBase,
    'BertCNN': BertCNN
}

text_model_names = ['TextCNN', 'LSTM', 'Bert', 'BertCNN']

tabular_model_names = []