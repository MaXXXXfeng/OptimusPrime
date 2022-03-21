#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Max
# @Time    : 2022/1/31
# @Desc    :
from OptimusPrime.models.nlp.TextCNN import TextCNN
from OptimusPrime.models.nlp.LSTM import LSTM

models = {
    'TextCNN': TextCNN,
    'LSTM': LSTM
}
