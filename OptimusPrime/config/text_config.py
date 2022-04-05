#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : text_config.py
# @Author  : Max
# @Time    : 2022/2/11
# @Desc    :

class TextDataConfig:
    MAX_SIZE = 10000 # 词表长度限制
    MINI_FREQ = 1 # 词表最低词频
    UNK = '<UNK>' # 未登录字 标志符
    PAD = '<PAD>' # padding 字符
    MAX_SEQ_LEN = 512 # 句子长度


TEXT_CONF = TextDataConfig()

if __name__=='__main__':
    print(TEXT_CONF.PAD)
