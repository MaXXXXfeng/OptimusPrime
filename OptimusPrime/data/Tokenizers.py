#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : Tokenizers.py
# @Author  : Max
# @Time    : 2022/2/11
# @Desc    : 记录分词方法

import logging

import jieba

jieba.setLogLevel(logging.INFO)


def token_by_char(sent):
    '''
    以单字级别进行分词
    Args:
        sent (str): 待分词的句子

    Returns:
        list，分词后结果
    '''
    chars = [c for c in sent]
    return chars


def token_by_word(sent):
    '''
    以词语级别进行分词，分词器为jieba
    Args:
        sent (str): 待分词的句子

    Returns:
        list，分词后结果
    '''
    words = list(jieba.cut(sent))
    return words


if __name__ == '__main__':
    sent = '今天天气很好！'
    print(token_by_char(sent))
    print(token_by_word(sent))
