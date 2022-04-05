#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : text_bert.py
# @Author  : Max
# @Time    : 2022/3/29
# @Desc    : 对bert类数据进行处理

import pandas as pd
import numpy as np
import swifter
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split

from OptimusPrime.config import TEXT_CONF, MODEL_CONF
from OptimusPrime.utils.logger import logger


class TextBertDataset(Dataset):
    '''Bert相关模型的文本数据数据集读取'''

    def __init__(self, data_path, vocab_path):
        '''
        初始化dataset，默认数据在content列，标签在label列
        不需要分词，自动会进行分词。label需要提前转换数字形式
        :param data_path: 数据路径，支持dataframe直接传入
        :type data_path: str/DataFrame
        :param vocab_path: 词表文件所在路径
        :type vocab_path:str
        '''
        logger.info('加载数据')
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        else:
            self.data = pd.read_csv(data_path)
        logger.info('数据加载完成')

        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)

        logger.info('开始分词')
        self.data['content'] = self.data['content'].swifter.apply(
            lambda x: self.tokenizer.encode_plus(x, max_length=TEXT_CONF.MAX_SEQ_LEN, padding="max_length",
                                                 truncation=True))  # 对文本进行分词
        logger.info('完成分词')
        if 'label' not in self.data.columns:
            self.data['label'] = -9
        self.data = self.data.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datas = self.data[item]
        data = datas[0]

        token_idx = np.array(data['input_ids'])
        seq_len = np.array(sum(data['attention_mask']))
        mask = np.array(data['attention_mask'])
        label = datas[1]

        return (token_idx, seq_len, mask), np.array(label)


def create_dataloader(data_path, vocab_path, frac=None, shuffle=False):
    '''
    根据文件路径生成对应的dataloader
    :param data_path: 文件路径，支持csv文件或DataFrame
    :type data_path: str/DataFrame
    :param vocab: 预训练词表文件路径
    :type vocab: str
    :param frac: 数据切分比例，若非空，则按比例切分训练集和验证集。默认为空
    :type frac: float
    :param shuffle: 是否需要打乱互数据，默认为False
    :type shuffle: bool
    :return: dataloader
    :rtype: dataloader
    '''
    dataset = TextBertDataset(data_path, vocab_path)
    if frac is None:
        data_loader = DataLoader(dataset, MODEL_CONF.BATCH_SIZE, shuffle=shuffle)
        return data_loader

    train_cnt = int(len(dataset) * frac)
    val_cnt = int(len(dataset) - train_cnt)
    dataset1, dataset2 = random_split(dataset=dataset, lengths=[train_cnt, val_cnt],
                                      generator=torch.Generator().manual_seed(2022))

    data_iter1 = DataLoader(dataset1, MODEL_CONF.BATCH_SIZE, shuffle=shuffle)
    data_iter2 = DataLoader(dataset2, MODEL_CONF.BATCH_SIZE, shuffle=shuffle)
    return data_iter1, data_iter2


def process_bert_text_data(train_path, test_path, val_path, vocab_path, frac=None):
    '''
    给定数据路径，处理后返回训练集、验证集和测试集对应的dataloader
    :param train_path: 训练数据路径
    :type train_path: str/DataFrame
    :param test_path: 测试数据路径
    :type test_path: str/DataFrame
    :param val_path: 验证数据路径
    :type val_path: str/DataFrame
    :param vocab_path: 词表路径
    :type vocab_path: str
    :param frac: 数据切分的比例
    :type frac: float
    :return: 训练集、验证集和测试集对应的dataloader
    :rtype: dataloader
    '''
    # 生成dataloader，train，val,test
    if val_path is None:
        if frac is None:
            frac = 0.7
        logger.info('验证集路径为空，自动切分训练集')
        train_iter, val_iter = create_dataloader(train_path, vocab_path, frac=frac)
    else:
        train_iter = create_dataloader(train_path, vocab_path, shuffle=True)
        val_iter = create_dataloader(val_path, vocab_path, shuffle=False)
    test_iter = create_dataloader(test_path, vocab_path, shuffle=False)

    return train_iter, val_iter, test_iter
