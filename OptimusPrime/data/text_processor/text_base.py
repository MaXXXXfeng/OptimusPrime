#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : text_base.py
# @Author  : Max
# @Time    : 2022/2/11
# @Desc    : 通用版文本数据处理
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import swifter
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from OptimusPrime.utils.utils import check_file_path
from OptimusPrime.config import TEXT_CONF, MODEL_CONF
from OptimusPrime.utils.logger import logger


class TextDataset(Dataset):
    '''常见文本数据数据集读取'''

    def __init__(self, data_path, vocab, tokenizer):
        '''
        初始化dataset，默认数据在content列，标签在label列
        不需要分词，自动会进行分词。label需要提前转换数字形式
        :param data_path: 数据路径，支持dataframe直接传入
        :type data_path: str/DataFrame
        :param vocab: 词表
        :type vocab: dict
        :param tokenizer: 分词器
        :type tokenizer: func
        '''
        logger.info('加载数据')
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        else:
            self.data = pd.read_csv(data_path)
        logger.info('数据加载完成')

        logger.info('开始分词')
        self.data['content'] = self.data['content'].swifter.apply(lambda x: tokenizer(x))  # 对文本进行分词
        logger.info('完成分词')
        if 'label' not in self.data.columns:
            self.data['label'] = -9

        self.data = self.data.values
        self.vocab_dict = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        tokens = data[0]
        label = data[1]
        token_idx = []

        seq_len = len(tokens)
        if len(tokens) < TEXT_CONF.MAX_SEQ_LEN:
            tokens.extend([TEXT_CONF.PAD] * (TEXT_CONF.MAX_SEQ_LEN - len(tokens)))
        else:
            tokens = tokens[:TEXT_CONF.MAX_SEQ_LEN]
            seq_len = TEXT_CONF.MAX_SEQ_LEN
        for token in tokens:
            token_idx.append((self.vocab_dict.get(token, self.vocab_dict.get(TEXT_CONF.UNK))))
        return (np.array(token_idx), np.array(seq_len)), np.array(label)


def build_vocab(vocab_path, tokenizer, build=False, save=False):
    '''
    基于词表文件或语料文件构建词表信息
    Args:
        vocab_path (str): 文件路径，支持txt或csv。如果针对原始语料进行词表构建，build需要设为True
        tokenizer (func): 分词函数
        build (bool): 是否需要基于语料文件进行词表构建。
                      True: 读取语料文件信息，生成词表
                      False: 直接读取词表文件
        save (str): 是否需要保存生成的词表文件，若非空，则根据save的路径保存为pkl文件
    Returns:
        词表字典，key:token value：token_index

    '''
    logger.info('开始构建词表')

    if not build:
        vocab_file = open(vocab_path, 'rb')
        vocab_dict = pickle.load(vocab_file)
        logger.info('词表构建完成')
        return vocab_dict
    if check_file_path(vocab_path, '.txt'):
        f = open(vocab_path, 'rb')
        lines = f.readlines()
        f.close()
    elif check_file_path(vocab_path, '.csv'):
        data = pd.read_csv(vocab_path)
        lines = list(data['content'])
    else:
        raise Exception('文件类型暂不支持')

    vocab_dict = {}
    logger.info('start building vocab dict')
    for line in tqdm(lines):
        content = line.strip()
        for word in tokenizer(content):
            vocab_dict[word] = vocab_dict.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= TEXT_CONF.MINI_FREQ], key=lambda x: x[1],
                        reverse=True)[
                 :TEXT_CONF.MAX_SIZE]
    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dict.update({TEXT_CONF.UNK: len(vocab_dict), TEXT_CONF.PAD: len(vocab_dict) + 1})
    if save:
        try:
            pickle.dump(vocab_dict, open(save, 'wb'))
        except Exception as e:
            logger.error(f'词表保存失败, {e}')
    logger.info('词表构建完成')
    return vocab_dict


def create_dataloader(data_path, vocab, tokenizer, frac=None, shuffle=False, **kwargs):
    '''
    根据文件路径生成对应的dataloader
    :param data_path: 文件路径，支持csv文件或DataFrame
    :type data_path: str/DataFrame
    :param vocab: 词表字典，直接传入路径进行加载或直接生成
    :type vocab: dict/str
    :param tokenizer: 分词器
    :type tokenizer: func
    :param frac: 数据切分比例，若非空，则按比例切分训练集和验证集。默认为空
    :type frac: float
    :param shuffle: 是否需要打乱互数据，默认为False
    :type shuffle: bool
    :param kwargs: 词表相关信息，包括是否重新训练，是否保存
    :type kwargs: dict
    :return: dataloader
    :rtype: dataloader
    '''
    if isinstance(vocab, str):
        build = kwargs.get('build_vocab', False)
        save = kwargs.get('save_vocab', None)
        vocab = build_vocab(vocab, tokenizer, build, save)  # 单独使用函数时，可以自动构建词表

    dataset = TextDataset(data_path, vocab, tokenizer)
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


def process_text_data(train_path, test_path, val_path, vocab_path, tokenizer, build, save, frac=None):
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
    :param tokenizer: 分词器
    :type tokenizer: func
    :param build: 是否需要基于词表内容训练新词表
    :type build: bool
    :param save: 训练后词表是否需要保存,默认为False不保存，若保存需传入路径
    :type save: str
    :param frac: 数据切分的比例
    :type frac: float
    :return: 训练集、验证集和测试集对应的dataloader
    :rtype: dataloader
    '''
    vocab = build_vocab(vocab_path, tokenizer, build, save)
    # 生成dataloader，train，val,test
    if val_path is None:
        if frac is None:
            frac = 0.7
        logger.info('验证集路径为空，自动切分训练集')
        train_iter, val_iter = create_dataloader(train_path, vocab, tokenizer, frac=frac)
    else:
        train_iter = create_dataloader(train_path, vocab, tokenizer, shuffle=True)
        val_iter = create_dataloader(val_path, vocab, tokenizer, shuffle=False)
    test_iter = create_dataloader(test_path, vocab, tokenizer, shuffle=False)

    return train_iter, val_iter, test_iter


if __name__ == '__main__':
    from OptimusPrime.data.Tokenizers import token_by_word

    train_path = '../../../data/sampel_train.csv'
    val_path = '../../../data/sampel_val.csv'
    test_path = '../../../data/sampel_test.csv'
    vocab_path = '../../../data/sample_vocab.pkl'
    a, b, c = process_text_data(train_path, test_path, vocab_path, token_by_word, **{'build': False})
