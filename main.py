#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : main.py
# @Author  : Max
# @Time    : 2022/4/5
# @Desc    :
from OptimusPrime.auto_run_report import ensemble_model_result

if __name__ == '__main__':
    # 示例1： 自动运行全部文本分类模型
    ## 文本分类模型参数
    text_params = {
        'class_num': 10,  # 类别数量
        'epoch': 1,
        'batch_size': 16,
        'vocab_path': './data/sample_vocab.pkl',  # 词表路径 / 词表构建语料路径
        'build': False,  # 是否需要构建词表
        'save': False,  # 词表保存路径
        'frac': 0.7,  # 若无验证集，训练集切分验证集训练集的占比
        'vocab_path_bert': './data/bert-base-chinese/vocab.txt',  # bert 预训练词表文件，需为txt格式
        'bert_path': './data/bert-base-chinese',  # bert 预训练模型路径(需下载模型)

    }
    train_path = './data/sampel_train.csv'  # 训练集路径
    val_path = './data/sampel_val.csv'  # 验证集路径
    test_path = './data/sampel_test.csv'  # 测试集路径
    report_path = './data/sample_result.csv'  # 汇总结果保留路径
    result = ensemble_model_result(train_path=train_path, test_path=test_path, val_path=val_path, task_type='text-cls',
                                   save_path=report_path, params=text_params)
