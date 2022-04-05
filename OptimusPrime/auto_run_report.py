#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : auto_run_report.py
# @Author  : Max
# @Time    : 2022/4/4
# @Desc    : 自动集成全部模型
import pandas as pd
from tabulate import tabulate
from OptimusPrime.auto_run_single import AutoDeep
from OptimusPrime.utils.logger import logger
from OptimusPrime.models import text_model_names, tabular_model_names


def get_model_result(task_type, model_name, datas, params):
    '''
    给定参数，及数据，获取模型结果
    :param task_type: 任务类型，文本分类或表格数据等
    :type task_type: str
    :param model_name: 模型名字
    :type model_name: str
    :param datas: 数据文件路径，顺序为训练集，测试集，验证集
    :type datas: list
    :param params: 模型参数，包含数据处理以及训练过程中的参数
    :type params: dict
    :return: 对测试集的评估结果,key：评估指标，value:指标值
    :rtype: dict
    '''
    if task_type == 'text-cls':
        if model_name.find('bert'.lower()) == 0:
            task_type = 'text-bert-cls'

    ad = AutoDeep(task=task_type, model_name=model_name, **params)
    ad.process_data(train_path=datas[0], test_path=datas[1], val_path=datas[2], **params)
    result = ad.train(predict=True)
    result = {
        'name': model_name,
        'acc': result.get('acc', 0),
        'loss': result.get('loss', 0),
        'macro_precision': result['report']['macro avg'].get('precision', 0),
        'macro_recall': result['report']['macro avg'].get('recall', 0),
        'macro_f1': result['report']['macro avg'].get('f1-score', 0),
        'weighted_precision': result['report']['weighted avg'].get('precision', 0),
        'weighted_recall': result['report']['weighted avg'].get('recall', 0),
        'weighted_f1': result['report']['weighted avg'].get('f1-score', 0),
    }
    return result


def ensemble_model_result(train_path, test_path, val_path, task_type, save_path, params):
    '''
    集合汇总同类任务下的全部模型结果
    :param train_path: 训练数据集
    :type train_path: str
    :param test_path: 测试数据集
    :type test_path: str
    :param val_path: 验证数据集，若为空，后续会根据训练集进行切分
    :type val_path: str
    :param task_type: 任务类型
    :type task_type: str
    :param save_path: 各模型结果保存路径，若为空则不保存。仅支持csv
    :type save_path: str
    :param params: 其他建模参数,根据建模任务需求，填充其他的参数
    :type params: dict
    :return: 各模型结果汇总
    :rtype: dict
    '''
    logger.info(f'开始集合建模模式,任务类型: {task_type}')
    if task_type == 'text-cls':
        all_models = text_model_names
    else:
        all_models = tabular_model_names
    datas = [train_path, test_path, val_path]

    all_models = ['LSTM', 'TextCNN']
    result = []
    for model_name in all_models:
        model_result = get_model_result(task_type=task_type, model_name=model_name, datas=datas, params=params)
        result.append(model_result)
    result = pd.DataFrame(result)

    logger.info('各模型结果汇总')
    logger.info(tabulate(result, headers='keys', tablefmt='psql'))
    if save_path is not None and save_path.endswith('.csv'):
        result.to_csv(save_path, index=False)
        logger.info(f'结果文件保存至 {save_path}')
    return result


if __name__ == '__main__':
    pass
