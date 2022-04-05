#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : auto_run_single.py
# @Author  : Max
# @Time    : 2022/2/17
# @Desc    :
import pandas as pd
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
from tabulate import tabulate

from OptimusPrime.config.model_config import MODEL_CONF
from OptimusPrime.data.Tokenizers import token_by_word
from OptimusPrime.data import process_text_data, create_text_dataloader, process_bert_text_data, \
    create_text_bert_dataloader
from OptimusPrime.models import models
from OptimusPrime.utils.logger import logger


class AutoDeep(object):
    def __init__(self, task='text-cls', model_name=None, **kwargs):
        '''
        初始化自动建模信息
        :param task: 任务类型
                    text-cls：常见文本分类任务
        :type task: str
        :param model_name: 使用的模型名字
        :type model_name: str
        :param kwargs: 其他配置信息字段，训练以及数据处理相关子弹
        :type kwargs: dict
        '''
        self.task = task
        MODEL_CONF.update_params(conf=kwargs)  # 更新相关配置参数
        if model_name is not None:
            self.build_model(model_name, **kwargs)

    def build_model(self, model_name, **kwargs):
        '''
        加载模型，更新self.model
        :param model_name: 模型名字
        :type model_name: str
        :param kwargs: 模型参数，具体参数参考对应模型文件中配置信息
        :type kwargs: dict
        '''
        model = models.get(model_name, None)
        if model is None:
            raise Exception('无法加载模型')
        self.model = model(conf=kwargs)
        logger.info(f'模型:{model_name} 创建完成')
        need_print = kwargs.get('print', False)
        logger.info(self.model)
        if need_print:
            print(self.model)

    def process_data(self, train_path, test_path, val_path=None, **kwargs):
        '''
        数据处理
        :param train_path: 训练集路径
        :type train_path: str
        :param test_path: 测试集路径
        :type test_path: str
        :param val_path: 验证集路径，若为空则自动切分训练集
        :type val_path: str
        :param kwargs: 不同类型数据处理任务所需的额外参数
        :type kwargs: dict
        :return: 处理后的训练集、验证集、测试集
        :rtype: Dataloader
        '''
        logger.info('开始数据处理')
        infos = kwargs
        if self.task in ['text-cls']:
            self.train_iter, self.val_iter, self.test_iter = self._process_text_cls([train_path, test_path, val_path],
                                                                                    infos)
            logger.info('数据处理完成')
            return self.train_iter, self.val_iter, self.test_iter
        if self.task in ['text-bert-cls']:
            self.train_iter, self.val_iter, self.test_iter = self._process_text_bert_cls(
                [train_path, test_path, val_path],
                infos)
            logger.info('数据处理完成')
            return self.train_iter, self.val_iter, self.test_iter
        return

    def _process_text_cls(self, data_path_list, infos):
        train_path = data_path_list[0]
        test_path = data_path_list[1]
        if len(data_path_list) == 3:
            val_path = data_path_list[2]
        else:
            val_path = None

        vocab_path = infos.get('vocab_path', None)
        build = infos.get('build_vocab', False)
        save = infos.get('save_vocab', False)
        frac = infos.get('split_frac', 0.7)
        tokenizer = infos.get('tokenizer', token_by_word)
        train_iter, val_iter, test_iter = process_text_data(train_path, test_path, val_path,
                                                            vocab_path, tokenizer,
                                                            build, save, frac)
        return train_iter, val_iter, test_iter

    def _process_text_bert_cls(self, data_path_list, infos):
        train_path = data_path_list[0]
        test_path = data_path_list[1]
        if len(data_path_list) == 3:
            val_path = data_path_list[2]
        else:
            val_path = None

        vocab_path = infos.get('vocab_path', None)
        frac = infos.get('split_frac', 0.7)
        train_iter, val_iter, test_iter = process_bert_text_data(train_path, test_path, val_path,
                                                                 vocab_path, frac)
        return train_iter, val_iter, test_iter

    def train(self, train_iter=None, val_iter=None, model_path=None, predict=True):
        '''
        给定数据，进行训练
        :param train_iter: 训练数据
        :type train_iter: dataloader
        :param val_iter: 验证数据
        :type val_iter: dataloader
        :param model_path: 模型保存路径
        :type model_path: str
        :param predict: 是否对测试集进行评估,默认评估
        :type predict: bool
        '''
        logger.info('开始训练')
        if train_iter is None:
            train_iter = self.train_iter
        if val_iter is None:
            val_iter = self.val_iter
        optimizer = torch.optim.Adam(self.model.parameters(), lr=MODEL_CONF.LR)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_func = MODEL_CONF.LOSS_FUNC

        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')

        logger.info(f'每隔{MODEL_CONF.PRINT_ITER} 轮，输出训练集和测试集的效果')
        for epoch in range(MODEL_CONF.EPOCH):
            print('Epoch [{}/{}]'.format(epoch + 1, MODEL_CONF.EPOCH))

            for trains, labels in tqdm(train_iter):
                outputs = self.model(trains)
                self.model.zero_grad()
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()  # 学习率衰减

                if total_batch % MODEL_CONF.PRINT_ITER == 0:
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss = self.evaluate(self.model, val_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        if model_path is not None:
                            torch.save(self.model, model_path)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                    logger.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                    self.model.train()
                total_batch += 1
        if predict:
            test_acc, test_loss, report = self.evaluate(model=self.model, data_iter=self.test_iter,
                                                        report=True)
            logger.info(f'测试集效果 Acc: {test_acc},  Loss: {test_acc}')
            logger.info('report')
            logger.info({tabulate(pd.DataFrame(report).T, headers='keys', tablefmt='psql')})
            eval_result = {
                'acc': test_acc,
                'loss': test_loss,
                'report': report
            }
            return eval_result
        return

    def predict(self, data, model, task_type, evaluate=False, **kwargs):
        '''
        预测数据，并打上对应标签
        :param data: 待预测数据,df或csv路径
        :type data: str/pd.DataFrame
        :param model: 模型路径，为空则加载self.model,默认为空
        :type model:str
        :param task_type: 任务类型，根据任务类型对数据进行对应处理
        :type task_type:str
        :param evaluate: 是否计算评估指标，包括acc,loss,以及report，默认不计算
        :type evaluate:bool
        :param kwargs: 不同类型任务数据处理参数等
        :type kwargs:dict
        :return: 预测后的dataframe，包括预测概率以及预测标签(若evaluate为True，会同时返回对应的评估结果)
        :rtype: DataFrame
        '''
        if isinstance(data, str) and data.endswith('.csv'):
            data = pd.read_csv(data)
        if task_type in ['text-cls']:
            # 加个参数获取的函数，更新分词器这些
            vocab = kwargs.get('vocab_path')
            tokenizer = kwargs.get('tokenizer', token_by_word)
            predict_iter = create_text_dataloader(data_path=data, vocab=vocab, tokenizer=tokenizer)
        elif task_type in ['text-bert-cls']:
            vocab = kwargs.get('vocab_path')
            predict_iter = create_text_bert_dataloader(data_path=data, vocab_path=vocab)
        else:
            predict_iter = None
        if model is None:
            if self.model is not None:
                model = self.model
            else:
                raise Exception('模型为空')
        else:
            model = self.load_model(model)
        acc, loss, report = None, None, None
        if evaluate:
            acc, loss, report = self.evaluate(model, predict_iter, result=False, report=True)
            logger.info('evaluate prdict data')
            logger.info(f'evaluate predict data, acc:{acc}, loss:{loss}')
            logger.info('report')
            logger.info({tabulate(pd.DataFrame(report).T, headers='keys', tablefmt='psql')})
        pred_y, pred = self.evaluate(model, predict_iter, result=True, report=False)
        data['pred_label'] = pred_y.tolist()
        data['pred'] = pred.tolist()

        label_map = kwargs.get('label_map', None)
        if label_map is not None:
            data['pred_label'] = data['pred_label'].map(label_map)

        if evaluate:
            return data, (acc, loss, report)
        return data

    def evaluate(self, model, data_iter, result=False, report=False):
        '''
        测评数据的整体表现，包括acc,loss和report
        :param model: 测试模型
        :type model: nn.model
        :param data_iter: 待评估数据
        :type data_iter: dataloader
        :param result: 是否返回预测结果，默认为False
        :type result: bool
        :param report: 是否计算report并返回，默认为False
        :type report: bool
        :return: 准确度，平均loss(result和report均为False的情况下)
        '''
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        outputs_all = np.array([], dtype=float)
        loss_func = MODEL_CONF.LOSS_FUNC
        with torch.no_grad():
            for texts, labels in data_iter:
                outputs = model(texts)
                loss = loss_func(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
                outputs_all = np.append(outputs_all, torch.max(outputs, 1)[0].numpy())

        acc = metrics.accuracy_score(labels_all, predict_all)
        # TODO: AUC,F-1
        avg_loss = loss_total / len(data_iter)

        if result:
            logger.info(f'Acc: {acc} , Loss:{avg_loss}')
            return predict_all, outputs_all
        if report:
            report = metrics.classification_report(labels_all, predict_all, output_dict=True)
            report.pop('accuracy')
            return acc, avg_loss, report

        return acc, avg_loss

    def load_model(self, model_path=None):
        '''
        加载模型
        :param model_path: 模型路径，若为空返回之前训练的模型，默认为空
        :type model_path: str
        :return: 训练好的模型
        :rtype: nn.model
        '''
        if model_path is not None:
            self.model = torch.load(model_path)
        logger.info('模型加载成功')
        return self.model


if __name__ == '__main__':
    # 单模型建模示例
    train_path = '../data/sampel_train.csv'
    val_path = '../data/sampel_train.csv'
    test_path = '../data/sampel_train.csv'
    vocab_path = '../data/sample_vocab.pkl'
    model_path = '../data/lstm_model_1'

    label_map = {'体育': 1, '娱乐': 2,
                 '家居': 3, '房产': 4,
                 '教育': 5, '时尚': 6,
                 '时政': 7, '游戏': 8,
                 '科技': 9, '财经': 0}
    label_map_rev = dict(zip(label_map.values(), label_map.keys()))  # k,v 翻转

    params = {'class_num': 10,
              'epoch': 5,
              'vocab_path': vocab_path,
              'batch_size': 64,
              'learning_rate': 0.05,
              'build_vocab': False,
              'bert_path': '../data//bert-base-chinese/',
              'print': False}

    ad = AutoDeep(task='text-cls', model_name='LSTM', **params)
    ad.process_data(train_path=val_path, test_path=val_path, val_path=val_path, **params)
    ad.train(model_path=model_path, predict=True)

    # 预测结果
    pre_param = {'vocab_oath': vocab_path, 'label_map': label_map_rev}
    ad.predict(data=test_path, model=model_path, task_type='text-cls', **pre_param)
