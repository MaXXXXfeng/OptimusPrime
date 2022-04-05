
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/MaXXXXfeng/typora-res/main/data/20220405220257.jpg" width="800"/>
    <br>
<p>

# OptimusPrime

![Version](https://img.shields.io/badge/Version-1.0-green.svg) ![Python3](https://img.shields.io/badge/Python-3-blue.svg?style=flat) ![pytorch](https://img.shields.io/badge/pytorch-1.10-orange)![transformers](https://img.shields.io/badge/transformers-4.12-lightgrey)
****

## 1 介绍

基于pytorch的分类任务合集。当前支持文本分类模型的快速建模和快速多模型建模。

## 2 快速使用

安装依赖：

``pip install -r requiremnts.txt``

### 2.1 集合建模

指定分类任务后，自动获取内置的全部模型进行建模，并汇总结果

共需三步完成：

1. 确定数据信息

   ```python
   from OptimusPrime.auto_run_report import ensemble_model_result
   
   # step1：确定数据信息
   train_path = './data/sampel_train.csv'  # 训练集路径
   val_path = './data/sampel_val.csv'  # 验证集路径
   test_path = './data/sampel_test.csv'  # 测试集路径
   report_path = './data/sample_result.csv'  # 汇总结果保留路径
   ```

   示例数据为随机抽取的数据，原数据为新浪新闻数据集[THUCNews](http://thuctc.thunlp.org/)

   **注意：当前文本类数据仅支持csv格式，内容与标签对应content和label列。**

2. 确定相关配置信息

   ```python
   # step2：确定相关配置信息
   text_params = {
           'class_num': 10,  # 类别数量
           'epoch': 5,
           'batch_size': 16,
           'vocab_path': './data/sample_vocab.pkl',  # 词表路径 / 词表构建语料路径
           'build': False,  # 是否需要构建词表
           'save': False,  # 词表保存路径
           'frac': 0.7,  # 若无验证集，训练集切分验证集训练集的占比
           'vocab_path_bert': './data/bert-base-chinese/vocab.txt',  # bert 预训练词表文件，需为txt格式
           'bert_path': './data/bert-base-chinese',  # bert 预训练模型路径(需下载模型)
   
       }
   ```

   注意：需要实现下载bert相关预训练模型，项目中使用的为[hugging-face预训练模型](https://huggingface.co/bert-base-chinese/tree/main)

3. 执行建模代码

   ```python
   result = ensemble_model_result(train_path=train_path, test_path=test_path, val_path=val_path, task_type='text-cls',
                                      save_path=report_path, params=text_params)
   ```

   运行代码后会自动进行数据处理和建模，各模型的结果会根据`report_path`保存

### 2.2 单模型建模

指定具体的模型，进行建模。支持以下功能：

- 自动处理数据
- 自动训练模型并保存
- 加载模型，预测并返回结果



以文本分类的LSTM任务为例：

1. 确定数据相关信息

   ```python
   from OptimusPrime.auto_run_single import AutoDeep
   
   train_path = '../data/sampel_train.csv'
   val_path = '../data/sampel_train.csv' 
   test_path = '../data/sampel_train.csv'
   vocab_path = '../data/sample_vocab.pkl'
   model_path = '../data/lstm_model_1'
   ```
   
   - 若验证集路径会空，会根据训练集进行切分
   - 若无事先构建的词表，可以传入语料进行自动构建
   
2. 确定相关配置信息

   ```python
   params = {
     'class_num': 10,
     'epoch': 5,
     'vocab_path': vocab_path,
     'batch_size': 64,
     'learning_rate': 0.05,
     'build_vocab': False,
      'print': False}
   ```

3. 构建任务，处理数据并训练

   ```python
   ad = AutoDeep(task='text-cls', model_name='LSTM', **params)
   ad.process_data(train_path=val_path, test_path=val_path, val_path=val_path, **params)
   ad.train(model_path=model_path, predict=True)
   ```

   - 不同模型可以调整的参数不同，具体参数可以参考模型文件对应的配置信息

4. 预测结果

   ```python
   pre_param = {'vocab_oath': vocab_path, 'label_map': label_map_rev}
       ad.predict(data=test_path, model=model_path, task_type='text-cls', **pre_param)
   ```

   

## 3 支持模型

当前仅支持文本分类模型，后续会添加表格类数据的模型

| 任务类型 | 任务类型               | 是否支持 | 说明                             |
| -------- | ---------------------- | -------- | -------------------------------- |
| 文本     | text-cls/text-bert-cls | 支持     | 分别对应常见模型和基于bert的模型 |
| 表格数据 | tabular-cls            | 待开发   |                                  |

### 3.1 文本分类模型

当前支持模型：

| 模型    | 说明                   |
| ------- | ---------------------- |
| TextCNN | cnn文本分类模型        |
| LSTM    | LSTM文本分类，支持变长 |
| Bert    | bert + 全连接          |
| BertCNN | bert + CNN             |

其他模型有时间会陆续更新



## 4 测试环境

- Python3
- Mac OS
