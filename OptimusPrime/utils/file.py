#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : file.py
# @Author  : Max
# @Time    : 2022/2/14
# @Desc    : 常见文件处理函数

import pandas as pd

from OptimusPrime.utils.utils import check_file_path


def read_text_lines(file_path):
    '''
    读取text文件
    Args:
        file_path (str): txt 文件路径

    Returns:
        读取的txt数据
        [line1,line2,...,line]
    '''
    if not check_file_path(file_path, '.txt'):
        raise Exception('File path must be txt file')
    with open(file_path) as f:
        lines = f.readlines()
    return lines


def split_content_label(line, tag, x_idx):
    '''
    拆分数据的内容和标签
    Args:
        line (str): 待拆分的文本
        tag (str): 分隔符
        x_idx (int): 拆分后内容所在数组索引

    Returns:
        拆分后的内容和标签
    '''
    sp_line = line.split(tag)
    content = sp_line[x_idx]
    label = sp_line[abs(1 - x_idx)]
    return content, label


def lines2df(lines, tag, x_idx):
    '''
    将字符串数组拆分后按content和label转为DataFrame
    Args:
        lines (list): 拆分后的字符串数组，[str,str,str]
        tag (str): 分隔符
        x_idx (int): 拆分后内容所在数组索引

    Returns:
        pd.DataFrame,包含content列和label列
    '''
    contents = []
    labels = []
    for line in lines:
        content, label = split_content_label(line, tag, x_idx)
        contents.append(content)
        labels.append(label)
    data = {'content': contents,
            'label': labels}
    data = pd.DataFrame(data)
    return data


def replace_label_by_dict(data, map_dict, label='label'):
    '''
    根据映射字典对指定列进行映射替换
    Args:
        data (DataFrame): 待映射数据
        map_dict (dict): 映射字典
        label (str): 需要替换的列名，默认为'label'

    Returns:
        映射替换后的数据
    '''
    # 根据字典进行映射
    data[label] = data[label].map(map_dict)
    return data


def text2csv(file_path, out_path, sp_tag, x_idx=0, label_map=None):
    '''
    读取文本文件，并将每行拆分为content和label。最终生成csv
    Args:
        file_path (str): text文件路径
        out_path (str): 要生产的csv文件路径，如果为空则不生成文件
        sp_tag (str): 每一行内容与标签的分隔符
        x_idx (int): 拆分后内容所在数组索引
        label_map (dict): 映射字典，默认为None,不进行替换

    Returns:
        pd.DataFrame,包含content列和label列

    '''
    lines = read_text_lines(file_path)
    data = lines2df(lines, sp_tag, x_idx)
    if label_map:
        data['label'] = data['label'].map(label_map)
    if out_path and check_file_path(out_path, '.csv'):
        data.to_csv(out_path, index=False)
    return data