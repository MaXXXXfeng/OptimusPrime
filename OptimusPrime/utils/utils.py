#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Max
# @Time    : 2022/2/11
# @Desc    : 通用函数

import random

def split_array_by_random(arr, frac=0.7, seed=2022):
    '''
    按比例将数组随机拆分成两部分
    Args:
        arr (list): 待拆分数组
        frac (float): 拆分比例，默认7：3
        seed (int): 随机seed

    Returns:
        part1 (list),拆分部分1
        part2 (list),拆分部分2

    '''
    random.seed(seed)
    random.shuffle(arr)
    idx = int(len(arr) * frac)
    part1 = arr[:idx]
    part2 = arr[idx:]
    return part1,part2

def check_file_path(path, suffix):
    '''
    检查文件路径是否符合文件类型要求
    Args:
        path (str):
        suffix (str or list): 支持的文件后缀，可以为str或字符串列表。如['.txt','.csv']

    Returns:
        bool,符合会返回True,反之为False

    '''
    if not isinstance(path, str):
        raise Exception('file path must be string')
    if isinstance(suffix, str):
        suffix = [suffix]
    for ft in suffix:
        if path.endswith(ft):
            return True
    return False