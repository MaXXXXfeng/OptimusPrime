#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : logger.py
# @Author  : Max
# @Time    : 2022/3/16
# @Desc    : 日志记录
import os
import logging
import datetime
from logging import handlers
from pathlib import Path

class Logger(object):
    '''日志记录,保存训练过程中的日志信息'''
    def __init__(self, filename, level='INFO', when='D', backCount=0,
                 fmt='%(asctime)s'
                     '-%(filename)s'
                     '-%(levelname)s: %(message)s'):

        now = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        filename = ''.join([now,'_',filename])
        cur_path = os.path.join(Path(os.path.dirname(__file__)).parent,'logs')

        file_path= os.path.join(cur_path,filename)
        self.logger = logging.getLogger(file_path)               # 根据文件名创建一个日志

        self.logger.setLevel(level)                             # 设置默认日志级别
        self.format_str = logging.Formatter(fmt)                # 设置日志格式

        screen_handler = logging.StreamHandler()                # 屏幕输出处理器
        screen_handler.setFormatter(self.format_str)            # 设置屏幕输出显示格式

        # 定时写入文件处理器
        time_file_handler = handlers.TimedRotatingFileHandler(filename=file_path,        # 日志文件名
                                                              when=when,                # 多久创建一个新文件
                                                              interval=1,               # 写入时间间隔
                                                              backupCount=backCount,    # 备份文件的个数
                                                              encoding='utf-8')         # 编码格式

        time_file_handler.setFormatter(self.format_str)

        # 添加日志处理器
        self.logger.addHandler(screen_handler)
        self.logger.addHandler(time_file_handler)

    def create_logger(self):
        return self.logger

logger = Logger('model_info.log').create_logger()
