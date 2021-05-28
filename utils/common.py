# -*- coding: utf-8 -*-
# @Date      : 2021/5/23 8:35 下午
# @Author    : Du Jing
# @Filename  : common.py
# ---- Description ----
#

import os
import sys
import time


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class Logger:
    """
    把print的部分保存到YYYY-MM-DD-HH-mm.log文件

    Usage:
        sys.stdout = Logger(sys.stdout)  # 将输出记录到log

    """

    def __init__(self, stream=sys.stdout, log_dir='log', log_name=None):
        output_dir = log_dir
        Checker.check_path(log_dir)
        log_name = log_name if log_name is not None else '{}.log'.format(time.strftime('%Y-%m-%d_%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Checker(object):

    @classmethod
    def check_path(cls, path):
        if not os.path.exists(path):
            parent = os.path.split(path)[0]
            cls.check_path(parent)
            os.mkdir(path)


def catch_exception(origin_func):
    def wrapper(self, *args, **kwargs):
        try:
            u = origin_func(self, *args, **kwargs)
            return u
        except Exception as e:
            print(e)
            sys.exit(1)
    return wrapper