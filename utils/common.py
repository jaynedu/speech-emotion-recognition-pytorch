# -*- coding: utf-8 -*-
# @Date    : 2020/9/16 18:44
# @Author  : Du Jing
# @FileName: common
# ---- Description ----

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
    """把print的部分保存到YYYY-MM-DD-HH-mm.log文件

    Usage: sys.stdout = Logger(sys.stdout)  # 将输出记录到log

    """
    def __init__(self, stream=sys.stdout, log_name=None):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = log_name if log_name is not None else '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class CheckDir(object):
    status = ConfigDict(
        exist=0,
        not_exist=1,
        empty=2,
        full=3,
        created=4,
        isdir=5,
        isfile=6
    )

    def __init__(self, path):
        self.path = path

    def create(self):
        if not os.path.exists(self.path):
            parent = CheckDir(os.path.split(self.path)[0])
            parent.create()
            os.mkdir(self.path)
        else:
            return self.status.exist
        return self.status.created

    def is_exist(self):
        return True if os.path.exists(self.path) else False

    def is_dir(self):
        return True if os.path.isdir(self.path) else False

    def is_empty(self):
        if self.is_exist():
            if self.is_dir():
                return len(os.listdir(self.path)) == 0
            else:
                return False
        return False



def check_dir(path):
    if not os.path.exists(path):
        parent = os.path.split(path)[0]
        check_dir(parent)
        os.mkdir(path)


def check_empty(path):
    content = os.listdir(path)
    return len(content) == 0


def catch_exception(origin_func):
    def wrapper(self, *args, **kwargs):
        try:
            u = origin_func(self, *args, **kwargs)
            return u
        except Exception as e:
            # self.revive() #不用顾虑，直接调用原来的类的方法
            print(e)
            sys.exit(1)
    return wrapper