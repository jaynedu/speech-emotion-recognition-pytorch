# -*- coding: utf-8 -*-
# @Date    : 2020/9/16 18:44
# @Author  : Du Jing
# @FileName: common
# ---- Description ----

import os


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


def check_dir(path):
    if not os.path.exists(path):
        parent = os.path.split(path)[0]
        check_dir(parent)
        os.mkdir(path)