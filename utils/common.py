# -*- coding: utf-8 -*-
# @Date    : 2020/9/16 18:44
# @Author  : Du Jing
# @FileName: common
# ---- Description ----


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


