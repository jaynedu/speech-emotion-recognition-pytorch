# -*- coding: utf-8 -*-
# @Date    : 2020/10/12 14:48
# @Author  : Du Jing
# @FileName: run.py
# ---- Description ----

import os
import data_util
import config
import utils


# # 提取特征
# extractor = data_util.Extractor(config.root_dir)
# train, test, val = extractor.split(True)
# data = [train, test, val]
# dirs = [config.train_dir, config.test_dir, config.val_dir]
# scopes = ['train', 'test', 'val']
# for (x, y), dir, scope in zip(data, dirs, scopes):
#     extractor.extract(x, y, dir, scope)
