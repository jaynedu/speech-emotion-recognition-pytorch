# -*- coding: utf-8 -*-
# @Date    : 2020/10/12 14:48
# @Author  : Du Jing
# @FileName: run.py
# ---- Description ----

import os
import data_util
import utils


database = 'emodb'
root_dir = data_util.EmoDB.dir
base_dir = r'E:\__dataset__'
train_dir = os.path.join(base_dir, database, 'train')
test_dir = os.path.join(base_dir, database, 'test')
val_dir = os.path.join(base_dir, database, 'val')


# 提取特征
extractor = data_util.Extractor(root_dir)
train, test, val = extractor.split(True)
data = [train, test, val]
dirs = [train_dir, test_dir, val_dir]
scopes = ['train', 'test', 'val']
for (x, y), dir, scope in zip(data, dirs, scopes):
    extractor.extract(x, y, dir, scope)
