# -*- coding: utf-8 -*-
# @Date    : 2020/10/12 14:54
# @Author  : Du Jing
# @FileName: config.py
# ---- Description ----


import os
import data_util


database = 'casia'
root_dir = data_util.Database.CASIA
base_dir = r'E:\__dataset__'
train_dir = os.path.join(base_dir, database, 'train')
test_dir = os.path.join(base_dir, database, 'test')
val_dir = os.path.join(base_dir, database, 'val')