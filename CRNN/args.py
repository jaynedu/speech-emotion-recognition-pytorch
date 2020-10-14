# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 13:18
# @Author  : Du Jing
# @FileName: args.py
# ---- Description ----

import os
import data_util


database = 'emodb'
emo_dict = data_util.EmoDB.emo_dict
root_dir = data_util.EmoDB.dir
base_dir = r'E:\__dataset__'
train_dir = os.path.join(base_dir, database, 'train')
test_dir = os.path.join(base_dir, database, 'test')
val_dir = os.path.join(base_dir, database, 'val')
log_dir = './logs'

batch_size = 4
epochs = 2
