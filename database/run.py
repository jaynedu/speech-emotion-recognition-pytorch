# -*- coding: utf-8 -*-
# @Date    : 2020/9/25 19:06
# @Author  : Du Jing
# @FileName: run
# ---- Description ----

import os
from database import Database
from database import ABC, CASIA, DES, eNTERFACE, EmoDB, GEMEP, IEMOCAP, SUSAS, URDU, VAM

base_dir = r'E:\dataset'
db = Database(EmoDB.dir)
db(train_dir=os.path.join(base_dir, 'emodb', 'train'),
   test_dir=os.path.join(base_dir, 'emodb', 'test'),
   only_split=True)