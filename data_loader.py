# -*- coding: utf-8 -*-
# @Date    : 2020/9/25 14:57
# @Author  : Du Jing
# @FileName: data_loader
# ---- Description ----

import torch
import torch.nn as nn
from torch.utils import data



class Trainset(data.Dataset):
    def __init__(self, dir):
        self.dir = dir