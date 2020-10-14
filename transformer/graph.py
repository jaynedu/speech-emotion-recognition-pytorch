# -*- coding: utf-8 -*-
# @Date    : 2020/10/12 15:45
# @Author  : Du Jing
# @FileName: graph.py
# ---- Description ----


import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.mudules as module


class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()



class TransformerLike(nn.Module):
    def __init__(self):
        super().__init__()

        self.position_encoding = module.PositionEncoding(hidden_size=, n_position=)
        self.multi_head_attention = module.MultiHeadAttention(n_head=, d_model=, d_k=, d_v=)
