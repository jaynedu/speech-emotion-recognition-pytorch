# -*- coding: utf-8 -*-
# @Date    : 2020/10/12 15:50
# @Author  : Du Jing
# @FileName: mudules.py
# ---- Description ----


import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scaled_factor, attn_dropout=0.1):
        super().__init__()
        self.scaled_factor = scaled_factor
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.scaled_factor, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
