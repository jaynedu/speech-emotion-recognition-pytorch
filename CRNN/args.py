# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 13:18
# @Author  : Du Jing
# @FileName: args.py
# ---- Description ----


params = {
    'h': 62,
    'w': 93,
    'in_size': 3,  # usually 1, here the same as n_chunk.
    'kernels': [64, 128, 256],
    'activation': ['leakyrelu', 'relu', 'gelu'],
    'dropout': 0.2,
    'hidden_size': 256,
    'num_layers': 1,
    'num_classes': 5
}