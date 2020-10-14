# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 12:20
# @Author  : Du Jing
# @FileName: graph.py
# ---- Description ----


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNNBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, activation='gelu', dropout=0.1):
        super().__init__()
        assert activation in ['gelu', 'relu', 'leakyrelu']
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride)
        self.batchnorm = nn.BatchNorm2d(out_size, 1e-6)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.LeakyReLU()

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.batchnorm(output)
        output = self.activation(output)
        output = self.dropout(output)
        return output


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(93, 128, 1, batch_first=True, dropout=0.1, bidirectional=True)
        self.cnn_1 = BasicCNNBlock(1, 16, stride=2)
        self.cnn_2 = BasicCNNBlock(16, 64, stride=2)

        self.linear = nn.Linear()

    def forward(self, x):
        output, (h, c) = self.gru(x, None)
        output = output.unsqueeze(1)
        output = self.cnn_1(output)
        output = self.cnn_2(output)
        return output

