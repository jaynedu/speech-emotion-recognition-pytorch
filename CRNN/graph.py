# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 12:20
# @Author  : Du Jing
# @FileName: graph.py
# ---- Description ----


import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_conv(in_size, kernel, stride, padding):
    """
    Compute H or W after convolution layer.

    Args:
        in_size: H or W
        kernel: (3, 3) => 3
        stride: (1, 1) => 1
        padding: 0 or 1
    """
    return (in_size + 2 * padding - kernel) // stride + 1


def compute_pool(in_size):
    """
    Compute H or W after max_pool layer.
    (in_size + 2 * padding - kernel) // stride + 1

    Args:
        in_size: H or W
    """
    return (in_size - 2) // 2 + 1


class BasicCNN(nn.Module):
    def __init__(self, in_size, kernels, dropout=0.2, activation=None):
        super().__init__()
        if activation is None:
            activation = ['relu', 'relu', 'relu']
        assert len(kernels) <= len(activation)

        self.cnn = nn.Sequential()
        for i in range(len(kernels)):
            if i == 0:
                self.cnn.add_module('conv_%s' % i, nn.Conv2d(in_size, kernels[i], 3, 1, 1))
            else:
                self.cnn.add_module('conv_%s' % i, nn.Conv2d(kernels[i-1], kernels[i], 3, 1, 1))
            self.cnn.add_module('activation_%s' % i, self.get_activation(activation[i]))
            self.cnn.add_module('max_pool_%s' % i, nn.MaxPool2d(2, 2))
            self.cnn.add_module('batch_norm_%s' % i, nn.BatchNorm2d(kernels[i]))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def get_activation(type='relu'):
        if type.lower() == 'leakyrelu':
            return nn.LeakyReLU(inplace=True)
        elif type.lower() == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)

    @staticmethod
    def compute_size(h, w, n):
        """
        Compute H and W after n basic CNNs.

        Returns:
            res => [h, w]
        """

        res = []
        for x in [h, w]:
            for i in range(n):
                x = compute_conv(x, 3, 1, 1)
                x = compute_pool(x)
            res.append(x)
        return res

    def forward(self, x):
        o = self.cnn(x)
        o = self.dropout(o)
        return o


class CRNN(nn.Module):
    def __init__(self, h, w, in_size, num_classes, kernels, hidden_size, num_layers, activation, dropout):
        super().__init__()
        self.cnn = BasicCNN(in_size, kernels, dropout, activation)
        h, w = self.cnn.compute_size(h, w, len(kernels))

        self.flat = nn.Flatten(1, -2)
        self.rnn = nn.GRU(w, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)

        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        o = self.cnn(x)
        o = self.flat(o)
        o, (h, c) = self.rnn(o, None)
        o = self.softmax(self.linear(h))
        return o

    def flatten_parameters(self):
        self.rnn.flatten_parameters()


