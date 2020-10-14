# -*- coding: utf-8 -*-
# @Date    : 2020/10/12 15:50
# @Author  : Du Jing
# @FileName: mudules.py
# ---- Description ----


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.scaled_dot_product_attention = ScaledDotProductAttention(scaled_factor=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # (batch, len, n*d) -> (batch, len, n, d)
        q = self.w_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.w_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.w_v(v).view(batch_size, len_v, self.n_head, self.d_v)

        # (batch, len, n, d) -> (batch, n, len, d)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # (batch, n, len, d) -> (batch, len, n, d) -> (batch, len, n*d)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        output = self.norm(output)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(input_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, input_size)
        self.norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        output = self.w_2(F.gelu(self.w_1(x)))
        output = self.dropout(output)
        output += residual

        output = self.norm(output)
        return output


class PositionEncoding(nn.Module):
    def __init__(self, hidden_size, n_position=200):
        super().__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, hidden_size))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, hidden_size):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / hidden_size) for hid_j in range(hidden_size)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
