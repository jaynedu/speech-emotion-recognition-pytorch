# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 12:23
# @Author  : Du Jing
# @FileName: dataloader.py
# ---- Description ----

import torch
from torch.utils import data

from data_util import Extractor


__all__ = ['data_loader']

class Dataset(data.Dataset):
    def __init__(self, data_dir, emo_dict):
        features, emotions = Extractor.read(data_dir, return_array=True)
        self.features = features
        self.labels = [emo_dict[key] for key in emotions]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def data_loader(dir, emo_dict, batch_size, drop_last=True, expand_dim=None):

    def collate_fn(batch):
        batch.sort(key=lambda data: len(data[0]), reverse=True)
        features = [torch.tensor(data[0], dtype=torch.float32, device='cuda') for data in batch]
        labels = [data[1] for data in batch]
        padded_feature = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
        if expand_dim is not None:
            padded_feature = padded_feature.unsqueeze(expand_dim)
        return padded_feature, torch.tensor(labels, dtype=torch.long, device='cuda')

    dataset = Dataset(dir, emo_dict)
    loader = data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
    return loader
