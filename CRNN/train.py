# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 14:54
# @Author  : Du Jing
# @FileName: train.py
# ---- Description ----


import torch
import torch.nn as nn
import sys
import os
import time
from tensorboardX import SummaryWriter
from CRNN.dataloader import data_loader
import utils

from CRNN import args
from CRNN.graph import CRNN


class Train(object):

    def __init__(self, train_dir, val_dir, test_dir, log_dir, emo_dict, seed=666):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.seed()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.emo_dict = emo_dict

        sys.stdout = utils.Logger(sys.stdout, log_dir=log_dir)
        train_log_dir = os.path.join(log_dir, 'train', '{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        val_log_dir = os.path.join(log_dir, 'val', '{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        test_log_dir = os.path.join(log_dir, 'val', '{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        for _dir in [train_log_dir, val_log_dir, test_log_dir]:
            utils.check_dir(_dir)
        self.train_writer = SummaryWriter(train_log_dir)
        self.test_writer = SummaryWriter(test_log_dir)
        self.val_writer = SummaryWriter(val_log_dir)

    def __call__(self, graph, epochs, batch_size):

        # Define Data Loader
        self.train_loader = data_loader(self.train_dir, self.emo_dict, batch_size)
        self.val_loader = data_loader(self.val_dir, self.emo_dict, batch_size, drop_last=False)
        self.test_loader = data_loader(self.test_dir, self.emo_dict, batch_size, drop_last=False)

        print('[[ Train Start ]]\n')
        self._train(graph, epochs, batch_size)

        print('\n[[ Train End ]]')

    def _train(self, graph, epochs, batch_size):
        for epoch in range(epochs):
            for i, (data, label) in enumerate(self.train_loader):
                print(epoch, i, data.size(), label)
                graph.train()
                output = graph(data)
                print(output.size())

            self._val(graph)

    def _val(self, graph):
        print(" Validation ".center(80, '='))
        for i, (data, label) in enumerate(self.val_loader):
            print('VAL...', i, data.size(), label)

        print('=' * 80)


if __name__ == '__main__':
    train = Train(
        args.train_dir,
        args.val_dir,
        args.test_dir,
        args.log_dir,
        args.emo_dict
    )
    train(CRNN().cuda(), args.epochs, args.batch_size)