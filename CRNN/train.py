# -*- coding: utf-8 -*-
# @Date    : 2020/10/14 14:54
# @Author  : Du Jing
# @FileName: train.py
# ---- Description ----

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import os
import time
from tensorboardX import SummaryWriter
from sklearn.metrics import recall_score, classification_report, confusion_matrix

import utils
import data_utils
from CRNN import args
from CRNN.graph import CRNN


class Build(object):

    def __init__(self, dataset: dict, **params):

        self.dataset = dataset
        self.global_step = 0

        # Init parameters
        self.warmup_steps = params.pop('warmup_steps', 1000.)
        self.opt = params.pop('opt', 'sgd')
        self.lr = params.pop('lr', 0.001)
        self.momentum = params.pop('momentum', 0.99)
        self.l2_decay = params.pop('l2_decay', 5e-4)
        self.epochs = params.pop('epochs', 1000)
        self.batch_size = params.pop('batch_size', 32)
        self.random_seed = params.pop('random_seed', 666)
        self.log_dir = params.pop('log_dir', './logs')

        # Log and Tensorboard settings
        sys.stdout = utils.Logger(sys.stdout, log_dir=self.log_dir)
        train_log_dir = os.path.join(self.log_dir, 'train', '{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        val_log_dir = os.path.join(self.log_dir, 'val', '{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        test_log_dir = os.path.join(self.log_dir, 'val', '{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
        for _dir in [train_log_dir, val_log_dir, test_log_dir]:
            utils.check_dir(_dir)
        self.train_writer = SummaryWriter(train_log_dir)
        self.test_writer = SummaryWriter(test_log_dir)
        self.val_writer = SummaryWriter(val_log_dir)

        # Cuda settings
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        else:
            torch.seed()

    def __call__(self, n_sample=None, net=None):
        if isinstance(n_sample, list) or isinstance(n_sample, tuple):
            try:
                train_sample, test_sample, val_sample = n_sample
            except:
                print('Error! %s will be set to (None, None, None)' % n_sample)
                train_sample, test_sample, val_sample = None, None, None
        else:
            train_sample, test_sample, val_sample = n_sample, n_sample, n_sample

        # Define Data Loader
        self.train_loader = data_utils.create_loader_with_fixed_chunks(**self.dataset, n_sample=train_sample,
                                                                       batch_size=self.batch_size, drop_last=False,
                                                                       scope='train')
        self.val_loader = data_utils.create_loader_with_fixed_chunks(**self.dataset, n_sample=val_sample,
                                                                     batch_size=self.batch_size, drop_last=False,
                                                                     scope='val')
        self.test_loader = data_utils.create_loader_with_fixed_chunks(**self.dataset, n_sample=test_sample,
                                                                      batch_size=self.batch_size, drop_last=False,
                                                                      scope='test')

        if net is not None:
            self.train(net)

    def noam_scheme(self, step):
        return self.lr * self.warmup_steps ** 0.5 * np.minimum(step * self.warmup_steps ** -1.5, step ** -0.5)

    def train(self, net):
        self.global_step = 0
        total_loss = 0
        total_num = 0
        criterion = nn.CrossEntropyLoss(reduction='sum').cuda()

        for epoch in range(self.epochs):
            for i, (X, y) in enumerate(self.train_loader):

                self.global_step += 1

                lr = self.noam_scheme(self.global_step)
                self.train_writer.add_scalar('lr', lr, self.global_step)
                if self.opt.lower() == 'sgd':
                    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=self.momentum,
                                          weight_decay=self.l2_decay)
                else:
                    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=self.l2_decay)

                y_true = y.type(torch.LongTensor).cuda()
                x_input = Variable(X, requires_grad=True).type(torch.FloatTensor).cuda()

                with torch.no_grad():
                    y_logits = net(x_input).requires_grad_().cuda()

                y_pred = y_logits.max(-1)[1]
                pred = y_pred.cpu()
                true = y_true.cpu()
                uar = recall_score(true, pred, average='macro')
                self.train_writer.add_scalar('UAR', uar, self.global_step)

                optimizer.zero_grad()
                loss = criterion(y_logits.contiguous(), y_true)
                total_loss += loss
                total_num += y_logits.size(0)
                loss.backward()
                optimizer.step()
                self.train_writer.add_scalar('loss', total_loss / total_num, self.global_step)

                if self.global_step % 10 == 0:
                    print("Train: \tEpoch %4d\tBatch %3d\tStep %6d\tLoss %.6f\tUAR %.6f\tlr %.6f" %
                          (epoch, i, self.global_step, total_loss/total_num, uar, lr))

            self.validation(net)

        self.test(net)

    def validation(self, net):
        total_loss = 0
        total_num = 0

        total_pred = []
        total_true = []

        criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
        net.eval()
        with torch.no_grad():
            for X, y in self.val_loader:
                x_input = Variable(X).type(torch.FloatTensor).cuda()
                y_true = y.type(torch.LongTensor).cuda()
                y_logits = net(x_input)
                y_pred = y_logits.max(-1)[1]

                total_pred += y_pred.cpu().tolist()
                total_true += y.tolist()
                total_loss += criterion(y_logits.contiguous(), y_true).item()
                total_num += y_logits.size(0)
        uar = recall_score(total_true, total_pred, average='macro')
        print("Validate: \tLoss %.6f\tUAR %.6f" % (total_loss / total_num, uar))

    def test(self, net):
        total_loss = 0
        total_num = 0

        total_pred = []
        total_true = []

        criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
        net.eval()
        with torch.no_grad():
            for X, y in self.test_loader:
                x_input = Variable(X).type(torch.FloatTensor).cuda()
                y_true = y.type(torch.LongTensor).cuda()
                y_logits = net(x_input)
                y_pred = y_logits.max(-1)[1]

                total_pred += y_pred.cpu().tolist()
                total_true += y.tolist()
                total_loss += criterion(y_logits.contiguous(), y_true).item()
                total_num += y_logits.size(0)
        uar = recall_score(total_true, total_pred, average='macro')
        print("Test: \tLoss %.6f\tUAR %.6f" % (total_loss / total_num, uar))


if __name__ == '__main__':
    model = CRNN(**args.params).cuda()
    builder = Build(data_utils.DES)
    builder(None, model)