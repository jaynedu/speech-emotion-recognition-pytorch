# -*- coding: utf-8 -*-
# @Date    : 2020/11/1 17:00
# @Author  : Du Jing
# @FileName: data_utils.py
# ---- Description ----
# Prepare database and extract features

import glob
import tqdm
import wave
import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data

import utils

np.set_printoptions(threshold=np.inf)

# ================================================================================
# Database settings
# Basic settings including: root dir and emotion dict
# ================================================================================


EMODB = utils.ConfigDict(
    root=r'E:\EmoDB',
    save_root=r'E:\__dataset__\emodb',
    mapping={
        'anger': 0,
        'boredom': 1,
        'disgust': 2,
        'fear': 3,
        'happiness': 4,
        'neutral': 5,
        'sadness': 6
    }
)

DES = utils.ConfigDict(
    root=r'E:\DES',
    save_root=r'E:\__dataset__\des',
    mapping={
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 3,
        'surprise': 4
    }
)

# ================================================================================
# Extract features by OpenSMILE
# Configuration file is self-designed
# ================================================================================


class OpenSMILE(object):
    """
    Using OpenSMILE to extract features

    Args:
        root: root dir of the raw dataset, where wav files exist.
        save_root: root dir of the extracted csv files.
        mapping: emotion dict, string to int.

    Others:
        categories: emotion in string type.
        classes: emotion in int type.

    Usage:
        extractor = OpenSMILE(root, save_root, mapping, **kwargs)
        extractor()

    """

    def __init__(self, root, save_root, mapping, **kwargs):
        self.root = root
        self.save_root = save_root
        self.mapping = mapping
        self.config = kwargs.pop('config', './my93.conf')
        self.opensmile = kwargs.pop('opensmile', r'D:/opensmile-2.3.0/bin/Win32/SMILExtract_Release.exe')
        self.random_seed = kwargs.pop('random_seed', 666)

        self.files = []
        self.categories = []
        self.classes = []
        self.durations = []

        self.train = None
        self.test = None
        self.val = None

    def __call__(self):
        self.get()
        (x_train, y_train), (x_test, y_test), (x_val, y_val) = self.split()
        self.train, self.test, self.val = len(y_train), len(y_test), len(y_val)
        extractList = [(x_train, y_train), (x_test, y_test), (x_val, y_val)]
        scopeList = ['train', 'test', 'val']
        for i, scope in enumerate(scopeList):
            X, y = extractList[i]
            self.extract(X, y, scope)
        self.show_info()

    def get(self):
        for path in tqdm.tqdm(glob.glob(os.path.join(self.root, '*', '*.wav')),
                              desc='Loading wav from %s' % self.root):
            self.files.append(path)
            emotion = os.path.split(os.path.split(path)[0])[1]
            self.categories.append(emotion)
            self.classes.append(self.mapping[emotion])
            with wave.open(path, 'r') as wf:
                fs, nframe = wf.getparams()[2: 4]
                duration = float(fs/nframe)
                self.durations.append(duration)
        return self

    def split(self):
        if len(self.files) == 0:
            self.get()

        x_train, x_test, y_train, y_test = train_test_split(self.files, self.categories, test_size=0.2, stratify=self.categories,
                                                            shuffle=True, random_state=self.random_seed)

        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test,
                                                        shuffle=True, random_state=self.random_seed)

        return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    def extract(self, X, y, scope='train'):
        save_dir = os.path.join(self.save_root, scope)
        utils.check_dir(save_dir)
        for wav, label in tqdm.tqdm(zip(X, y), desc='Extracting %s features' % scope.capitalize()):
            filename = os.path.split(wav)[1]
            output = os.path.join(save_dir, filename.split('.')[0] + '_%s.csv' % label)
            cmd = self.opensmile + " -noconsoleoutput -C " + self.config + " -I " + wav + " -O " + output
            res = subprocess.call(cmd)

    def show_info(self):
        print(' Dataset Information '.center(60, '='))

        print('Root Dir \t\t\t%s' % self.root)
        print('Save Root Dir \t\t%s' % self.save_root)
        print('Emotion Dict \t\t%s' % self.mapping)
        print('Train Length \t\t%s' % self.train)
        print('Test Length \t\t%s' % self.test)
        print('Val Length \t\t\t%s' % self.val)
        print('Max Duration \t\t%s' % max(self.durations))

        print('=' * 60)

# ================================================================================
# Create DataLoader
# ================================================================================


def read_features(save_root, mapping, n_sample=None, scope='train'):
    """
    Read csv features from root dir.

    Args:
        save_root: root dir of the extracted csv files.
        mapping: emotion dict, string to int.
        n_sample: number of samples to read, if n_sample == None, read all samples.
        scope: ['train', 'test', 'val']

    Returns:
        features: variable feature list, each feature in shape [nframe, nfeature]
        labels: label list, int.
    """

    features = []
    labels = []
    index = 0

    for file in tqdm.tqdm(glob.glob(os.path.join(save_root, scope, '*.csv')),
                          desc='Reading %s features' % scope):
        df = pd.read_csv(file)
        label = file.split('.')[0].split('_')[-1]

        features.append(df.to_numpy(np.float32))
        labels.append(mapping[label])
        index += 1
        if index == n_sample:
            break

    return features, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    features = [torch.tensor(data[0], dtype=torch.float32) for data in batch]
    labels = [data[1] for data in batch]
    padded_feature = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    return padded_feature, torch.tensor(labels, dtype=torch.long)


def fixed_chunk_feature(Xs, durations: list, window_length, window_step, chunk_length):
    """
    Convert frame-level features to fixed chunk-level features,
    solving the problem of variable length.

    n * step + (winlen - step) = total time => (n-1) * step + winlen = total time
    Referenced by: https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Lin_2020.pdf

    Args:
        Xs: frame-level features
        durations: durations of raw audios, seconds.
        window_length: default 0.032 seconds.
        window_step: default 0.016 seconds.
        chunk_length: default 1 second.
    """

    n_chunk = int(max(durations) / chunk_length + 1)
    steps = [(time - chunk_length) / (n_chunk - 1) for time in durations]
    frame_per_chunk = int((chunk_length - window_length) / window_step + 1) + 1
    frame_per_step = [int((step - window_length) / window_step + 1) + 1 for step in steps]
    # print("time durations: %s" % durations)
    # print("chunk number: %s" % n_chunk)
    # print("chunk steps: %s" % steps)
    # print("nframe per step: %s" % frame_per_step)
    # print("nframe per chunk: %d" % frame_per_chunk)

    features = []
    for i, x in enumerate(Xs):
        n_frame, n_feature = x.shape
        new_feature = np.zeros((n_chunk, frame_per_chunk, n_feature))
        for j in range(n_chunk):
            start = i * frame_per_step[j]
            end = i * frame_per_step[j] + frame_per_chunk
            chunk_feature = np.zeros((frame_per_chunk, n_feature))
            for k, frame in enumerate(x[start: end]):
                chunk_feature[k, :] = frame
            new_feature[j, :, :] = chunk_feature
        features.append(new_feature)
    return np.array(features)


def create_loader_with_padding(save_root, mapping, n_sample=None, scope='train', drop_last=True, batch_size=32, **kwargs):
    """
    Raw features are directly feed into model.
    To solve the variable length problem, pad 0 every batch by using function <collate_fn>

    feature shape: (batch, nframe, nfeature)
    """

    Xs, Ys = read_features(save_root, mapping, n_sample, scope)
    dataset = Dataset(Xs, Ys)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
    return loader


def create_loader_with_fixed_chunks(root, save_root, mapping, n_sample=None, scope='train', drop_last=True, batch_size=32,
                                    window_length=0.032, window_step=0.016, chunk_length=1, **kwargs):
    """
    Frame feature are converted to fixed-chunk features.
    Referenced by: https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Lin_2020.pdf

    feature shape: (batch, nchunk, chunk_length, nfeature)
    """

    durations = OpenSMILE(root, save_root, mapping).get().durations
    Xs, Ys = read_features(save_root, mapping, n_sample, scope)
    Xs = fixed_chunk_feature(Xs, durations, window_length, window_step, chunk_length)
    dataset = Dataset(Xs, Ys)
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=drop_last)
    return loader



