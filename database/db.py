# -*- coding: utf-8 -*-
# @Date    : 2020/9/22 18:51
# @Author  : Du Jing
# @FileName: main
# ---- Description ----

import glob
import tqdm
import os
import sys
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

import utils


def catch_exception(origin_func):
    def wrapper(self, *args, **kwargs):
        try:
            u = origin_func(self, *args, **kwargs)
            return u
        except Exception as e:
            # self.revive() #不用顾虑，直接调用原来的类的方法
            print(e)
            sys.exit(1)
    return wrapper


class Database(object):
    """创建数据集

    给定数据集dir(从config继承),索引数据集

    """
    def __init__(self, dir):
        self.dir = dir

        files, emotions = self.__get_files(dir)
        self.files = files
        self.emotions = emotions

        self.length = len(self.files)

        self.opensmile = 'D:\\opensmile-2.3.0\\bin\\Win32\\SMILExtract_Release.exe'
        self.config = r'D:\speech-emotion-recognition-pytorch\database\my93.conf'

    def __getitem__(self, index):
        return self.files[index], self.emotions[index]

    def __len__(self):
        return self.length

    @catch_exception
    def __call__(self, train_dir, test_dir, only_split=False, only_read=False, random_state=666):

        assert not (only_split and only_read), 'only_split, only_read CANNOT be TRUE at the same time.'

        if not only_read:
            utils.check_dir(train_dir)
            utils.check_dir(test_dir)

            # split data set
            x_train, x_test, y_train, y_test = train_test_split(self.files,
                                                                self.emotions,
                                                                test_size=0.2,
                                                                stratify=self.emotions,
                                                                shuffle=True,
                                                                random_state=random_state)

            # extract features using opensmile
            for wav, label in tqdm.tqdm(zip(x_train, y_train), desc='Extracting train features...'):
                filename = os.path.split(wav)[1]
                self._get_feature_93(wav, os.path.join(train_dir, filename.split('.')[0] + '_%s.csv' % label))

            for wav, label in tqdm.tqdm(zip(x_test, y_test), desc='Extracting test features...'):
                filename = os.path.split(wav)[1]
                self._get_feature_93(wav, os.path.join(test_dir, filename.split('.')[0] + '_%s.csv' % label))

        if only_split:
            return

        # read features
        self.train = self._read_feature_93(train_dir)
        self.test = self._read_feature_93(test_dir)

    @staticmethod
    def __get_files(db_path):
        filepaths = []
        emotions = []

        for file in glob.glob(os.path.join(db_path, '*', '*.wav')):
            filepaths.append(file)
            emotion = os.path.split(os.path.split(file)[0])[1]
            emotions.append(emotion)

        return filepaths, emotions

    def _get_feature_93(self, input_wav, output_csv):
        cmd = self.opensmile + " -noconsoleoutput -C " + self.config + " -I " + input_wav + " -O " + output_csv
        res = subprocess.call(cmd)

    @catch_exception
    def _read_feature_93(self, dir):
        result = []
        filenames = os.listdir(dir)
        for filename in tqdm.tqdm(filenames, desc='Loading data from %s...' % dir):
            df = pd.read_csv(os.path.join(dir, filename))
            category = filename.split('.')[0].split('_')[-1]
            result.append([df, category])
        return result


