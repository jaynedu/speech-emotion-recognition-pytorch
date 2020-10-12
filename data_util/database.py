# -*- coding: utf-8 -*-
# @Date    : 2020/9/22 18:51
# @Author  : Du Jing
# @FileName: main
# ---- Description ----

import glob
import tqdm
import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

import utils


class Database:
    ABC = r'E:\ABC'
    CASIA = r'E:\CASIA'
    DES = r'E:\DES'
    eNTERFACE = r'E:\eNTERFACE'
    EmoDB = r'E:\EmoDB'
    GEMEP = r'E:\GEMEP'
    IEMOCAP = r'E:\IEMOCAP'
    SUSAS = r'E:\SUSAS'
    URDU = r'E:\URDU'
    VAM = r'E:\VAM'


class Extractor(object):

    def __init__(self, root_dir, config=None, random_seed=666):

        self.root_dir = root_dir
        self.random_seed = random_seed

        self.opensmile = 'D:\\opensmile-2.3.0\\bin\\Win32\\SMILExtract_Release.exe'
        self.config = config if config is not None else r'D:\PycharmProjects\speech-emotion-recognition-pytorch\data_util\my93.conf'

    def get_files(self, dir):
        """ Get pairs of 'filepath and label' from data_util path. """

        filepaths = []
        emotions = []

        for file in glob.glob(os.path.join(dir, '*', '*.wav')):
            filepaths.append(file)
            emotions.append(self.get_emotion(file))

        return filepaths, emotions

    @staticmethod
    def get_emotion(filepath):
        """ Get emotion from specific filepath. """
        return os.path.split(os.path.split(filepath)[0])[1]

    def split(self, val=False):
        """ Split dataset.
        if val is true, then generate validate set, else not.
        """
        filepaths, emotions = self.get_files(self.root_dir)

        x_train, x_test, y_train, y_test = train_test_split(filepaths, emotions, test_size=0.2, stratify=emotions,
                                                            shuffle=True, random_state=self.random_seed)
        if val:
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test,
                                                            shuffle=True, random_state=self.random_seed)
            return (x_train, y_train), (x_test, y_test), (x_val, y_val)

        return (x_train, y_train), (x_test, y_test)

    def _extract(self, input_wav, output_csv):
        """ Extract features from input_wav to output_csv using opensmile with my93.conf. """
        cmd = self.opensmile + " -noconsoleoutput -C " + self.config + " -I " + input_wav + " -O " + output_csv
        res = subprocess.call(cmd)

    @utils.catch_exception
    def extract(self, x, y, data_dir, scope='train'):
        utils.check_dir(data_dir)
        for wav, label in tqdm.tqdm(zip(x, y), desc='Extracting %s features...' % scope.capitalize()):
            filename = os.path.split(wav)[1]
            self._extract(wav, os.path.join(data_dir, filename.split('.')[0] + '_%s.csv' % label))

    @utils.catch_exception
    def read(self, data_dir):
        """ Read features from csv file extracted by opensmile with my93.conf. """
        result = []
        filenames = os.listdir(data_dir)
        for filename in tqdm.tqdm(filenames, desc='Loading data from %s...' % data_dir):
            df = pd.read_csv(os.path.join(data_dir, filename))
            category = filename.split('.')[0].split('_')[-1]
            result.append([df, category])
        return result


