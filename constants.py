# -*- coding: utf-8 -*-
# @Date      : 2021/5/23 8:20 下午
# @Author    : Du Jing
# @Filename  : constants.py
# ---- Description ----
# 全局变量定义


# ================================================================================
# 特征相关变量
# ================================================================================
import enum

FEATURE_SET = r"my93.conf"
FEATURE_NAME = ['voiceProb', 'HNR', 'F0', 'F0raw', 'F0env', 'jitterLocal',
                'jitterDDP', 'shimmerLocal', 'harmonicERMS', 'noiseERMS',
                'pcm_loudness_sma', 'pcm_fftMag_mfcc_sma[0]',
                'pcm_fftMag_mfcc_sma[1]', 'pcm_fftMag_mfcc_sma[2]',
                'pcm_fftMag_mfcc_sma[3]', 'pcm_fftMag_mfcc_sma[4]',
                'pcm_fftMag_mfcc_sma[5]', 'pcm_fftMag_mfcc_sma[6]',
                'pcm_fftMag_mfcc_sma[7]', 'pcm_fftMag_mfcc_sma[8]',
                'pcm_fftMag_mfcc_sma[9]', 'pcm_fftMag_mfcc_sma[10]',
                'pcm_fftMag_mfcc_sma[11]', 'pcm_fftMag_mfcc_sma[12]',
                'pcm_fftMag_mfcc_sma[13]', 'pcm_fftMag_mfcc_sma[14]',
                'pcm_loudness_sma_de', 'pcm_fftMag_mfcc_sma_de[0]',
                'pcm_fftMag_mfcc_sma_de[1]', 'pcm_fftMag_mfcc_sma_de[2]',
                'pcm_fftMag_mfcc_sma_de[3]', 'pcm_fftMag_mfcc_sma_de[4]',
                'pcm_fftMag_mfcc_sma_de[5]', 'pcm_fftMag_mfcc_sma_de[6]',
                'pcm_fftMag_mfcc_sma_de[7]', 'pcm_fftMag_mfcc_sma_de[8]',
                'pcm_fftMag_mfcc_sma_de[9]', 'pcm_fftMag_mfcc_sma_de[10]',
                'pcm_fftMag_mfcc_sma_de[11]', 'pcm_fftMag_mfcc_sma_de[12]',
                'pcm_fftMag_mfcc_sma_de[13]', 'pcm_fftMag_mfcc_sma_de[14]',
                'pcm_fftMag[0]', 'pcm_fftMag[1]', 'pcm_fftMag[2]',
                'pcm_fftMag[3]', 'pcm_fftMag[4]', 'pcm_fftMag[5]',
                'pcm_fftMag[6]', 'pcm_fftMag[7]', 'pcm_fftMag[8]',
                'pcm_fftMag[9]', 'pcm_fftMag[10]', 'pcm_fftMag[11]',
                'pcm_fftMag[12]', 'pcm_fftMag[13]', 'pcm_fftMag[14]',
                'pcm_fftMag[15]', 'pcm_fftMag[16]', 'pcm_fftMag[17]',
                'pcm_fftMag[18]', 'pcm_fftMag[19]', 'pcm_fftMag[20]',
                'pcm_fftMag[21]', 'pcm_fftMag[22]', 'pcm_fftMag[23]',
                'pcm_fftMag[24]', 'pcm_fftMag[25]', 'logMelFreqBand[0]',
                'logMelFreqBand[1]', 'logMelFreqBand[2]',
                'logMelFreqBand[3]', 'logMelFreqBand[4]',
                'logMelFreqBand[5]', 'logMelFreqBand[6]',
                'logMelFreqBand[7]', 'lpcCoeff[0]', 'lpcCoeff[1]',
                'lpcCoeff[2]', 'lpcCoeff[3]', 'lpcCoeff[4]', 'lpcCoeff[5]',
                'lpcCoeff[6]', 'lpcCoeff[7]', 'lspFreq[0]', 'lspFreq[1]',
                'lspFreq[2]', 'lspFreq[3]', 'lspFreq[4]', 'lspFreq[5]',
                'lspFreq[6]', 'lspFreq[7]', 'pcm_zcr']
FEATURE_SAVE_ROOT = "./csv"



