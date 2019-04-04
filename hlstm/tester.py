from __future__ import division, print_function
import numpy as np
import os, sys, time, math
from collections import defaultdict
import pickle
import h5py
import scipy
import scipy.io.wavfile as siow

import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

# from models import encdec_model
from encdec_model import enc_dec
from loss import LapLoss
from process_data import get_weights
from settings import GS_FIELDNAMES, LOSS_NAMES, FINAL_CRTR, DEVICE, CSVFILENAME, ARCH_DIR
from data_loader import normalize, denormalize, load_test_data, get_data_loader

# SPECT_PATH = "/media/bighdd7/irene/HLSTM/data/WAV_16000/spect/"
# freq_size = 200
# freq_path = SPECT_PATH + 'frame_%d/'%freq_size
# test_file = 'ZzL2sHtTWRc.p'
# data = pickle.load(open(freq_path + test_file, 'r'))
# print(data.shape)
# np.savez('sample', data, data, data, data, data, data, data, data)

config_num = 200

if (config_num == 200):
	# 3 levels
	INPUT_DIM = 100
	N_LEVELS = 3
	HIDDEN_DIMS = [512, 512, 512]
	N_SPLITS = [32, 8, 1]
	SELF_NORM = False
	REVERSE = True
	DROPOUT = 0
	TIME_STEPS = 256

elif (config_num == 400):
	# 1 level
	INPUT_DIM = 100
	N_LEVELS = 1
	HIDDEN_DIMS = [512]
	N_SPLITS = [1]
	SELF_NORM = False
	REVERSE = True
	DROPOUT = 0


weights_path = "/media/bighdd7/irene/HLSTM/results/mosei_results/test_space/config%d/weights.pt"%config_num
model = enc_dec(INPUT_DIM, N_LEVELS, HIDDEN_DIMS, N_SPLITS, DEVICE, SELF_NORM, REVERSE, DROPOUT)
states = torch.load(weights_path)
# print(states['state_dict'].keys())
model.load_state_dict(states['state_dict'])





