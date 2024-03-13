
from __future__ import print_function
import numpy as np
from time import time
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from os import path, makedirs

from custom_callbacks import *
from data_generators_gaussian import *
import matplotlib.pyplot as plt

import numpy as np

import scipy.signal
import glob
import json
import multiprocessing

# func. to make preds
def make_preds(whitened_L1, whitened_H1, whitened_V1, Model):

    # Load Strain
    data = np.stack(( whitened_L1, whitened_H1, whitened_V1), axis=1)

    # Create datagenerators
    dg_0 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=0, batch_size=256)
    dg_5 = TimeseriesGenerator(data=data, targets=data, length=4096, stride=4096, start_index=2047, batch_size=256)
    
    # Make preds
    preds_0 = Model.predict_generator(dg_0, verbose=1)
    preds_5 = Model.predict_generator(dg_5, verbose=1)
    
    return preds_0.ravel(), preds_5.ravel()





# func. to find peaks
def find_peaks(preds, threshold = 0.9, width = [1500,3000], mean = 0.9):
    '''
    preds: 1D numpy array of sigmoid output from the NN
    '''
    test_p = preds
    
    peaks, properties =  scipy.signal.find_peaks(test_p, height=threshold, width = width, distance = 4096*1 )

    left = properties['left_ips']
    right = properties['right_ips']

    f_left = []
    f_right = []
    for i in range(len(left)):
        sliced = test_p[int(left[i]):int(right[i])] 
        if (np.mean(sliced>mean)>mean):
            f_left.append(int(left[i]))
            f_right.append(int(right[i]))
            
    return peaks, f_left, f_right


# data_dir = '/home/mtian8/Gaussian_Noise/gaussian_4096.hdf'

from models import *


Model = keras.models.load_model('/home/mtian8/gw_detection-master/checkpoint_node/gaussian_spin_14_more/weights/model_66-0.01436.h5')

data_dirs = sorted(glob.glob('/home/mtian8/Gaussian_year/gaussian_4096_*.hdf'))

for data_dir in data_dirs:
    # Load strains
    fp = h5py.File(data_dir, 'r')
    strain_L1 = fp['strain_L1'][:]
    strain_H1 = fp['strain_H1'][:]
    strain_V1 = fp['strain_V1'][:]
    fp.close()

    # Make preds
    preds_0, preds_5 = make_preds(strain_L1, strain_H1, strain_V1, Model)

    preds_0 = preds_0.reshape(-1, 4096)
    preds_5 = preds_5.reshape(-1, 4096)

    np.save(f"/home/mtian8/Gaussian_year/preds/preds_model_66/preds_0_{data_dir.split('/')[-1].split('.')[0]}", preds_0)
    np.save(f"/home/mtian8/Gaussian_year/preds/preds_model_66/preds_5_{data_dir.split('/')[-1].split('.')[0]}", preds_5)
