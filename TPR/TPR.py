from __future__ import print_function
import numpy as np
from time import time
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from os import path, makedirs

import matplotlib.pyplot as plt

import sys
sys.path.append('../../training_2')
 
from custom_callbacks import *
# from data_generators import *
from data_generators_gaussian import *
from models_node import *

import pandas as pd
import scipy.signal
import tqdm
import time
# func. to find peaks
def find_peaks(preds, threshold=0.9, width=[1200, 3000], mean=0.9 ):
    '''
    preds: 1D numpy array of sigmoid output from the NN
    '''
    test_p = preds
    
    peaks, properties =  scipy.signal.find_peaks(test_p, height=threshold, width =width, distance = 4096*5 )

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

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

data_dir  = '/home/mtian8/combined_5_25/'
# data_dir  = '/home/mtian8/combined_spin/'

# noise_dir = '/home/mtian8/gw_detection-master/data/'
noise_dir ='/home/mtian8/Gaussian_Noise/'

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

## 1. Load Model and Data Generators 
Model = keras.models.load_model('/home/mtian8/gw_detection-master/checkpoint_node/gaussian_16/weights/model_74-0.01783.h5')

for nr in [0.6, 1.0, 1.3, 1.8]:
    
    batch_size = 32

    val_gen = wfGenerator(noise_dir, data_dir, batch_size=batch_size, n_channels=3,
                             shuffle=False, train=0, gaussian = 1, noise_prob=0.0, noise_range=[nr,nr], hvd_rank=0, hvd_size=1)

    preds = Model.predict(val_gen, workers=8, max_queue_size=1000, verbose=1)

#     np.save(f'preds_gaussian_embed_128_spin_14_model_46_nr_{nr}', preds)
    np.save(f'preds_year_gaussian_16_model_74_nr_{nr}', preds)
