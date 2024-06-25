from __future__ import print_function
import numpy as np
from time import time
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from os import path, makedirs

# from custom_callbacks import *
from data_generators_gaussian import *
import matplotlib.pyplot as plt

import numpy as np

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import scipy.signal
from time import time

import os
from os import path, makedirs
import glob
import json
import multiprocessing





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



# data_dirs = sorted(glob.glob('/home/mtian8/Gaussian_test/gaussian_4096_*.hdf'))
data_dirs = sorted(glob.glob('/home/mtian8/Gaussian_year/gaussian_4096_*.hdf'))
# [0, 0.1, 0.5, 0.9, 0.99,0.9999,0.99999,0.999999,1.0]
for threshold in [0, 0.1, 0.5, 0.9, 0.99,0.9999,0.99999,0.999999,1.0]:
    detections = {}
    for data_dir in data_dirs:
        for width in [2000]:
            preds_0 = np.load(f"/home/mtian8/Gaussian_year/preds/preds_model_81/preds_0_{data_dir.split('/')[-1].split('.')[0]}.npy")
            preds_5 = np.load(f"/home/mtian8/Gaussian_year/preds/preds_model_81/preds_5_{data_dir.split('/')[-1].split('.')[0]}.npy")

            detection_0 = []
            detection_5 = []
            pred_peak_0 = []
            pred_peak_5 = []
            for j in range(preds_0.shape[0]):
                p_0 = preds_0[j]
                peaks_0, f_left_0, f_right_0 =  find_peaks(p_0.flatten(), threshold=threshold, width=[width, 3000], mean=0.95)

                if peaks_0.size != 0 :
                    detection_0.append(1)
                    pred_peak_0.append( f_right_0 )
                else:
                    detection_0.append(0)
                    pred_peak_0.append(-1)

            for k in range(preds_5.shape[0]):
                p_5 = preds_5[k]
                peaks_5, f_left_5, f_right_5 =  find_peaks(p_5.flatten(), threshold=threshold, width=[width, 3000], mean=0.95)

                if peaks_5.size != 0 :
                    detection_5.append(1)
                    pred_peak_5.append( f_right_5 )
                else:
                    detection_5.append(0)
                    pred_peak_5.append(-1) 


        detection = np.concatenate((np.nonzero(detection_0)[0], np.nonzero(detection_5)[0]))
        detection = set(detection)
        detections[data_dir.split('_')[-1].split('.')[0]] = list(detection)
        with open(f'detection_year_14_model_81_{threshold}.pkl', 'wb') as det:
            pickle.dump(detections, det)
