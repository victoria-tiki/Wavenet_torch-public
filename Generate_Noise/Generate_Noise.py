# %pip install pycbc
# %pip install scipy==1.7.3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py 
import csv
import tqdm
import glob
import pickle

from scipy import signal
from scipy.interpolate import interp1d
import json
import matplotlib.mlab as mlab

from pycbc.noise import gaussian
from pycbc.psd import aLIGOZeroDetHighPower, AdvVirgo, read

import scipy

# The PSD needs to have the following properties
flen = 4096*2 + 1  # The number of (complex) samples has to be equal to len(wav) // 2 + 1
delta_f = 1.0/4  # The frequency-distance between two samples has to match the delta_f of the waveform
flow = 19  # The lower frequency cutoff has to be a little smaller than the lower frequency cutoff used to generate the waveform

psd_LH = read.from_txt('/ccs/home/mtian8/aligo_O4high.txt',     length=flen, delta_f=delta_f, low_freq_cutoff=flow, is_asd_file=True)
psd_V  = read.from_txt('/ccs/home/mtian8/avirgo_O5low_NEW.txt', length=flen, delta_f=delta_f, low_freq_cutoff=flow, is_asd_file=True)

delta_t = 1.0 / 4096
tsamples = int(4096/ delta_t)

for i in range(6960):
    train_file = f"/ccs/home/mtian8/Gaussian_decade/gaussian_4096_{i}.hdf"
    # train_file = f"/gpfs/alpine/ast176/scratch/mtian8/Gaussian_test/gaussian_4096_{i}.hdf"

    fp = h5py.File(train_file, "a")
    H1s        = fp.create_dataset("strain_H1", (int(4096*4096),))
    L1s        = fp.create_dataset("strain_L1", (int(4096*4096),))
    V1s        = fp.create_dataset("strain_V1", (int(4096*4096),))

    ts_L = gaussian.noise_from_psd(tsamples, delta_t, psd_LH).numpy()
    ts_H = gaussian.noise_from_psd(tsamples, delta_t, psd_LH).numpy()
    ts_V = gaussian.noise_from_psd(tsamples, delta_t, psd_V).numpy()
    L1s[:]        = (ts_L/np.std(ts_L))
    H1s[:]        = (ts_H/np.std(ts_H))
    V1s[:]        = (ts_V/np.std(ts_V))
    fp.close()