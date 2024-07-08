import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy
from tqdm import tqdm

import sys
sys.path.append("/u/amatchev/Wavenet_torch/")
from models_torch_2channel import *
from data_generators_torch import *


#custom sampler        
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, length, stride=1, start_index=0, end_index=None):
        self.data = data
        self.targets = targets
        self.length = length
        self.stride = stride
        
        if end_index is None or end_index > len(data):
            end_index = len(data) - 1

        self.start_index = start_index
        self.end_index = end_index
        
        self.sample_indices = np.arange(self.start_index, self.end_index - self.length + 1, self.stride)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx = self.sample_indices[idx]
        end_idx = start_idx + self.length

        # Slice the data and targets for the current sample
        sample_data = self.data[start_idx:end_idx]
        sample_target = self.targets[start_idx:end_idx]

        return sample_data.float(), sample_target.float()


def normalize(strain):
    std = np.std(strain[:])
    strain[:] /= std
    return strain
            
class InferenceConfig:
    batch_size = 4
    feb_data_dir = '/scratch/bbke/victoria/WaveNet_data/Feb_Events/'
    checkpoint_dir = '/u/amatchev/Wavenet_torch/checkpoints/'
    noise_dir = '/scratch/bbke/victoria/WaveNet_data/Gaussian_Noise/'
    n_channels = 3
    length = None

# Create an instance of the configuration
inference_args = InferenceConfig()

# Load the checkpointed state_dict
checkpoint_path = os.path.join(inference_args.checkpoint_dir, '2channelcontd_epoch=66-val_loss=0.01677.ckpt')
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'] 
# Remove the "model." prefix from keys
print(checkpoint_path)
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("model."):
        new_key = key[len("model."):]  # Remove the "model." prefix
    else:
        new_key = key
    new_state_dict[new_key] = value
# Load the modified state_dict into the model
model = full_module()
model.load_state_dict(new_state_dict)

# Move the model to GPU if available
if torch.cuda.is_available():
    model = model.to("cuda")

def make_preds(whitened_L1, whitened_H1, whitened_V1, model, length=None):
    normalized_L1 = normalize(whitened_L1)
    normalized_H1 = normalize(whitened_H1)
    normalized_V1 = normalize(whitened_V1)
    if length is not None:
        whitened_L1 = torch.tensor(whitened_L1[:length])
        whitened_H1 = torch.tensor(whitened_H1[:length])
        whitened_V1 = torch.tensor(whitened_V1[:length])
    else:
        whitened_L1 = torch.tensor(whitened_L1)
        whitened_H1 = torch.tensor(whitened_H1)
        whitened_V1 = torch.tensor(whitened_V1)    # Load Strain
    data = torch.stack((whitened_L1, whitened_H1, whitened_V1), dim=1)

    
    dataloader_0 = TimeSeriesDataset(data=data, targets=data, length=4096, stride=4096, start_index=0)
    dataloader_5 = TimeSeriesDataset(data=data, targets=data, length=4096, stride=4096, start_index=2047)
    
    dataloader_0 = DataLoader(dataloader_0, batch_size=256, shuffle=False)
    dataloader_5 = DataLoader(dataloader_5, batch_size=256, shuffle=False)
    
    model.eval()

    # Make predictions
    preds_0 = []
    for inputs in tqdm(dataloader_0, desc='Predicting 0', leave=True, disable=False):
        with torch.no_grad():
            outputs = model(inputs[0])
            preds_0.append(outputs.detach().numpy())
    preds_0 = np.concatenate(preds_0)


    preds_5 = []
    for inputs in tqdm(dataloader_5, desc='Predicting 5', leave=True, disable=False):
        with torch.no_grad():
            outputs = model(inputs[0])
            preds_5.append(outputs.detach().numpy())
    preds_5 = np.concatenate(preds_5)
    
    preds_0,preds_5=preds_0.ravel(), preds_5.ravel()
    return preds_0,preds_5

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

# func. to merge windows
def merge_windows(triggers_0, triggers_5):

    triggers = {}
    
    for key in triggers_0.keys():

        right_0 = triggers_0[key]
        right_5 = triggers_5[key]

        combined = right_0.copy()
        
        for r_5 in right_5:
            keep = True
            for r_0 in right_0:
#                 if abs(r_5 - r_0) < 1:
                if abs(r_5 - r_0) == 1/2:
                    keep = False
            if keep:
                combined.append(r_5)
        
        triggers[key] = combined
    
    return triggers
 
# Convert right ips to GPS times and merge the two windows
def get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048):#, threshold=threshold, width=[width, 3000], mean=0.95):
    
    triggers_0, triggers_5 = {}, {} 
    key = 'detection'
        
    peak_0, left_0, right_0 = find_peaks(preds_0, threshold=threshold, width=[width, 3000], mean=0.95)
    peak_5, left_5, right_5 = find_peaks(preds_5, threshold=threshold, width=[width, 3000], mean=0.95)

    triggers_0[key] = [(x + truncation)/4096 for x in right_0]  
    triggers_5[key] = [(x + truncation + window_shift)/4096 for x in right_5]
    
    # Merge the two windows
    triggers = merge_windows(triggers_0, triggers_5)
    
    return triggers


data_dirs = sorted(glob.glob(f"{inference_args.feb_data_dir}/GW*.hdf5"))

for threshold in [0.1,0.5,0.8,0.9]:#[0.9999]:
    print(f"\n\n #### THRESHOLD OF {threshold} ####", flush=True)	
    for width in [1000,2000]:#[2000]:
        print('width: ',width, flush=True)
        for data_dir in data_dirs:
            print('\ndataset',data_dir.split('/')[-1].split('_')[0], flush=True)
        
            for i in [1]:#range(len(H1_files)):
                
                # Load strains
                fp = h5py.File(data_dir, 'r')
                strain_L1 = fp['strain_L1'][:]
                strain_H1 = fp['strain_H1'][:]
                strain_V1 = fp['strain_V1'][:]
                
                
                # Make preds
                start_time = time.time()
                preds_0, preds_5 = make_preds(strain_L1, strain_H1, strain_V1, model, inference_args.length)
                elapsed_time = time.time() - start_time
                print(f"Time to make predictions: {elapsed_time:.2f} seconds", flush=True)

                # Post Process and find triggers
                start_time = time.time()
                triggers = get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048)
                elapsed_time = time.time() - start_time
                print(f"Time for post-processing: {elapsed_time:.2f} seconds", flush=True)    

                print(data_dir.split('/')[-1].split('_')[0], triggers, flush=True)
        