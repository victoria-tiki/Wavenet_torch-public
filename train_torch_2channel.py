#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from time import time
import os
import sys
import gc
#import cProfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import LightningDataModule

from models_torch_2channel import *
from data_generators_torch import *
# from custom_callbacks_torch import *

print(torch.__version__)

import argparse

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

class LightningModel(LightningModule):
    def __init__(self, lr):
        super(LightningModel, self).__init__()
        self.model = full_module()
        self.lr = lr
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-10), #verbose=true
                'monitor': 'val_loss',  
                'interval': 'epoch',  
                'frequency': 1,  
                'strict': True,
            }
        }

    def training_step(self, batch, batch_idx):
        inputs, targets = batch  
        outputs = self.model(inputs)
        loss = F.binary_cross_entropy(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss
        
    def on_train_epoch_end(self, unused=None):  
        # Increment the epoch counter of training dataset at the end of each epoch
        self.trainer.datamodule.train_dataset.increment_epoch()
        
def main():
    print("Parsing arguments")

    parser = argparse.ArgumentParser(description="gw detection")

    parser.add_argument('--batch_size', type=int, help='batch size', default=32)#original default=32
    parser.add_argument('--data_dir', help='root directory', default='/scratch/bcbw/amatchev/data/WaveNet_data/combined_spin/') #spin waveform
    parser.add_argument('--checkpoint_dir', help='root directory', default='/scratch/bcbw/amatchev/training/WaveNet_checkpoints/')
    parser.add_argument('--resume_model', help='model directory', default=None)
    parser.add_argument('--noise_dir', help='noise directory', default='/scratch/bcbw/amatchev/data/WaveNet_data/Gaussian_Noise/')
    parser.add_argument('--n_channels', type=int, help='number of channels', default=2)
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=1)
    parser.add_argument('--num_nodes', type=int, help='number of nodes', default=1)
    parser.add_argument('--lr_init', type=float,help='initial learning rate', default=0.001)

    args = parser.parse_args()

    print("Callbacks")
    #callbacks
    RegularModelCheckpoints=ModelCheckpoint(dirpath=args.checkpoint_dir,filename='2channelcontd_{epoch:02d}-{val_loss:.5f}',monitor='val_loss',mode='min',save_top_k=-1)
    StopCriteria=EarlyStopping(monitor='val_loss',patience=7, verbose=True, mode='min')
    callbacks=[RegularModelCheckpoints,LearningRateMonitor(logging_interval='epoch'),]
    
    devices=torch.cuda.device_count()
    devices=devices if devices!=0 else 4 
    
    
    #profiler = cProfile.Profile()
    #profiler.enable()

    print("Defining trainer")
    # define trainer
    trainer = Trainer(
        max_epochs=100,
        num_nodes=args.num_nodes,devices=devices,accelerator="gpu", strategy="ddp",     
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )

    print("Defining model")
    #define model
    model = LightningModel(lr=args.lr_init)
    
    print("Loading data generators")
    # Load Data Generators
    data_module = WaveformDataModule(args.noise_dir, args.data_dir, batch_size=args.batch_size,n_channels=args.n_channels,gaussian=1,noise_prob=0.7, noise_range=None, num_workers=args.num_workers)

    print("Training model")
    # train
    if args.resume_model:
        trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_model)
    else:
        trainer.fit(model,datamodule=data_module)
    t1 = time()

    print('**Evaluation time: %s' % (t1 - t0))
    
    #profiler.disable()
    #profiler.print_stats()


if __name__ == '__main__':
    main()
