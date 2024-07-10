#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 10:57:52 2023

@author: paul
"""

import torch
import numpy as np
from FFDNET import FFDNet
import matplotlib.pyplot as plt
from ckpt_manager import CheckpointManager
import os
from data_loader import generate_batch
import multiprocessing



# JeanZay filesystem
# save_path = os.environ['WORK'] + '/stage_thai/'
save_path = './'
checkpoint_dir = save_path + 'checkpoints/'

## device and type
dtype = torch.float32
torch.set_default_dtype(dtype )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


# Parameters
N = 256
batch_size = 32
database_size = 3e5
n_sub_epoch = int( np.floor( database_size /  batch_size  ))
max_epoch = 100 * n_sub_epoch  # one epoch is 300k images to mimic mscoco size 1.3M image for imagenet
epoch = n_sub_epoch / 100 
std_min = 0.3
std_max = 0.3
   
model = FFDNet(num_input_channels=1).to(device)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

manager = CheckpointManager(
    assets={
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    },
    directory=checkpoint_dir,
    file_name='model',
    maximum=float("inf"),
    file_format='pt'
)

# Resume training or start anew
start_epoch, load_data = manager.load()
model.load_state_dict(load_data["model"])

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())


X = generate_batch(batch_size,N,pool=pool)

for i in range(X.shape[0]):
    plt.imshow(X[i,0])
    plt.title('%d' % i)
    plt.show()

model.eval()
with torch.no_grad():
    X = generate_batch(batch_size,N,pool=pool).to(device)
    std_noise = (std_max - std_min)*torch.rand(batch_size,device=device)+ std_min
    std_noise = std_noise[:,None,None,None]
    
    noise = std_noise * torch.randn((batch_size,1,N,N),device=device)
    Y = X + noise
    outputs = model(Y,std_noise)
    
    i = 9
    
    plt.subplot(131)
    plt.imshow(X[i,0].detach().cpu())
    plt.subplot(132)
    plt.imshow(Y[i,0].detach().cpu())
    plt.subplot(133)
    plt.imshow(outputs[i,0].detach().cpu())

pool.close()

from helper_functions import read_transparent_png
u = read_transparent_png('vaisseau_photo_3.png')
u = 255 - np.mean(u, axis=2)
u = u[:-1,:-1]
u = torch.tensor(u, dtype = dtype).to(device)[None, None] / 255

def snr(u,v):
    return -10*torch.log10(torch.sum((u-v)**2) / torch.sum(u**2))

batch_size=1
N = u.shape[2]
std_noise = std_max*torch.rand(batch_size,device=device)+ std_min
std_noise = std_noise[:,None,None,None]
noise = std_noise * torch.randn((batch_size,1,N,N),device=device)
Y = (u + noise)

with torch.no_grad():
    output = model(Y,std_noise)
print(snr(u,Y))
plt.subplot(131)
plt.imshow(u[0,0].detach().cpu())
plt.subplot(132)
plt.imshow(Y[0,0].detach().cpu())
plt.subplot(133)
plt.imshow(output[0,0].detach().cpu())
print(snr(u,output))
