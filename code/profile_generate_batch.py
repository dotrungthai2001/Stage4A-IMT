"""
Created on Thu Sep 21 10:57:52 2023

@author: paul
"""

import torch
from FFDNET import FFDNet
import matplotlib.pyplot as plt
from ckpt_manager import CheckpointManager
import os
import multiprocessing
from functools import partial
import time
from data_loader import generate_batch


## device and type
dtype = torch.float32
torch.set_default_dtype(dtype )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


# Parameters
N = 256
batch_size = 192

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

tic = time.perf_counter()
for i in range(10):
    X = generate_batch(batch_size,N,pool=pool)
toc = time.perf_counter()
print(f"batch generated in {toc - tic:0.4f} seconds")
pool.close()


