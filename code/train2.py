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

## device and type
dtype = torch.float32
torch.set_default_dtype(dtype )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# Parameters
N = 256
batch_size = 192
database_size = 3e5
n_sub_epoch = int( np.floor( database_size /  batch_size  ))
max_epoch = 100 * n_sub_epoch  # one epoch is 300k images to mimic mscoco size 1.3M image for imagenet
epoch = n_sub_epoch / 100 
std_min = 0 
std_max = 0.3

ckpt_every = 10
log_loss_every = 1
resume_training = True
log_dir = save_path + 'logs/'
checkpoint_dir = save_path + 'checkpoints/'
# prefix_model_name = 'ffdnet_curves_epoch_'
# prefix_log_ct = 'log_ffdnet_curves_'


   
model = FFDNet(num_input_channels=1).to(device)
model.train()
criterion = torch.nn.MSELoss(size_average=False)
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,60,61], gamma=0.1)

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
if resume_training:
    start_epoch, load_data = manager.load()
    model.load_state_dict(load_data["model"])
    optimizer.load_state_dict(load_data["optimizer"])
else:
    start_epoch = 0

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())


print('Loss for max noise level: %f \n' %(batch_size * std_max**2 * N**2))
print('Loss for mean noise level: %f \n' %(batch_size * ((std_max-std_min)/2)**2 * N**2))

for i in range(start_epoch,max_epoch):
    X = generate_batch(batch_size,N,pool=pool).to(device)
    std_noise = (std_max - std_min)*torch.rand(batch_size,device=device)+ std_min
    std_noise = std_noise[:,None,None,None]
    
    noise = std_noise * torch.randn((batch_size,1,N,N),device=device)
    Y = X + noise
    
    optimizer.zero_grad()
    
    outputs = model(Y,std_noise)

    # Compute the loss and its gradients
    loss = criterion(outputs, X)
    loss.backward()

    print('%d -- %f \n' %(i,loss))
    
    if i % log_loss_every == 0:
        with open( '%s/%d.txt' %(log_dir,i) , 'w') as file:
            file.write('%f \n' % loss)

    # Adjust learning weights
    optimizer.step()
       
    
    if i % ckpt_every== 0:
        manager.save(i)
        
pool.close()
        
# plt.plot(range(epoch),cf.detach().cpu(), range(epoch), np.ones(epoch) * batch_size * std_max**2 * N**2, range(epoch), np.ones(epoch) * batch_size * ((std_max-std_min)/2)**2 * N**2)
    
    # model.eval()
    # with torch.no_grad():
    #     X = generate_batch(batch_size,N).to(device)
    #     std_noise = (std_max - std_min)*torch.rand(batch_size,device=device)+ std_min
    #     std_noise = std_noise[:,None,None,None]
        
    #     noise = std_noise * torch.randn((batch_size,1,N,N),device=device)
    #     Y = X + noise
    #     outputs = net(Y,std_noise)
        
    #     plt.subplot(131)
    #     plt.imshow(X[0,0].detach().cpu())
    #     plt.subplot(132)
    #     plt.imshow(Y[0,0].detach().cpu())
    #     plt.subplot(133)
    #     plt.imshow(outputs[0,0].detach().cpu())
    
# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(description="Training FFDNET on simulated curves")
# 	
#     parser.add_argument("--log_dir", type=str, default="logs", \
# 					 help='path of log files')
# 	#Training parameters
#     parser.add_argument("--batch_size", type=int, default=128, 	\
# 					 help="Training batch size")
#     parser.add_argument("--epochs", "--e", type=int, default=80, \
# 					 help="Number of total training epochs")
#     parser.add_argument("--resume_training", "--r", action='store_true',\
# 						help="resume training from a previous checkpoint")
        
#     parser.add_argument("--lr", type=float, default=1e-3, \
# 					 help="Initial learning rate")
        
#     parser.add_argument("--save_every", type=int, default=10,\
# 						help="Number of training steps to log")
                        
#     parser.add_argument("--save_every_epochs", type=int, default=5,\
# 						help="Number of training epochs to save state")
        
#     parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 0.3], \
# 					 help="Noise training interval")
        
#     parser.add_argument("--val_noiseL", type=float, default=0.1, \
# 						help='noise level used on validation set')
        
#     argspar = parser.parse_args()

#     print("\n### Training FFDNet model ###")
#     print("> Parameters:")
#     for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
#         print('\t{}: {}'.format(p, v))
#         print('\n')

#     main(argspar)