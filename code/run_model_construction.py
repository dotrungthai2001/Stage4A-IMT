# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:19:08 2022

File to construct the model

@authors: Caroline, Paul
"""

import numpy as np
from PAT import *
from tools import *
import matplotlib.pyplot as plt

# phantom loading
N = 636
u = np.zeros((N,N))
u[N//2,N//2] = 1

# general parameters 
c = 1500 #m/s
L = 30e-3
Fs = 50e6 # frequency of the sensor in Hz

# sensor parameters
R = 25e-3 # circle radius
Ntrans = 51 # number of punctual transducers
theta = np.linspace(0, np.pi, Ntrans) # can be reduced to [0,pi] for limited view
capteurs = np.zeros((2, 1, Ntrans))

capteurs[0,:,:] = R*np.cos(theta)
capteurs[1,:,:] = R*np.sin(theta)

# parameters 
c = 1500 #m/s
L = 30e-3
Fs = 50e6 # frequenxy of the sensor in Hz
R=25e-3

# sensor_width = 7.5e-3
# thetaMax= 2*np.arctan(sensor_width/(2*R))
# Dx = L / (N-1)
# discCapteur = int(20*np.floor(sensor_width/Dx))
# x_transducer = capteur_cercle(R, thetaMax, discCapteur)
# capteurs = np.zeros((20,x_transducer.shape[0],2))
# capteurs[0,:,:] = x_transducer 
# capteurs[1,:,:] = x_transducer 
# capteurs[0,:,:] = capteurs[0,:,:] + 25e-3



# capteurs = systeme_jerome(x_transducer, np.pi/8, 11, 0, 10e-3, 21)
# x_capteurs = np.zeros((2,1))
# capteurs = np.zeros((2,2,2))
capteurs[:,:,0] = capteurs[:,:,0] + 25e-3

# creating matrix
tstart = 5e-3/c
tend = 50e-3/c

A = PAT(capteurs, L, N, c, Fs, tstart, tend)

# saving the sensors
np.savez('sensor_array', capteurs)

# saving the model A
save_sparse_csr('Afwd', A)
    