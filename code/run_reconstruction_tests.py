# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:46:21 2023

@author: Caroline, Paul
"""

from algorithms import *
from tools import *
from error_measures import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import numpy as np

from PIL import Image
from helper_functions import read_transparent_png


#load the model
A = load_sparse_csr('Afwd')
normA = svds(A, k=1, return_singular_vectors = False)[0]
capteurs = np.load('sensor_array.npz')
capteurs = capteurs['arr_0']

# define the phantom
N = 636
# im = Image.open('data/vaisseau_photo_64_white.tiff')

u = read_transparent_png('vaisseau_photo_3.png')
u = 255 - np.mean(u, axis=2)
u = u[:-1,:-1]
# u = torch.tensor(u, dtype = dtype).to(device)[None, None] / 255
u = np.array(u,dtype=np.float32) / 255


# u = np.zeros((N,N))
# u[N//2,N//2] = 1



    
# constructing the observations
s = A.dot(u.reshape(-1))
s = np.reshape(s, (A.shape[0]//capteurs.shape[2],capteurs.shape[2]))

# printing some signal
plt.figure()
plt.plot(s[:,0])
plt.title('One observed signal')
plt.show()


# solving the inverse problem
s = A.dot(u.reshape(-1)) 
sigma = 0.01 * np.max(np.abs(s))
s = s + sigma*np.random.randn(s.shape[0])

# Using least squares
recon_ls = moindre_carres(A, s)
recon_ls = np.reshape(recon_ls[0],(N,N))

# measuring the reconstruction quality
print('SNR for least square : '+f'{SNR(u,recon_ls)} dB')
print('####')

# show the results
plt.figure
plt.imshow(recon_ls)
plt.title('Least squares')
plt.show()

# algorithm paramaters
beta = normA**2
gamma = 1.5/beta #gradient step size
n_iter = 100 # iteration number
l = 0.001 #regularisation parameter



# Using non negative least squares
recon_nls, cost_p = moindre_carres_nonneg(A, s, gamma,n_iter)
recon_nls = np.reshape(recon_nls,(N,N))

# measuring the reconstruction quality
print('SNR for Non neg least square : '+f'{SNR(u,recon_nls)} dB')
print('####')

# show the results
plt.figure
plt.imshow(recon_nls)
plt.title('Non neg Least squares')
plt.show()

plt.figure
plt.plot(cost_p)
plt.title('Non neg Least squares criterion')
plt.show()

# l2-l2 optim. pb
recon_l2,cost_l2 = moindre_carres_l2(A,s,gamma,l,n_iter)
recon_l2 = np.reshape(recon_l2,(N,N))

# measuring the reconstruction quality
print('SNR for l2-l2 : '+f'{SNR(u,recon_l2)} dB')
print('####')

# show the results
plt.figure
plt.imshow(recon_l2)
plt.title('l2-l2')
plt.show()

plt.figure
plt.plot(cost_l2)
plt.title('l2-l2 criterion')
plt.show()

# l2-l1 optim. pb
recon_l1,cost_l1 = FBS_l1wave(A,s,gamma,l,N,N,n_iter)
recon_l1 = np.reshape(recon_l1,(N,N))

# measuring the reconstruction quality
print('SNR for l2-l1 : '+f'{SNR(u,recon_l1)} dB')
print('####')

# show the results
plt.figure
plt.imshow(recon_l1)
plt.title('l2-l1')
plt.show()

plt.figure
plt.plot(cost_l1)
plt.title('l2-l1 criterion')
plt.show()

# l2-TV optim. pb
betat = 1
tau = 1/np.sqrt(beta+4*2*betat**2)
sigma = 1/np.sqrt(beta+4*2*betat**2)
epsilon = 0.00001 # stopping criterion
theta = 1

recon_tv,cost_tv,cost_tvd = primal_dual_TV(A, sigma, betat, tau, N, N, s, l, epsilon, n_iter, theta=1)
recon_tv = np.reshape(recon_tv,(N,N))

# measuring the reconstruction quality
print('SNR for l2-tv : '+f'{SNR(u,recon_tv)} dB')
print('####')

# show the results
plt.figure
plt.imshow(recon_tv)
plt.title('l2-tv')
plt.show()

plt.figure
plt.plot(cost_tv)
plt.title('l2-tv criterion')
plt.show()

# l2-Cauchy optim. pb (FB)
recon_c,cost_c = FBS_cauchy(A, s, gamma, l, N, N, 100)
recon_c = np.reshape(recon_c,(N,N))

# measuring the reconstruction quality
print('SNR for l2-Cauchy FB : '+f'{SNR(u,recon_c)} dB')
print('####')

# show the results
plt.figure
plt.imshow(recon_c)
plt.title('l2-Cauchy FB')
plt.show()

plt.figure
plt.plot(cost_c)
plt.title('l2-Cauchy FB criterion')
plt.show()


# l2-Cauchy optim. pb (LBFGS)
recon_clb,cost_clb = LMBFGS(A, s, 100, l, 0.01, N, N, epsilon,normA)
recon_clb = np.reshape(recon_clb,(N,N))

# measuring the reconstruction quality
print('SNR for l2-Cauchy BFGS : '+f'{SNR(u,recon_clb)} dB')
print('####')

# show the results
plt.figure()
plt.imshow(recon_clb)
plt.title('l2-Cauchy BFGS')
plt.show()

plt.figure
plt.plot(cost_clb)
plt.title('l2-Cauchy BFGS criterion')
plt.show()


# l2-Cauchy optim. pb (LBFGS)
reg_grad_cauchy = l
recon_cglb,cost_cglb = LMBFGS_grad(A, s, 100, reg_grad_cauchy, 1e-2, N, N, epsilon,normA)
recon_cglb = np.reshape(recon_cglb,(N,N))

# measuring the reconstruction quality
print('SNR for l2-Cauchy grad BFGS : '+f'{SNR(u,recon_cglb)} dB')
print('####')

# show the results
plt.figure()
plt.imshow(recon_cglb)
plt.title('l2-Cauchy grad BFGS ')
plt.show()

plt.figure()
plt.plot(cost_cglb)
plt.title('l2-Cauchy grad BFGS criterion')
plt.show()


# Plotting the data
plt.plot(cost_p, label='Cost P')
plt.plot(cost_l2, label='Cost L2')
plt.plot(cost_l1, label='Cost L1')
plt.plot(cost_tv, label='Cost TV')
plt.plot(cost_c, label='Cost C')
plt.plot(cost_clb, label='Cost CLB')

# Adding legend
plt.legend()

# Adding labels and title
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Curves')

# Display the plot
plt.show()
