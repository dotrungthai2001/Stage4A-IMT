#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:59:17 2023

@author: trungthai
"""

import numpy as np
import matplotlib.pyplot as plt
from helper_functions import *


def outside_pts_handling(xi,yi, out_pts = 'remove'):
    if out_pts == 'limit':
        xi = np.maximum(np.minimum(xi,1),0)
        yi = np.maximum(np.minimum(yi,1),0)
    elif out_pts == 'scale':
        xi = xi / np.maximum(np.max(xi),1)
        yi = yi / np.maximum(np.max(yi),1)
    elif out_pts == 'remove':
        ind1 = np.logical_or((xi < 0), (yi < 0))
        ind2 = np.logical_or((xi > 1), (yi > 1))
        ind = np.logical_or(ind1, ind2)
        xi = np.delete(xi, ind)
        yi = np.delete(yi, ind)
        
    return xi,yi

# Create the circle:
def simulate_circle(Nr=50, Nt=50, r_min=0.1, r_max=0.2, out_pts = 'remove'):
    rad = np.random.uniform(r_min,r_max)
    r = np.linspace(0,rad,Nr)
    theta = np.linspace(0, 2*np.pi, Nt)
    R,T=np.meshgrid(r,theta)

    center = np.random.uniform(0,1,2)
    X = center[0] + R*np.cos(T)
    Y = center[1] + R*np.sin(T)
    
    xi = X.reshape(-1)
    yi = Y.reshape(-1)
    
    xi,yi = outside_pts_handling(xi,yi, out_pts)
    return xi,yi

##
# out_pts: what to do with points outside of [0,1] {remove, scale, limit}
#           - remove: discard points 
#           - scale the whole curve to fit in [0,1]
#           - limit the points outside to be 0 or 1
def generate_random_shape(n, n_pts=200, out_pts = 'remove'):
    a = generate_randompoints2D_withmindistance(n, scaling_factor=1)
    rad = np.random.choice([0.2, 0.3, 0.4])

    xi, yi = interpolate_bezier(a, n_pts, radius_coef=rad, angle_combine_factor=0.4)[:2]
    
    xi,yi = outside_pts_handling(xi,yi, out_pts)
    return xi, yi

def thicken_curve(x, y, line_width = 1, grid_size = 256, out_pts = 'remove'):
    n = x.shape[0]
    xx = np.repeat(x, line_width**2)
    yy = np.repeat(y, line_width**2)
    
    rx,ry = np.meshgrid(np.arange(0,line_width),np.arange(0,line_width))
    xx = xx + np.tile(rx.reshape(-1),n) / grid_size
    yy = yy + np.tile(ry.reshape(-1),n) / grid_size
    
    xx,yy = outside_pts_handling(xx,yy, out_pts)
    
    return xx, yy

if __name__ == '__main__':
    n_pts = 500 
    n_controlled = 10
    x1, y1 = generate_random_shape(10, n_pts)
    
    xx1,yy1 = thicken_curve(x1, y1, 3, 128)
    # x2, y2 = simulate_circle(500,500)
    
    # x = np.concatenate((xx1,x2))
    # y = np.concatenate((yy1,y2))
    x = xx1
    y = yy1
    
    
    N = x.shape[0] # should be (n_controlled - 1)*n_pts  
    u = np.ones(N)
    
    interpolated_array = bilinear_interpolation(u,x,y, 128, 128)

    # Add Gaussian noise to the image
    mean = 0
    sigma = 0.1  # Adjust the standard deviation as needed
    noisy_image = interpolated_array + np.random.normal(mean, sigma, interpolated_array.shape)
    
    im = interpolated_array
    # line_width = 3
    # im = scipy.ndimage.grey_dilation(interpolated_array,size=(line_width,line_width))
    plt.imshow(im)
    plt.show()
    plt.imshow(noisy_image)    
    plt.show()