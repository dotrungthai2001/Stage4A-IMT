#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:25:50 2023

@author: trungthai
"""

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
import random
import cv2
import scipy
import torch

def read_transparent_png(filename): # Code take from https://stackoverflow.com/questions/3803888/how-to-load-png-images-with-4-channels
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)  

def random_points1D_withmindst(low_bound, up_bound, mindst, npoints):
    space = up_bound - low_bound - (npoints-1)*mindst
    assert space >= 0 
    
    coef = np.random.rand(npoints)
    a = space * np.sort(coef, axis=-1)
    return np.ceil(low_bound + a + mindst * np.arange(npoints)).astype(int)


def bernstein(n, i, t):
    return binom(n,i)*(1.-t)**(n-i)*t**i 

def bezier_explicit(controlled_points, num=200):
    N = len(controlled_points)
    t = np.linspace(0, 1, num=num)
    bezier_curve = np.zeros((num, 2))
    for i in range(N):
        bezier_curve += np.outer(bernstein(N - 1, i, t), controlled_points[i])
    return bezier_curve

def sort_counterClockWise(a):
    #sort an array of 2D points by counter clock wise order
    #center point is the mean of all points
    dst_from_mean = a-np.mean(a,axis=0)
    
    #find the angle between each point and the center point
    #chosing right quadrant with arctan2(x1, x2) for counter-clockwise order
    angle = np.arctan2(dst_from_mean[:,0], dst_from_mean[:,1])
    
    #index that would sort angle array from low to high value 
    #in case of tie, it follows the index of points in initial array
    index_sort = np.argsort(angle)
    
    return a[index_sort,:]

def interpolate_bezier(a,n_pts, radius_coef=0.2, angle_combine_factor = 0.5):
        
    #sort counterclockwise 
    a = sort_counterClockWise(a)
    
    
    #calculate angles linking adjacent points
    adjacent_vect = np.diff(a, axis=0)
    angle = np.arctan2(adjacent_vect[:,1],adjacent_vect[:,0]) #angle between -pi and pi
    translate_angle = lambda angle : (angle>=0)*angle + (angle<0)*(angle+2*np.pi)
    angle = translate_angle(angle) #angle between [0, 2*pi[
    
    #caculate the angle as the mean of adjacent angles
    angle_before = angle
    angle_after = np.roll(angle,1)
    angle = angle_combine_factor*angle_before + (1-angle_combine_factor)*angle_after + (np.abs(angle_after-angle_before) > np.pi )*np.pi
    
    #add first point to last index to have a cycle
    angle = np.append(angle, [angle[0]])
    
    a = np.append(a, np.atleast_2d(angle).T, axis=1)
    
    #from the angles defined, calculate radius and get bezier curves cubic
    spline_segments = []
    for i in range(len(a)-1):
        start_p = a[i,:2]
        end_p = a[i+1,:2]
        
        angle_before = a[i,2]
        angle_after = a[i+1,2]
        
        #calculate radius of intermediate points
        dst = np.sqrt(np.sum((end_p - start_p)**2))
        radius = radius_coef*dst
        
        #calculate intermediate points with radius and angle known
        intermed_p_1 = start_p + np.array([radius*np.cos(angle_before),
                                    radius*np.sin(angle_before)])
        intermed_p_2 = end_p + np.array([radius*np.cos(angle_after+np.pi),
                                    radius*np.sin(angle_after+np.pi)])
        
        #get bezier curves of degree 3 with all 4 controlled points
        controlled_points = np.array([start_p, intermed_p_1, intermed_p_2, end_p])
        segment_curve = bezier_explicit(controlled_points, num=n_pts)
        
        spline_segments.append(segment_curve)
        
    curve = np.concatenate([segment_curve for segment_curve in spline_segments])
    
    x,y = curve.T
    return x,y,a

def generate_randompoints2D_withmindistance(npoints=10, scaling_factor=1, nitermax=200):
    niter=0
    dst_min = 0.7/npoints
    while niter<nitermax:
        a = np.random.rand(npoints,2)
        dst_array = np.sqrt(np.sum(np.diff(sort_counterClockWise(a), axis=0)**2, axis=1))
        if np.all(dst_array >= dst_min):
            return a*scaling_factor
        niter+=1
    return a*scaling_factor

    
def read_transparent_png(filename): # Code take from https://stackoverflow.com/questions/3803888/how-to-load-png-images-with-4-channels
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)  


# x,y are assumed to be in [0,1]
def bilinear_interpolation(u,x,y,height, width):
    im = np.zeros((height,width))
    normalization = np.zeros(im.shape)
    
    xx = (height-2)*x
    yy = (width-2)*y
    lox, loy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    hix, hiy = lox+1, loy+1
    
    np.add.at(im, (lox,loy), (hix-xx)*(hiy-yy)*u)
    np.add.at(im, (lox,hiy), (hix-xx)*(yy-loy)*u)
    np.add.at(im, (hix,loy), (xx-lox)*(hiy-yy)*u)
    np.add.at(im, (hix,hiy), (xx-lox)*(yy-loy)*u)
       
    np.add.at(normalization, (lox,loy), (hix-xx)*(hiy-yy))
    np.add.at(normalization, (lox,hiy), (hix-xx)*(yy-loy))
    np.add.at(normalization, (hix,loy), (xx-lox)*(hiy-yy))
    np.add.at(normalization, (hix,hiy), (xx-lox)*(yy-loy))

    ind = normalization > 1e-6
    im[ind] = im[ind] / normalization[ind]
    return im


def bilinear_interpolation_torch(u, x, y, height, width):
    im = torch.zeros((height, width), dtype=u.dtype)
    normalization = torch.zeros_like(im)

    xx = (height - 2) * x
    yy = (width - 2) * y
    lox, loy = torch.floor(xx).to(torch.int), torch.floor(yy).to(torch.int)
    hix, hiy = lox + 1, loy + 1
    
    im = im.reshape(-1)
    normalization = normalization.reshape(-1)
    ind1 = width*lox + loy
    ind2 = width*lox + hiy
    ind3 = width*hix + loy
    ind4 = width*hix + hiy
    
    w1 = (hix - xx) * (hiy - yy)
    w2 = (hix - xx) * (yy - loy)
    w3 = (xx - lox) * (hiy - yy)
    w4 = (xx - lox) * (yy - loy)

    im.index_add_(0, ind1, w1 * u)
    im.index_add_(0, ind2, w2 * u)
    im.index_add_(0, ind3, w3 * u)
    im.index_add_(0, ind4, w4 * u)

    normalization.index_add_(0, ind1, w1)
    normalization.index_add_(0, ind2, w2)
    normalization.index_add_(0, ind3, w3)
    normalization.index_add_(0, ind4, w4)

    ind = normalization > 1e-6
    im[ind] = im[ind] / normalization[ind]
    im = im.reshape((height,width))
    return im