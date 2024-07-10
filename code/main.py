#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:53:24 2023

@author: trungthai
"""

import cv2
from simulatevaisseau import *
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# Create a directory to store the images
output_directory_original = 'generated_images_original'
os.makedirs(output_directory_original, exist_ok=True)

# # Create a directory to store the noisy images
# output_directory_noisy = 'generated_images_noisy'
# os.makedirs(output_directory_noisy, exist_ok=True)

# Define the dataset size
dataset_size = 200

# Generate and save the dataset of images
for idx in range(dataset_size):
    n = np.random.randint(5, 8)  # Randomly select n
    n_pts = np.random.randint(100, 200)  # Randomly select n_pts
    # out_pts = np.random.choice(['remove', 'scale', 'limit'])  # Randomly select out_pts
    x = []
    y = []
    
    nb_cou = np.random.randint(2, 4)
    nb_cir = np.random.randint(3, 5)
    for i in range(nb_cou):
        xi, yi = generate_random_shape(n, n_pts, 'remove')
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))
        
    for i in range(nb_cir):
        xi, yi = simulate_circle(50, 50)
        x = np.concatenate((x, xi))
        y = np.concatenate((y, yi))
    
    N = x.shape[0]  # should be (n_controlled - 1) * n_pts
    u = np.ones(N)

    interpolated_array = bilinear_interpolation(u, x, y, 256, 256)

    # Create a PIL image object
    image = Image.fromarray((interpolated_array * 255).astype('uint8'))

    # Save the image to the output directory with a unique filename
    original_filename = os.path.join(output_directory_original, f'image_{idx}.png')
    image.save(original_filename)
    
    # # Add Gaussian noise to the image
    # mean = 0
    # sigma = np.random.uniform(0.05, 0.1)  # Adjust the standard deviation as needed
    # noisy_image = interpolated_array + np.random.normal(mean, sigma, interpolated_array.shape)

    # # Clip values to be in the range [0, 255]
    # noisy_image = np.clip(noisy_image, 0, 1)
    
    # # Create a PIL image object for the noisy image
    # noisy_image = Image.fromarray(noisy_image.astype('uint8'))

    # # Save the noisy image/home/trungthai/Documents/Semettre 8/Stage 4A/2023_07_PAT_Python/generated_images_original/image_0.pngo the noisy images folder
    # noisy_filename = os.path.join(output_directory_noisy, f'image_{idx}.png')
    # noisy_image.save(noisy_filename)

image_path = '/home/trungthai/Documents/Semettre 8/Stage 4A/2023_07_PAT_Python/generated_images_original/image_0.png'  # Replace with the actual path
image = Image.open(image_path)
image_array = np.array(image)
noisy = image_array + np.random.normal(0, 15, image.size)

plt.imshow(image)
plt.imshow(noisy)
