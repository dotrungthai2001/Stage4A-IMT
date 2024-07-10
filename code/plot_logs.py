#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:11:50 2023

@author: paul
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

"""
This function is used to plot logs of the trainings on Jean Zay (or on your own machine). 
It allows checking whether the trainings are right or not.
"""

#%%
window_size = 10
directory = 'logs/'
prefix = ''


# #%%
# # Function to extract number2 from a given line of text
# def extract_numbers(line):
#     # Use regular expression to find all numbers on the line
#     floats = re.findall(r'\d+\.\d+', line)
#     return floats 

# losses = []
# filenames = sorted(os.listdir(directory), key=lambda x: int(re.search(r'\d+', x).group()))
# # Iterate over all files in the directory
# for filename in filenames:
#     if filename.startswith(prefix) and filename.endswith('.txt'):
#         file_path = os.path.join(directory, filename)
#         with open(file_path, 'r') as file:
#             for line in file:
#                 numbers = extract_numbers(line)
#                 if numbers is not None and len(numbers)==3:
#                     losses.append(float(numbers[1]))

losses = []
filenames = sorted(os.listdir(directory), key=lambda x: int(re.search(r'\d+', x).group()))
# # Iterate over all files in the directory
for filename in filenames:
    if filename.startswith(prefix) and filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            for line in file:
                losses.append(float(line))
                # numbers = extract_numbers(line)
                # if numbers is not None and len(numbers)==3:
                    # 


# Specify the window size for the moving average
moving_average = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')

losses = np.array(losses)
plt.semilogy(losses,'r')
plt.semilogy(moving_average,'g')
plt.semilogy( 192*256**2 * 0.15**2 * np.ones(moving_average.shape[0]), 'b')
plt.semilogy( 192*256**2 * (0.15/5)**2 * np.ones(moving_average.shape[0]), 'b')
plt.semilogy( 192*256**2 * 0.3**2 * np.ones(moving_average.shape[0]), 'b')
plt.semilogy( 192*256**2 * (0.3/5)**2 * np.ones(moving_average.shape[0]), 'b')
plt.show()