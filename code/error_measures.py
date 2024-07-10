# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:49:05 2023

File containing all the measures used to assess the reconstruction quality

@author: Caroline, Paul
"""

import numpy as np

# Error measures

def SNR(xref,xest): 
    return 10*np.log10(np.linalg.norm(xref)**2/np.linalg.norm(xest-xref)**2)

# Contrast measure