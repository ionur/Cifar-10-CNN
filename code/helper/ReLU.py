#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import numpy as np
  
# setting path
sys.path.append('../model')

from nn_transform import Transform

"""
    implements ReLU activation function
"""
class ReLU(Transform):
    def __init__(self):
        Transform.__init__(self)

    """
        x if x bigger than 0, 0 otherwise
    """
    def forward(self, x, train=True):
        self.gradient = (x > 0)*1
        return np.maximum(0,x)

    """
        1 if x bigger than 0, 0 otherwise
    """
    def backward(self, grad_wrt_out):
        return (self.gradient * grad_wrt_out)

