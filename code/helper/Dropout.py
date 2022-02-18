#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import numpy as np
  
# setting path
sys.path.append('../model')

from nn_transform import Transform

"""
    Dropout 
"""
class Dropout(Transform):
    """
        p is the dropout probability
    """
    def __init__(self, p=0.5):
        Transform.__init__(self)
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    """
        Apply a mask generated from np.random.binomial
        Scale the output accordingly during testing
        
    """
    def forward(self, x, train=True):
        if train:
            #each hidden layer is turned off with probability p   
            mask = np.random.binomial(1, self.p, x.shape)
            self.mask = mask
            return np.multiply(x, mask)
        #normally, at test time, we replace masks with their expectation
        #geometric average of all possible binary masks. Here we just multiply by constant
        #since expectation of binomial is constant
        else:
            x = x * self.p
        return x

    def backward(self, grad_wrt_out):
        #we only propagate the neurons that were not off
        return self.mask * grad_wrt_out

