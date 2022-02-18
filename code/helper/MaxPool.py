#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
  
# setting path
sys.path.append('../model')

from helper import *
from cnn_transform import Transform

"""
    MaxPool layer
"""
class MaxPool(Transform):
    """
        filter_shape is (filter_height, filter_width)
        stride is a scalar
    """
    def __init__(self, filter_shape, stride):
        self.filter_height, self.filter_width = filter_shape
        self.stride       = stride
        self.window_size  = self.filter_height * self.filter_width
    
    """
        forward pass of MaxPool
        inputs: (N, C, H, W) where N: num batches , C: channel, H: height, W: weight
        
        keep track of the max indices for backward prop
    """
    def forward(self, inputs):
        N, C, H, W     = inputs.shape
        self.X_shape   = inputs.shape
        #this returns [c*filter_shape, H*W*N]
        im2col_mtx     = im2col(inputs, self.filter_height, self.filter_width, padding=0, stride=self.stride)
        #gets the max of each feature after splitting the input col into number of channels
        #keeps track of maximum indices
        def getMax(col, C):
            #split into channels
            c_features = np.array(np.split(col, C))  #get max of each row
            maxs       = np.max(c_features, 1) #get argmax
            argmaxs    = np.argmax(c_features, 1)
            return [maxs, argmaxs]
            
        maxPool        = np.array(np.apply_along_axis(getMax, 0, im2col_mtx, C))
        self.maxIdx    = maxPool[1, :, :].astype(int)
        maxPool        = maxPool[0, :, :]
        
        #reshape to N, C, Hout, Wout
        H_out, W_out   = calculate_shape_out(H, W, self.filter_height, self.filter_width, self.stride)
        maxPool        = im2col_inv(maxPool, (N, C, H_out, W_out), channel_order = True)
        return maxPool
    
    """
        dloss is the gradients wrt the output of forward()
        
        backprop is 1 for the max indices 0 for the rest
    """     
    def backward(self, dloss):
        N, C, H, W      = self.X_shape
        dloss           = im2col(dloss, 1, 1, padding=0, stride=1)
        #expand each feature to original window size. Each feature is represented in columns
        dloss           = np.repeat(dloss, self.window_size, axis = 0)
        #expand maxIndices to match window size, only the index location is 1, rest is 0
        #maxIndices is currently ordered as C,N*H*W (rows represent max indices)
        self.maxIdx    += np.arange(0, self.window_size * C, self.window_size).reshape(C,-1)
        self.maxIdx    += np.arange(0, self.window_size * C * dloss.shape[1],(self.window_size * C)) #account for column continuity
        self.maxIdx     = self.maxIdx.reshape(-1, order='F') #flatten
        tmp             = np.zeros(self.window_size * C * dloss.shape[1])
        tmp[self.maxIdx]= 1
        self.maxIdx     = tmp.reshape((self.window_size * C, -1), order='F')
        grad_x_out      = self.maxIdx * dloss    
        grad_x_out      = im2col_bw(grad_x_out, self.X_shape, self.filter_height, self.filter_width, padding=0, stride=self.stride)
        return grad_x_out

