#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
  
# setting path
sys.path.append('../helper')

from helper import *
from cnn_transform import Transform

"""
    implements Convolution Layer
"""
class Conv(Transform):
    """
        input_shape is a tuple: (channels, height, width)
        filter_shape is a tuple: (num of filters, filter height, filter width)
        weights shape (number of filters, number of input channels, filter height, filter width)
        Use Xavier initialization for weights
        Initialze biases as an array of zeros in shape of (num of filters, 1)
    """
    def __init__(self, input_shape, filter_shape, rand_seed=0):
#         np.random.seed(rand_seed) # keep this line for testing
        
        self.C, self.H, self.Width                    = input_shape
        self.num_filters, self.k_height, self.k_width = filter_shape
        self.window_size                              = self.k_height * self.k_width
        b      = np.sqrt(6) / np.sqrt((self.C + self.num_filters) * self.k_height * self.k_width)
        self.W = np.random.uniform(-b, b, (self.num_filters, self.C, self.k_height, self.k_width))
        self.b = np.zeros((self.num_filters, 1))
        
        self.mm     = { 'W': np.zeros(self.W.shape), 'b': np.zeros(self.b.shape) }

    """
        Forward pass of convolution between input and filters
        inputs is in the shape of (batch_size, num of channels, height, width)
        Return the output of convolution operation in shape (batch_size, num of filters, height, width)
    """
    def forward(self, inputs, stride=1, pad=2):
        self.X_shape    =inputs.shape
        self.stride     = stride
        self.pad        = pad
        self.batch_size = inputs.shape[0]
        self.inputs     = im2col(inputs, self.k_height, self.k_width, pad, stride)
        filters         = self.W.reshape(self.num_filters,-1)
        feat_size       = filters.shape[1]   
        #for each filter, multiply the flattened filter feature with the input feature of each window
        def helper(row, args):
            feat_size   = args
            return sum(self.inputs * row.reshape(feat_size, -1))
        #colvolve with each filter and add filter bias
        convolved       = np.apply_along_axis(helper,1, filters, (feat_size) ) + self.b
        out_shape       = (self.batch_size, self.num_filters, self.H, self.Width)
        convolved       = im2col_inv(convolved, out_shape, channel_order = True)
        return convolved
        
    """
        dloss shape (batch_size, num of filters, output height, output width)
        Return [gradient wrt weights, gradient wrt biases, gradient wrt input to this layer]
    """    
    def backward(self, dloss):
        filters    = self.W.reshape(self.num_filters,-1)
        dloss      = im2col(dloss, 1, 1, padding=0, stride=1)
        #if received col then get dx
        #if receiving row then get dW
        def helper(inp, args):
            filters, col = args
            if col == True:
                return np.sum(np.multiply(filters, inp.reshape((self.num_filters,1))), axis = 0)
            else:
                dW = np.sum(np.multiply(self.inputs, inp), axis = 1)
                #add b's as the last column
                db = np.sum(inp).reshape(1,1)
                return np.append(dW,db)
                
        #apply the function to each patch that is represented in a column
        dx = np.apply_along_axis(helper, 0, dloss, (filters, True))
        dx = im2col_bw(dx, self.X_shape, self.k_height, self.k_width, padding=self.pad, stride=self.stride)
        
        #apply to each row of dloss
        #each ith column in the return result represents the dL/dWi
        dW = np.apply_along_axis(helper, 1, dloss, (filters, False))
        db = dW[:, -1].reshape(self.num_filters, 1)
        dW = dW[:,:-1].reshape((self.num_filters, self.C, self.k_height, self.k_width))
        
        self.dW = dW
        self.db = db
        return dW, db, dx
    
    """
        Update weights and biases with gradients calculated by backward()
        Divide gradients by batch_size (because using sum Loss instead of
          mean Loss during backpropogation)
    """
    def update(self, learning_rate=0.001, momentum_coeff=0.5):
        dW           = self.dW / self.batch_size
        db           = self.db / self.batch_size
        
        self.mm['W'] = (momentum_coeff * self.mm['W']) + dW
        self.mm['b'] = (momentum_coeff * self.mm['b']) + db
        dW           = self.W - (learning_rate * self.mm['W'])
        db           = self.b - (learning_rate * self.mm['b'])
        self.W       = dW  
        self.b       = db

    """
        Return weights and biases
    """
    def get_wb_conv(self):
        return self.W, self.b

