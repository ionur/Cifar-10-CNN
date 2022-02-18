#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import numpy as np
  
# setting path
sys.path.append('../helper')

from helper import *
from nn_transform import Transform

"""
       implements linear layer y = wx + b
"""
class LinearMap(Transform):
    """
        indim     : input dimension
        outdim    : output dimension
        alpha     : parameter for momentum updates
        lr        : learning rate
        network   : whether NN or CNN
        
        Use Xavier initialization for weights
    """
    def __init__(self, indim, outdim, network, alpha=0, lr=0.01):
        Transform.__init__(self)
#         np.random.seed(0) #keep this line for testing
        self.dims    = { 'indim' : indim, 
                        'outdim': outdim 
                      }
        self.params  = { 'alpha' : alpha, 
                        'lr'    : lr, 
                        'W'     : random_weight_init(indim, outdim),
                        'b'     : zeros_bias_init(outdim)
                      }
                
        self.mm      = { 'W': zeros_weight_init(indim, outdim), 'b': zeros_bias_init(outdim) }
        self.grads   = { 'W': zeros_weight_init(indim, outdim), 'b': zeros_bias_init(outdim) }
        
        self.network = network
        
    """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
    """
    def forward(self, x):
        self.batch_size,_  = x.shape
        self.x = x
        return np.matmul(x, self.params['W']) + self.params['b'].T

    """
        dloss  shape (batch_size, outdim)
        return shape (batch_size, indim)
    """
    def backward(self, dloss):
        """
            calculates for weight i
            input col represents the loss for particular output neuron for all batches
            multiplies the loss with the input, sum along the columns give for w1i, w2i, w3i
            which is the gradient for ith weight vector, so transpose
        """
        def helper_gradW(col):
            col = np.array(col).reshape(col.shape[0],1)
            tmp = np.sum(np.multiply(self.x, col), axis = 0)
            return np.transpose(tmp)
        
        #calculate gradients
        #get the columns
        self.grads['W'] = np.apply_along_axis(helper_gradW, 0, dloss)
        
        #calculate gradients b
        #for each example db = grad_wrt_out, so sum along columns
        self.grads['b'] += np.sum(dloss, axis = 0).reshape((self.dims['outdim'], 1))
        
        dx = np.transpose(np.dot(self.params['W'], np.transpose(dloss)))
        
        if self.network == "NN":
            return dx
        else:
            return [self.grads['W'], self.grads['b'], dx] 

    """
        parameter update with momentum
    """
    def step(self):
        """
        divide gradients by batch_size if CNN because using sum Loss instead of mean Loss
        """
        if self.network == "CNN":
            self.grads['W'] = self.grads['W'] / self.batch_size
            self.grads['b'] = self.grads['b'] / self.batch_size
            
        self.mm['W']       = (self.params['alpha'] * self.mm['W']) + self.grads['W']
        self.mm['b']       = (self.params['alpha'] * self.mm['b']) + self.grads['b'] 
    
        dW                 = self.params['W'] - (self.params['lr'] * self.mm['W'])
        db                 = self.params['b'] - (self.params['lr'] * self.mm['b'])
        self.params['W']   = dW
        self.params['b']   = db

    def zerograd(self):
    # reset parameters
        self.grads['W']    = zeros_weight_init(self.dims['indim'],self.dims['outdim'])
        self.grads['b']    = zeros_bias_init(self.dims['outdim'])

    def getW(self):
    # return weights
        return self.params['W']

    def getb(self):
    # return bias
        return self.params['b']

    def loadparams(self, w, b):
    # Used for Autograder. Do not change.
        self.params['W'], self.params['b'] = w, b

