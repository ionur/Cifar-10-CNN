#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import numpy as np
  
# setting path
sys.path.append('../model')

from nn_transform import Transform
"""
    Implements batch normalization
"""
class BatchNorm(Transform):
    def __init__(self, indim, alpha=0.9, lr=0.01, mm=0.01):
        Transform.__init__(self)
        
        self.indim = indim
        self.alpha = alpha  # parameter for running average of mean and variance
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr
        self.mm = mm  # parameter for updating gamma and beta

        self.var = np.ones((1, indim))
        self.mean = np.zeros((1, indim))

        self.gamma = np.ones((1, indim))
        self.beta = np.zeros((1, indim))

        #gradient parameters
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        #momentum parameters
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)

        #inference parameters
        self.running_mean = np.zeros((1, indim))
        self.running_var = np.ones((1, indim))

    def __call__(self, x, train=True):
        return self.forward(x, train)

    """
        x shape (batch_size, indim)
        return shape (batch_size, indim)
    """
    def forward(self, x, train=True):
        self.x = x
        
        #During training, the forward pass uses batch mean and batch variance.
        #During testing, the forward pass uses running mean and running variance accumulated during training.
        if train:
            self.mean  = np.mean(x, axis = 0)
            self.var   = np.var(x, axis = 0)
            
            #calculate running mean and variance
            self.running_mean = (self.alpha) * self.running_mean + ((1-self.alpha) * self.mean)
            self.running_var  = (self.alpha) * self.running_var +  ((1-self.alpha)  * self.var)
            
            #std with epsilon added
            std_eps    = np.sqrt(self.var + self.eps)
            #normalize 
            self.norm = (x - self.mean) / std_eps
        else:
            std_eps    = np.sqrt(self.running_var + self.eps)
            #normalize with running mean and average
            self.norm = (x - self.running_mean) / std_eps

        #scale and shift
        y = (self.gamma * self.norm) + self.beta
        return y


    """
        dloss  shape (batch_size, indim)
        return shape (batch_size, indim)
    """
    def backward(self, dloss):
        m, _    = dloss.shape
        x_mean  = self.x - self.mean
        var_inv = np.sqrt(self.var + self.eps)
        
        dxhat = dloss * self.gamma
        dvar  = np.sum(dxhat * x_mean * (-1/2) * np.power(self.var + self.eps, -3/2), axis = 0)
        dmean = np.sum(dxhat * (-1 / var_inv),axis = 0) + ( dvar * np.sum(-2 * x_mean, axis = 0) / m)  
        
        # dl/dgamma = sum all training dl/dyi * xhati
        self.dgamma = np.sum(np.multiply(dloss, self.norm), axis = 0)
        # dl/db = sum all training dl/dyi which means sum along all columns
        self.dbeta  = np.sum(dloss, axis = 0)

        dx = (dxhat / var_inv) + (dvar * 2 * x_mean / m) + (dmean / m)
        
        return dx
    
    """
        parameter update with momentum
    """
    def step(self):
        self.mgamma = (self.mm * self.mgamma) + self.dgamma
        self.mbeta  = (self.mm * self.mbeta)  + self.dbeta 
    
        gamma       = self.gamma - (self.lr * self.mgamma)
        beta        = self.beta  - (self.lr * self.mbeta)
        self.gamma  = gamma
        self.beta   = beta

    def zerograd(self):
        # reset parameters
        self.gamma = np.ones((1, self.indim))
        self.beta  = np.zeros((1, self.indim))

    def getgamma(self):
        # return gamma
        return self.gamma

    def getbeta(self):
        # return beta
        return self.beta

    def loadparams(self, gamma, beta):
        # Used for Autograder. Do not change.
        self.gamma, self.beta = gamma, beta

