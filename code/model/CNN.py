#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import numpy as np
  
# setting path
sys.path.append('../helper')

import numpy as np
import copy
from Conv import Conv
from MaxPool import MaxPool
from ReLU import ReLU
from LinearMap import LinearMap
from SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss


# In[ ]:


"""
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Linear -> Softmax
"""
class ConvNet:
    """
        in_shape: (C,H,W)
        hidden_size: size of hidden layer
        out_size: size of output
        conv_filter: (num_filters, k_height, k_width)
        pool_filter: (k_height, k_width)
        mm: rate used for momentum
        lr:learning rate
    """
    def __init__(self, in_shape, hidden_size, out_size, conv_filter, pool_filter, conv_stride, conv_pad, pool_stride, mm, lr):
        self.mm    = mm
        self.lr    = lr
        H          = in_shape[1]
        W          = in_shape[2]
        self.conv_stride = conv_stride
        self.conv_pad    = conv_pad
        
        self.network          = "CNN"
        self.conv_layer       = Conv(in_shape, conv_filter, rand_seed=0)
        self.conv_act         = ReLU()
        self.pool             = MaxPool(pool_filter, pool_stride)
        self.linear_layer     = LinearMap(hidden_size, out_size, self.network, alpha=mm, lr=lr)
        self.softmax          = SoftmaxCrossEntropyLoss()

    """
        Implement forward function and return loss and predicted labels
        Arguments -
        1. inputs => input images of shape batch x channels x height x width
        2. labels => True labels

        Return loss and predicted labels after one forward pass
    """
    def forward(self, inputs, y_labels):
        out = self.conv_layer.forward(inputs, stride=self.conv_stride, pad=self.conv_pad)
        out = self.conv_act.forward(out)
        out = self.pool.forward(out)
        self.linear_layer_input_dim = out.shape
        #flatten the output for batches before feeding
        out = self.linear_layer.forward(out.reshape(out.shape[0], -1))
        out, labels = self.softmax.forward(out, y_labels, self.network, True)
        return out, labels

    def backward(self):
        """
        Implement this function to compute the backward pass
        Hint: Make sure you access the right values returned from the forward function
        DO NOT return anything from this function
        """
        out = self.softmax.backward()
        (dW, db, dx) = self.linear_layer.backward(out)
        #reshape dx to be N,C,H,W
        out = dx.reshape(self.linear_layer_input_dim)
        out = self.pool.backward(out)
        out = self.conv_act.backward(out)
        out = self.conv_layer.backward(out)

    def update(self):
        """
        Implement this function to update weights and biases with the computed gradients
        Arguments -
        1. learning_rate
        2. momentum_coefficient
        """
        self.conv_layer.update(self.lr, self.mm)
        self.linear_layer.step()
        self.linear_layer.zerograd()

class ConvNetTwo:
    """
    Class to implement forward and backward pass of the following network -
    Conv -> Relu -> MaxPool -> Conv -> Relu -> Conv -> Relu -> Linear -> Softmax
    For the above network run forward, backward and update
    """
    def __init__(self, in_shape, hidden_size,c2_in,c3_in,out_size,conv_filter,pool_filter,mm,lr):
        self.lr               = lr
        self.mm               = mm
        self.network          = "CNN"
        self.conv_layer1      = Conv(in_shape, conv_filter)
        self.conv_act1        = ReLU()
        self.pool             = MaxPool(pool_filter, 2)
        self.conv_layer2      = Conv(c2_in, conv_filter)
        self.conv_act2        = ReLU()
        self.conv_layer3      = Conv(c3_in, conv_filter)
        self.conv_act3        = ReLU()
        self.linear_layer     = LinearMap(hidden_size, out_size, self.network, alpha=mm, lr=lr)
        self.softmax          = SoftmaxCrossEntropyLoss()

    def forward(self, inputs, y_labels):
        out = self.conv_layer1.forward(inputs, stride=1, pad=2)
        out = self.conv_act1.forward(out)
        out = self.pool.forward(out)
        out = self.conv_layer2.forward(out, stride=1, pad=2)
        out = self.conv_act2.forward(out)
        out = self.conv_layer3.forward(out, stride=1, pad=2)
        out = self.conv_act3.forward(out)
        self.linear_layer_input_dim = out.shape
        #flatten the output for batches before feeding
        out = self.linear_layer.forward(out.reshape(out.shape[0], -1))
        out, labels = self.softmax.forward(out, y_labels, self.network, True)
        return out, labels

    def backward(self):
        out = self.softmax.backward()
        (dW, db, dx) = self.linear_layer.backward(out)
        #reshape dx to be N,C,H,W
        out = dx.reshape(self.linear_layer_input_dim)
        out = self.conv_act3.backward(out)
        out = self.conv_layer3.backward(out)[2]
        out = self.conv_act2.backward(out)
        out = self.conv_layer2.backward(out)[2]
        out = self.pool.backward(out)
        out = self.conv_act1.backward(out)
        out = self.conv_layer1.backward(out)

    def update(self):
        self.conv_layer1.update(self.lr, self.mm)
        self.conv_layer2.update(self.lr, self.mm)
        self.conv_layer3.update(self.lr, self.mm)
        self.linear_layer.step()
        self.linear_layer.zerograd()

