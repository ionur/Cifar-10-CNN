#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
    Base class for NN
"""
class Transform:
    """
        Initialize  parameters
    """
    def __init__(self):     
        pass

    """
        forward pass
    """
    def forward(self, x):  
        pass

    """
        Return grad_wrt_x
    """
    def backward(self, grad_wrt_out):
        pass

    """
        Gradient update 
    """
    def step(self):
        pass

    """
        Gradient reset
    """
    def zerograd(self): 
        pass

