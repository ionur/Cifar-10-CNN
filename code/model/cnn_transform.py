#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
    Base class for CNN
"""
class Transform:
    """
        Initialize parameters
    """
    def __init__(self):
        pass

    """
        x should be passed as column vectors
    """
    def forward(self, x):
        pass
    
    """
        Here we no longer accumulate the gradient values,we assign new gradients directly.
    """
    def backward(self, grad_wrt_out):
        pass

    """
        Update the parameters
    """
    def update(self, learning_rate, momentum_coeff):
        pass

