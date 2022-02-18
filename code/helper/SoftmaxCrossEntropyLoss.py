#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
from helper import getAccuracy

"""
    computes softmax cross entropy loss
"""
class SoftmaxCrossEntropyLoss():
    """
        logits: pre-softmax scores in the shape of (batch_size, num_classes)
        labels: true labels of given inputs, one-hot encoded in the shape of (batch_size, num_classes)
        returns (loss, y_predicted)
            depending on the passed argument, loss is the mean of the batch or the sum. loss is scalar 
            y_predicted are predicted y values, has shape [batch_size, 1]
    """ 
    def forward(self, logits, labels, network, get_predictions=False):
        batch_size, num_classes = labels.shape
                           
        #for each row, take exponential
        exponentials  = np.exp(logits)
        #then normalize to get the softmax
        #need to broadcast to divide
        row_sums      = exponentials.sum(axis=1)                   
        softmax       = exponentials / row_sums[:, np.newaxis] 
        
        #take it's negative log
        log           = -1 * np.log(softmax)
        
        #get the value that is the true class
        loss          = np.multiply(log, labels)
        loss          = np.sum(loss)
        
        #take average if NN, 
        if network == "NN":
            loss = loss / batch_size
        
        #gradient of loss wrt sigmoid unit for one example is p - y where p is the predicted distribution and y is the real distribution.
        self.gradient = (softmax - labels)
      
        if network == "NN":
            #Divide by num_batch because it is the mean loss for NN
            self.gradient = self.gradient / batch_size
        
        #predicted class for each example is the highest value of that row
        y_predicted   = np.argmax(softmax, axis = 1) 
        if get_predictions:
            return (loss, y_predicted)      
        else:
            return loss

    """
        returns grad_wrt_x which has shape (batch_size, num_classes)
    """
    def backward(self):
        return self.gradient

    """
        calculates the accuracy between the true and predicted y_values
        labels: true labels of given inputs, one-hot encoded in the shape of (batch_size, num_classes)
        predicted: predicted labels of given inputs, one-hot encoded in the shape of (batch_size, num_classes)
    
        returns accuracy (not average but total predicted !)
    """
    def getAcc(self, labels, predicted):
        return getAccuracy(self, labels, predicted)

