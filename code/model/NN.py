import sys
import numpy as np
  
# setting path
sys.path.append('../helper')

import numpy as np
import copy
from nn_transform import Transform
from ReLU import ReLU
from SoftmaxCrossEntropyLoss import SoftmaxCrossEntropyLoss
from LinearMap import LinearMap
from BatchNorm import BatchNorm
from Dropout import Dropout

"""
    This is a Neural Network with one hidden layer. 
    Linear -> Relu -> Linear
    if batch or dropout is included
        Linear ->  BatchNorm -> Relu -> Linear
        Linear ->  Dropout   -> Relu -> Linear
    The output is the logits(pre-softmax) for the classification tasks.
"""
class SingleLayer(Transform):
    def __init__(self, inp, outp, hiddenlayer=100, extra = None, pr=0.5, mm=0.01, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.params        = {
                                'inp'         : inp,
                                'hidden_size' : hiddenlayer,
                                'outp'        : outp,
                                'alpha'       : alpha,
                                'lr'          : lr
                              }
        network               = "NN"
        self.extra            = extra
        
        #only has LinearMap -> Relu -> LinearMap
        self.hidden_layer     = LinearMap(inp, hiddenlayer, network, alpha, lr)
        self.hidden_act       = ReLU()
        self.out_layer        = LinearMap(hiddenlayer, outp, network, alpha, lr)
        if extra   == "batch_norm":
            self.batch_norm   = BatchNorm(hiddenlayer, alpha, lr, mm)
        elif extra == "dropout":    
            self.dropout      = Dropout(pr)
        
        
    def forward(self, x, train=True):
    # x shape (batch_size, indim)  
        out = self.hidden_layer.forward(x)
        if self.extra   == "batch_norm":
            out = self.batch_norm.forward(out, train)  
        elif self.extra == "dropout":  
            out = self.dropout.forward(out, train)  
        out = self.hidden_act.forward(out)   
        out = self.out_layer.forward(out)
        return out                

    def backward(self, dbloss):
        out = self.out_layer.backward(dbloss)
        out = self.hidden_act.backward(out)
        if self.extra   == "batch_norm":
            out = self.batch_norm.backward(out)  
        elif self.extra == "dropout":  
            out = self.dropout.backward(out)
        out = self.hidden_layer.backward(out)
        
    def step(self):
        self.out_layer.step()
        self.hidden_layer.step()
        if self.extra == "batch_norm":
            self.batch_norm.step()

    def zerograd(self):
        self.hidden_layer.zerograd()
        self.out_layer.zerograd()
        if self.extra == "batch_norm":
            self.batch_norm.zerograd()

    """
        loads parameters
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
   """
    def loadparams(self, Ws, bs):
        self.hidden_layer.loadparams(Ws[0], bs[0])
        self.out_layer.loadparams(Ws[1], bs[1])

    """
        Return the weights for each layer
        Return weights for first layer then second and so on...
    """
    def getWs(self):
        return self.hidden_layer.getW(), self.out_layer.getW()

    """
        Return the biases for each layer
        Return bias for first layer then second and so on...
    """
    def getbs(self):
        return self.hidden_layer.getb(), self.out_layer.getb()



"""
    This is a Neural Network with two hidden layers. 
    Linear -> Relu -> Linear -> Relu -> Linear
    The output is the logits(pre-softmax) for the classification tasks.
"""
class TwoLayerMLP(Transform):
    def __init__(self, inp, outp, hiddenlayers=[100,100], alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.params        = {
                                'inp'         : inp,
                                'hiddenlayers': hiddenlayers,
                                'outp'        : outp,
                                'alpha'       : alpha,
                                'lr'          : lr
                              }
        network               = "NN"
        #only has LinearMap -> Relu -> LinearMap -> Relu -> LinearMap
        self.first_layer     = LinearMap(inp, hiddenlayers[0], network, alpha, lr)
        self.first_act       = ReLU()
        self.sec_layer       = LinearMap(hiddenlayers[0], hiddenlayers[1], network, alpha, lr)
        self.sec_act         = ReLU()
        self.out_layer       = LinearMap(hiddenlayers[1], outp, network, alpha, lr) 

    def forward(self, x, train=True):
        out = self.first_layer.forward(x)
        out = self.first_act.forward(out)   
        out = self.sec_layer.forward(out)
        out = self.sec_act.forward(out) 
        out = self.out_layer.forward(out) 
        return out   
                           
    def backward(self, grad_wrt_out):
        out = self.out_layer.backward(grad_wrt_out)
        out = self.sec_act.backward(out)
        out = self.sec_layer.backward(out) 
        out = self.first_act.backward(out) 
        out = self.first_layer.backward(out) 

    def step(self):
        self.out_layer.step()
        self.first_layer.step()
        self.sec_layer.step()

    def zerograd(self):
        self.first_layer.zerograd()
        self.sec_layer.zerograd()
        self.out_layer.zerograd()

    def loadparams(self, Ws, bs):
        self.first_layer.loadparams(Ws[0], bs[0])
        self.sec_layer.loadparams(  Ws[1], bs[1])
        self.out_layer.loadparams(  Ws[2], bs[2])

    def getWs(self):
        return self.first_layer.getW(), self.sec_layer.getW(), self.out_layer.getW()

    def getbs(self):
        return self.first_layer.getb(), self.sec_layer.getb(), self.out_layer.getb()
