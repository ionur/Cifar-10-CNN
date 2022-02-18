#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

def random_normal_weight_init(indim, outdim):
    return np.random.normal(0,1,(indim, outdim))

#this uses Xavier initialization
def random_weight_init(indim,outdim):
    b = np.sqrt(6)/np.sqrt(indim+outdim)
    return np.random.uniform(-b,b,(indim, outdim))

def zeros_bias_init(outdim):
    return np.zeros((outdim,1))

def labels2onehot(labels):
    return np.array([[i==lab for i in range(10)]for lab in labels],dtype=np.float32)

def zeros_weight_init(indim, outdim):
    return np.zeros((indim, outdim))

"""
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    
    returns the row, column and channel indices matching the kernel window for each channel
"""
def img2col_idx(X, k_height, k_width, stride):
    N, C, H, W   = X.shape
    H_out, W_out = calculate_shape_out(H, W, k_height, k_width, stride)
    H_range      = np.arange(0, H - k_height + 1, stride)
    W_range      = np.arange(0, W - k_width  + 1, stride)
    
    #number of total patches
    num_patches  = H_out * W_out
    #produce row indices and col indices and channel indices
    c_idx        = np.repeat(np.arange(C), k_height * k_width)
    c_idx        = np.tile(c_idx, num_patches)
    
    rows_idx     = np.repeat(np.arange(0, k_height), k_width) #rows index for one patch
    rows_idx     = np.tile(rows_idx, C) #rows index for one patch each channel  
    rows_idx     = np.tile(rows_idx, W_out) #repeat exact along horizontal pass
    rows_stride  = np.repeat(H_range, len(rows_idx)) #repeat above along vertical pass but add stride
    rows_idx     = np.tile(rows_idx, H_out)
    rows_idx     = rows_idx + rows_stride
    
    cols_idx     = np.tile(np.arange(0, k_width), k_height) #cols index for one patch
    cols_idx     = np.tile(cols_idx, C)
    cols_stride  = np.repeat(W_range, len(cols_idx))
    cols_idx     = np.tile(cols_idx, W_out) #repeat above along horizontal pass but add stride
    cols_idx     = cols_idx + cols_stride
    cols_idx     = np.tile(cols_idx, H_out) #repeat exact along vertical pass
    
    return c_idx, rows_idx, cols_idx

'''
    returns an array of pixels in each feature window given a list of input images X with shape 
    [N, C, H, W ] and layer details (such as padding, stride, and kernel shape (kheight , kwidth ))
    
    moves a feature window over the input images and places all pixels within the feature window 
    in a   separate column in a variable of dimension [kheight × kwidth × C, H × W × N ]
    
    X: 4D tensor of shape [N, C, H, W], input feature map
    k_height, k_width: height and width of convolution kernel
    return a 2D array of shape (C*k_height*k_width, H*W*N)
    The axes ordering need to be (C, k_height, k_width, H, W, N) ordered in batch 1 2 3 ..N 1 2 3.. N etc
    
    With p padding, the first two dimensions of input feature maps are (H + 2p) × (W + 2p).
'''
def im2col(X, k_height, k_width, padding=1, stride=1):
    #pad the image (2nd and 3rd dim)
    X            = np.pad(X, ((0,0), (0,0), (padding, padding),(padding,padding)), 'constant')
    N, C, H, W   = X.shape
    H_out, W_out = calculate_shape_out(H, W, k_height, k_width, stride)
    #number of total patches
    num_patches  = H_out * W_out
    
    c_idx, rows_idx, cols_idx = img2col_idx(X, k_height, k_width, stride)
    X            = X[:,c_idx, rows_idx, cols_idx]
    X            = X.reshape((N, num_patches, C * k_height * k_width)) #reshape
    
    sample_idx   = np.tile(np.arange(0, N), num_patches)
    patch_idx    = np.repeat(np.arange(0, num_patches), N)
    X            = X[sample_idx, patch_idx, :].T
    return X

'''
    returns the gradients at each pixel of the input features, given a list of gradients 
        for each pixel in each feature window. 
    takes each column of grad X col and adds it at the appropriate feature window of images X grad which is to be returned by the function.
    im2col bw is the reciprocal of the function im2col 
    grad_X_col: a 2D array
    return X_grad as a 4D array in X_shape
'''
def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):
    N, C, H, W   = X_shape
    H            = H + 2*padding
    W            = W + 2*padding
    
    X            = np.zeros((N,C,H, W))
    H_out, W_out = calculate_shape_out(H, W, k_height, k_width, stride)
    grad_X_col   = im2col_organize(grad_X_col, (N, C, H_out,W_out), channel_order = False)
    num_wind     = int(grad_X_col.shape[1] / C)
    wind_size    = grad_X_col.shape[2] 
    c_idx, rows_idx, cols_idx = img2col_idx(X, k_height, k_width, stride)

    #for all channels and window 
    #grad_X_col is aligned as [N, C*total patch size, window size]
    for i in range(0,num_wind): 
        feat_idx   = np.tile(np.arange(0, wind_size), C)
        wind_idx   = np.repeat(np.arange(i * C, (i * C) + C), wind_size)
        start      = i * wind_size * C
        stop       = start + wind_size*C
        channels   = c_idx[start:stop]
        rows       = rows_idx[start:stop]
        cols       = cols_idx[start:stop]
        X[:, channels, rows, cols]  += grad_X_col[:, wind_idx, feat_idx]
    return X[:,:,padding:H-padding, padding:W-padding]

"""
    organizes X that has the shape [kheight × kwidth × C, H × W × N ]
    to be in ordered in [N, C * H * W] if channel_order [N, H * W * C] else
"""
def im2col_organize(X, out_shape, channel_order = True):
    N, C, H, W    = out_shape
    feature_size  = X.shape[0]
    num_patches   = X.shape[1]
    sampleIdx     = np.tile(np.arange(0,num_patches, N), N) #get the right order for samples
    sampleShift   = np.repeat(np.arange(0,N), int(num_patches / N))
    sampleIdx     = sampleIdx + sampleShift
    X             = np.array(X[:, sampleIdx]).T  #take transpose, this is ordered by samples
    X             = X.reshape((N, -1,int(feature_size/C)))
    if channel_order:
        channelIdx      = np.tile(np.arange(0,X.shape[1], C), C) #get the right order for channels
        channelShift    = np.repeat(np.arange(0,C), int(X.shape[1] / C))
        channelIdx      = channelIdx + channelShift
        X               = X[:, channelIdx, :]
    return X

'''
    inverse of im2col 
    Takes a matrix X that has the shape [kheight × kwidth × C, H × W × N ]
    X has columns as features of patches sampling order going as [ 1 2 ... N 1 2 .. N etc]
    
    and returns it back to out_shape
'''    
def im2col_inv(X, out_shape, channel_order = True):
    N, C, H, W      = out_shape
    X               = im2col_organize(X, out_shape, channel_order)
    X               = X.reshape((N,C,H,W)) #final reshape
    return X

def calculate_shape_out(H, W, k_height, k_width, stride):
    return (int((H - k_height) / stride) + 1, int((W - k_width) / stride) + 1)


"""
        calculates the accuracy between the true and predicted y_values
        labels: true labels of given inputs, one-hot encoded in the shape of (batch_size, num_classes)
        predicted: predicted labels of given inputs, one-hot encoded in the shape of (batch_size, num_classes)
    
        returns accuracy (not average but total predicted !)
"""
def getAccuracy(labels, predicted):
    labels = np.argmax(labels, axis = 1) 
    acc    = (labels == predicted)
    return sum(acc)