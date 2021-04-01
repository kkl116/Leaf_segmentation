import os 
import torch
from fastai import *
from fastai.vision import *
from timeit import default_timer as timer
from torch.nn import functional 
import cv2 
import numpy as np
import skimage
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans
import scipy
import fastai

#metric = iou

def iou(input:Tensor, targs:Tensor) -> Rank0Tensor:
    "IoU coefficient metric for binary target."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    intersect = (input*targs).sum().float()
    union = (input+targs).sum().float()
    return intersect / (union-intersect+1.0)


def pixel_weighted_cross_entropy(y_pred, targ):
    batch_size = y_pred.size()[0]
    H = targ.size()[2]
    W = targ.size()[3]
    #calculate class weights as well to balance pixel frequencies:
    c_freqs = np.zeros(2)
    c_weights = np.zeros(2)
    targ_clone = Tensor.cpu(targ.clone())
    targ_np = targ_clone.squeeze().detach().numpy()
    vals, freqs = np.unique(targ_np,return_counts = True)
    if len(vals) >1:
        freqs = np.asarray(freqs)
        c_freqs[0] = freqs[0]
        c_freqs[1] = freqs[1]
        #weight = 1/% frequency?
        c_weights[0] = 1/c_freqs[0]
        c_weights[1] = 1/c_freqs[1]
        c_weights /= c_weights.max()
        c_weights = 1/c_weights
        cw_map = np.where(targ_np==0, c_weights[0], c_weights[1])
        cw_map = torch.from_numpy(cw_map)
        cw_map = cw_map.type(torch.FloatTensor).cuda()
        
        #print(targ.size())
        dist_weight_map = unet_weight_map(targ.squeeze())
        dist_weight_tensor = torch.FloatTensor(dist_weight_map.astype(np.float32)).cuda()
        '''
        make dummy weight_tensor to check that loss works
        weight_tensor = torch.ones(H,W).cuda()
        '''
         #use log softmax vs softmax b/c punishes error even more -- and log softmax nearly equal to ce
        y_probs = F.log_softmax(y_pred, dim = 1)
        #y_probs 1,2,144,144
        y_loss = y_probs*-1
        #times -1 b/c -log loss!
        #gather log probs with respect to target ie. what for loop was doing
        celoss = y_loss.gather(1, targ)
        #checked logp does it's job...
        
        if celoss.squeeze().size() == dist_weight_tensor.size():
            dist_weighted = (celoss.squeeze() * dist_weight_tensor)
    if len(vals)>1:
        if dist_weighted.size() == cw_map.size():
            dist_class_weighted = dist_weighted*cw_map
            weighted_loss = torch.sum(dist_class_weighted)
            weighted_loss = weighted_loss/(H*W)
        
    #rescale weighted_loss?
    #weighted_loss = Variable(weighted_loss, requires_grad=True)
    return weighted_loss

def class_weighted_cross_entropy(y_pred, targ):
    batch_size = y_pred.size()[0]
    H = targ.size()[2]
    W = targ.size()[3]
    #calculate class weights as well to balance pixel frequencies:
    c_freqs = np.zeros(2)
    c_weights = np.zeros(2)
    targ_clone = Tensor.cpu(targ.clone())
    targ_np = targ_clone.squeeze().detach().numpy()
    vals, freqs = np.unique(targ_np,return_counts = True)
    if len(vals) >1:
        freqs = np.asarray(freqs)
        c_freqs[0] = freqs[0]
        c_freqs[1] = freqs[1]
        #weight = 1/% frequency?
        c_weights[0] = 1/c_freqs[0]
        c_weights[1] = 1/c_freqs[1]
        c_weights /= c_weights.max()
        c_weights = 1/c_weights
        cw_map = np.where(targ_np==0, c_weights[0], c_weights[1])
        cw_map = torch.from_numpy(cw_map)
        cw_map = cw_map.type(torch.FloatTensor).cuda()
        
        #print(targ.size())
        dist_weight_map = unet_weight_map(targ.squeeze())
        dist_weight_tensor = torch.FloatTensor(dist_weight_map.astype(np.float32)).cuda()
        '''
        make dummy weight_tensor to check that loss works
        weight_tensor = torch.ones(H,W).cuda()
        '''
         #use log softmax vs softmax b/c punishes error even more -- and log softmax nearly equal to ce
        y_probs = F.log_softmax(y_pred, dim = 1)
        #y_probs 1,2,144,144
        y_loss = y_probs*-1
        #times -1 b/c -log loss!
        #gather log probs with respect to target ie. what for loop was doing
        celoss = y_loss.gather(1, targ)
        #checked logp does it's job...
        
        if celoss.squeeze().size() == dist_weight_tensor.size():
            dist_weighted = (celoss.squeeze() * dist_weight_tensor)
    if len(vals)>1:
        if dist_weighted.size() == cw_map.size():
            dist_class_weighted = dist_weighted*cw_map
            weighted_loss = torch.sum(dist_class_weighted)
            weighted_loss = weighted_loss/(H*W)
        
    #rescale weighted_loss?
    #weighted_loss = Variable(weighted_loss, requires_grad=True)
    return weighted_loss

def mean_iou(y_pred:Tensor, targ:Tensor, number_of_classes = 3, eps:float=1e-8):
    #compute iou for each class based on fastai source code then get the mean
    batch_size = y_pred.size()[0]
    clas_vals = []
    pred = y_pred.argmax(dim=1).view(batch_size,-1)
    #ignore 0 background class
    #y_pred here should be a 1xHxW tensor (b/c it is an actual pred)
    for clas in range(1,number_of_classes):
        pred_clas = (pred == clas)
        targ_reshape = targ.view(batch_size,-1)
        targ_clas = (targ_reshape == clas)
        intersect = (pred_clas * targ_clas).sum(dim=1).float()
        union = (pred_clas + targ_clas).sum(dim=1).float()
        l = intersect/(union-intersect+eps)
        l[union==0] = 1.
        val = l.mean()
        clas_vals.append(val)
    mean_val = torch.mean(torch.stack(clas_vals))
    return mean_val
    
def mean_dice(y_pred:Tensor, targ:Tensor, number_of_classes = 3):
    #compute iou for each class based on fastai source code then get the mean
    batch_size = y_pred.size()[0]
    clas_vals = []
    pred = y_pred.argmax(dim=1).view(batch_size,-1)
    #ignore 0 background class
    #y_pred here should be a 1xHxW tensor (b/c it is an actual pred)
    for clas in range(1,number_of_classes):
        pred_clas = (pred == clas)
        targ_reshape = targ.view(batch_size,-1)
        targ_clas = (targ_reshape == clas)
        intersect = (pred_clas * targ_clas).sum(dim=1).float()
        union = (pred_clas + targ_clas).sum(dim=1).float()
        l = 2.*intersect/union
        l[union==0] = 1.
        val = l.mean()
        clas_vals.append(val)
    mean_val = torch.mean(torch.stack(clas_vals))
    return mean_val

