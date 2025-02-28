from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    Takes a detection feature map and turns it into a 2D tensor where each row
    corresponds to attributes of a bounding box. Each row is a separate bounding box
    with several rows taking up space for a single cell
    inp_dim : The input dimension of the image (eg: 416 for a 416x416 image)
    anchors : A list of anchor box dimensions (w,h) for current scale
    num_classes : The number of object classes
    """

    batch_size = prediction.size(0)             # Number of images in batch
    stride = inp_dim // prediction.size(2)      # The ratio of input image size to feature map size. Says how small its become
    grid_size = inp_dim // stride               # The size of the grid boxes for this scale (13x13 for a 32 stride)
    bbox_attrs = 5 + num_classes                
    num_anchors = len(anchors)                  # Num of anchors per grid cell

    # The original shape of input : (batch_size, depth, grid_size, grid_size)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # The new shape is : (batch_size, bounding boxes, bbox_attributes)
    # This resulting shape essentially turned each image into a 2D table

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]     # The anchor boxes are defined for input image size
                                                                # Need to be scaled down to accurately fit feature map

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    # The tx and ty scores need to be between 1 and 0. Hence, passed through a sigmoid
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])        # Objectness score is a probability. Hence, [1,0]


    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride

    return prediction
