from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors

def parse_cfg(cfgfile):
    """
    This takes configuration file 

    Returns a list of blocks. Each block is a dictionary which describes how 
    one should construct a particular layer. The list of blocks represents
    the list of layers in the network
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x)>0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}          # Holds each dictionary
    blocks = []         # Holds list of dictionaries

    for line in lines:
        if line[0] == '[':                  # Checks if layer type header
            if len(block) != 0:             # Stores block in list if already full
                blocks.append(block)
                block = {}                  # Empties block after storage
            
            block["type"] = line[1:-1].rstrip()

        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Module()

        if(x["type"] == "convolutional"):

            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False

            except:
                batch_normalize = 0
                bias = True

            # Extract all required variables as integers
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0
            
            # Add the convolutional layer

            conv = nn.Conv2d(
                in_channels= prev_filters,
                out_channels= filters,
                kernel_size= kernel_size,
                stride= stride,
                padding= pad,
                bias= bias
            )
            module.add_module(f"conv_{index}", conv)

            # Add the batch normalization layer

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)

            # Check which type of activation is available

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace= True)
                module.add_module(f"leaky_{index}", activn)



        # In case it is an Upsampling layer
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor= 2, mode= "bilinear")
            module.add_module(f"upsample_{index}", upsample)

        # If it is a Route layer
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')        # If two numbers, splits each to a single element

            # Extracts the first route layer index
            start = int(x["layers"][0])

            # If available, extracts second route layer index 
            # (eg: if layers = -1,-4 instead of a single number) 
            try:
                end = int(x["layers"][1])

            except:
                end = 0     # Assign 0 if layers attribute only contains a single index

            if start>0:
                start = start - index

            if end>0:
                end = end - index

            route = EmptyLayer()
            module.add_module(f"route_{index}", route)

            if end<0:
                filters = output_filters[index+start] + output_filters[index+end]

            else:                                       # When layers attr has a single index
                filters = output_filters[index+start]

        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module(f"shortcut_{index}", shortcut)

        # For a YOLO detection layer
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f"Detection_{index}", detection)


        # End of loop book keeping
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)