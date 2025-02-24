from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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

blks = parse_cfg("cfg")
print(blks[2])