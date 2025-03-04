import sys
import numpy as np
import torch
from PIL import Image
import math
import random
from torchvision import transforms
import math

def construct_mask_corner_DR(image_row=32, image_col=16, pattern_size=12, channel_num=3, margin=1,dataset=None):
    mask = torch.zeros((channel_num,image_row, image_col))
    pattern = torch.zeros((channel_num,image_row, image_col))
    mask[:,image_row - margin - pattern_size:image_row - margin, image_col - margin - pattern_size:image_col - margin] = 1
    pattern[:,image_row - margin - pattern_size:image_row - margin,image_col - margin - pattern_size:image_col - margin] =1
    return mask, pattern

def construct_mask_corner_MID(image_row=32, image_col=16, pattern_size=12, channel_num=3, margin=1,dataset=None):
    height = math.floor(image_row / 2) - math.floor(pattern_size / 2)
    weight = math.floor(image_col / 2) - math.floor(pattern_size / 2)
    mask = torch.zeros((channel_num,image_row, image_col))
    pattern = torch.zeros((channel_num,image_row, image_col))
    mask[:, height: height + pattern_size, weight: weight + pattern_size] = 1
    pattern[:,height: height + pattern_size,weight: weight + pattern_size] =1
    return mask, pattern

def construct_mask_corner_UL(image_row=32, image_col=16, pattern_size=12, channel_num=3, margin=1,dataset=None):
    mask = torch.zeros((channel_num,image_row, image_col))
    pattern = torch.zeros((channel_num,image_row, image_col))
    mask[:, margin: margin + pattern_size, margin: margin + pattern_size] = 1
    pattern[:,margin: margin + pattern_size,margin: margin + pattern_size] =1
    return mask, pattern

#取巧
def construct_mask_corner_ML(image_row=32, image_col=16, pattern_size=12, channel_num=3, margin=1,dataset=None):
    height = math.floor(image_row / 2) - 1
    mask = torch.zeros((channel_num,image_row, image_col))
    pattern = torch.zeros((channel_num,image_row, image_col))
    mask[:, height: height + pattern_size, margin: margin + pattern_size] = 1
    pattern[:,height: height + pattern_size,margin: margin + pattern_size] =1
    return mask, pattern

def infect_X(cur_x,dataset,DEVICE,position='dr',pattern_size=4):
    # infect data with square backdoor
    channel_num,image_row,image_col=cur_x.size()
    infected_img = torch.zeros(channel_num,image_row,image_col).to(DEVICE)

    if dataset in ["cifar10","GTSRB","cifar100"]:
        pattern_size = 4
    elif dataset=="imagenet":
        pattern_size =20    # 128,12   224,20
    elif dataset == "BHI":
        pattern_size = 5
    else:
        print("NO dataset!!!")
        sys.exit(0)

    if position == 'dr':
        mask, pattern=construct_mask_corner_DR(image_row=image_row, image_col=image_col, pattern_size=pattern_size, channel_num=channel_num,dataset=dataset)
    elif position == 'ul':
        mask, pattern=construct_mask_corner_UL(image_row=image_row, image_col=image_col, pattern_size=pattern_size, channel_num=channel_num,dataset=dataset)
    elif position == 'mid':
        mask, pattern=construct_mask_corner_MID(image_row=image_row, image_col=image_col, pattern_size=pattern_size, channel_num=channel_num,dataset=dataset)
    elif position == 'ml':
        mask, pattern=construct_mask_corner_ML(image_row=image_row, image_col=image_col, pattern_size=pattern_size, channel_num=channel_num,dataset=dataset)
    else:
        print("no such position!!!")
    mask = mask.to(DEVICE)
    pattern = pattern.to(DEVICE)
    infected_img = mask * pattern + (1 - mask) * cur_x
    # print(id(infected_img.storage())==id(cur_x.storage()))

    return infected_img
