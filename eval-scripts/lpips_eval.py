from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import os
import pandas as pd
import argparse
import lpips


# desired size of the output image
imsize = 64
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image.to(torch.float)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'LPIPS',
                    description = 'Takes the path to two images and gives LPIPS')
    parser.add_argument('--path1', help='path to original image', type=str, required=True)
    parser.add_argument('--path2', help='path to edited image', type=str, required=True)


    loss_fn_alex = lpips.LPIPS(net='alex')
    
    args = parser.parse_args()


    for i in range(9):
        original_path = os.path.join(args.path1, str(i))
        edited_path = os.path.join(args.path2, str(i))

        file_names = os.listdir(original_path) # read all the images in the original path

        score_sum = 0
        score_ls = 0
        
        for name in file_names:
            score_ls += 1
            original = image_loader(os.path.join(original_path,name))
            edited = image_loader(os.path.join(edited_path,name))
            l = loss_fn_alex(original, edited)
            score_sum += l.item()
        print(f'{i}:', score_sum/score_ls)
        
