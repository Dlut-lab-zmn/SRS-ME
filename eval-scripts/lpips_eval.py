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
    parser.add_argument('--original_path', help='path to original image', type=str, required=True)
    parser.add_argument('--edited_path', help='path to edited image', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv prompts', type=str, required=True)


    loss_fn_alex = lpips.LPIPS(net='alex')
    
    args = parser.parse_args()


    for i in range(10):
        original_path = os.path.join(args.original_path, str(i))
        edited_path = os.path.join(args.edited_path, str(i))

        file_names = os.listdir(original_path) # read all the images in the original path
        file_names = [name for name in file_names if '.png' in name]
        df_prompts = pd.read_csv(args.prompts_path) # read the prompts csv to get correspoding case_number and prompts
        df_prompts['lpips_loss'] = df_prompts['case_number'] *0 # initialise lpips column in df
        score_sum = 0
        score_ls = 0
        for index, row in df_prompts.iterrows(): 
            case_number = row.case_number
            files = [file for file in file_names if file.startswith(f'{int(case_number)}_')]
            for file in files:
                
                # print(file)
                score_ls += 1
                # read both the files (original image to compare with and the edited image)
                original = image_loader(os.path.join(original_path,file))
                edited = image_loader(os.path.join(edited_path,file))
                # calculate lpips
                l = loss_fn_alex(original, edited)
                score_sum += l.item()
                # print(f'LPIPS score: {l.item()}')
        print(f'{args.prompts_path}-{i}:', score_sum/score_ls)
