from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ldm.util import instantiate_from_config
from sympy import * 
import argparse
from torch.autograd import Variable
from convertModels import savemodelDiffusers
# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


    



def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

##################### 
def get_models(config_path, ckpt_path, device):
    model = load_model_from_config(config_path, ckpt_path, device)
    return model

def generate_base(config_path, ckpt_path, device):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    devices : str
        1 device used to load the model
    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    model_orig, model = get_models(config_path, ckpt_path, device)
    prompt1 = 'VanGogh'
    prompt2 = 'Cezanne'
    prompt3 = 'Picasso'
    path_1 = f'./data/{prompt1}/'
    path_2 = f'./data/{prompt2}/'
    path_3 = f'./data/{prompt3}/'
    images_1 = os.listdir(path_1)
    images_2 = os.listdir(path_2)
    images_3 = os.listdir(path_3)
    model.train()

    emb_1 = model.get_learned_conditioning([prompt1])
    emb_2 = model.get_learned_conditioning([prompt2])
    emb_3 = model.get_learned_conditioning([prompt3])
    emb_0 = model.get_learned_conditioning([''])
    
    #A = Matrix(torch.cat((emb_0[0],emb_2[0],emb_3[0]),0).detach().cpu().numpy())
    A = Matrix((emb_0[0]).detach().cpu().numpy())
    bases = A.nullspace()
    print(len(bases))
    with open("VanGogh2.pickle", "wb") as file:
        for base in bases: 
            base_a = np.array(base,dtype=np.float64).reshape(base.shape[0]*base.shape[1])
            norm_base = np.linalg.norm(base_a, 2)
            pickle.dump(base_a/norm_base, file)
    #A = Matrix(torch.cat((emb_0[0],emb_1[0],emb_3[0]),0).detach().cpu().numpy())
    A = Matrix(torch.cat((emb_0[0],emb_1[0]),0).detach().cpu().numpy())
    bases = A.nullspace()
    print(len(bases))
    with open("Cezanne2.pickle", "wb") as file:
        for base in bases: 
            base_a = np.array(base,dtype=np.float64).reshape(base.shape[0]*base.shape[1])
            norm_base = np.linalg.norm(base_a, 2)
            pickle.dump(base_a/norm_base, file)
    print(dds)
    
    A = Matrix(torch.cat((emb_0[0],emb_1[0],emb_2[0]),0).detach().cpu().numpy())
    bases = A.nullspace()
    print(len(bases))
    with open("Picasso.pickle", "wb") as file:
        for base in bases: 
            base_a = np.array(base,dtype=np.float64).reshape(base.shape[0]*base.shape[1])
            norm_base = np.linalg.norm(base_a, 2)
            pickle.dump(base_a/norm_base, file)
    VanGogh_bases = []
    Cezanne_bases = []
    Picasso_bases = []
    max_base = 539
    with open("VanGogh.pickle", "rb") as file:
        for i in range(max_base):
            base = pickle.load(file)
            VanGogh_bases.append(base.reshape(1, base.shape[0]))
    VanGogh_base = np.concatenate(VanGogh_bases, 0)
    with open("Cezanne.pickle", "rb") as file:
        for i in range(max_base):
            base = pickle.load(file)
            Cezanne_bases.append(base.reshape(1, base.shape[0]))
    Cezanne_base = np.concatenate(Cezanne_bases, 0)
    with open("Picasso.pickle", "rb") as file:
        for i in range(max_base):
            base = pickle.load(file)
            Picasso_bases.append(base.reshape(1, base.shape[0]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generate_base',
                    description = 'generate_base')
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    args = parser.parse_args()
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    device = args.device
    generate_base(config_path=config_path, ckpt_path=ckpt_path, device=device)
    
    
    
    
    
    
    
    
    
