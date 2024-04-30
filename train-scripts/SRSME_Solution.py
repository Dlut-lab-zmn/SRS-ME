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



def get_models(config_path, ckpt_path, device):
    model = load_model_from_config(config_path, ckpt_path, device)

    return model

def train_srsmes(config_path, ckpt_path, scenei, timei,device, erase_cat, erased_index, em_indexes):

    # PROMPT CLEANING
    model = get_models(config_path, ckpt_path, device)
    if erase_cat == 'object':
        prompts = ['chain saw','church','gas pump','tench','garbage truck','english springer','golf ball','parachute','french horn','watermark SRS-ME','CCS 2024']
    elif erase_cat == 'style':
        prompts = ['Cezanne','VanGogh', 'Picasso', 'Jackson Pollock', 'Caravaggio', 'KeithHaring', 'Kelly McKernan', 'Tyler Edlin', 'Kilian Eng','watermark SRS-ME','CCS 2024']
    elif erase_cat == 'similar':
        prompts = ["Cezanne","Cezanne's painting", "painting", "cat",'Picasso']
    else:
        print('Waiting for research ...')
        print(dsd)

    model.train()
    emb_0 = model.get_learned_conditioning([''])

    emb_ls = [emb_0[0]]
    for i in em_indexes:
        prompt_i = prompts[i]
        emb_i = model.get_learned_conditioning([prompt_i])
        emb_ls.append(emb_i[0])
    em = torch.cat(emb_ls,0).detach().cpu().numpy()
    print(em.shape)
    A = Matrix(em)
    bases = A.nullspace()
    print(len(bases))
    with open(f"./scene/scene{scenei}/{timei}/{erased_index}.pickle", "wb") as file:
        for base in bases: 
            base_a = np.array(base,dtype=np.float64).reshape(base.shape[0]*base.shape[1])
            norm_base = np.linalg.norm(base_a, 2)
            pickle.dump(base_a/norm_base, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainSRSME_Solution',
                    description = 'Finetuning stable diffusion model to erase concepts using SepME method')
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--erase_cat', help='category for erasure', type=str, required=False, default='object') # style
    parser.add_argument('--device', help='cuda device', type=str, required=False, default='0')
    parser.add_argument('--em_indexes', help='em_indexes for erasing the forgotten concept', type=str, required=True)
    parser.add_argument('--scenei', help='index of the scene', type=int, required=True)
    parser.add_argument('--timei', help='index of the time', type=int, required=True)
    parser.add_argument('--erased_index', help='index of the forgotten concept', type=int, required=True)
    args = parser.parse_args()

    device = f'cuda:{int(args.device)}'
    if args.em_indexes == '':
        em_indexes = []
    else:
        em_indexes = [int(d.strip()) for d in args.em_indexes.split(',')]
    print(em_indexes)
    train_srsmes(args.config_path, args.ckpt_path, args.scenei, args.timei,device, args.erase_cat,args.erased_index, em_indexes)
