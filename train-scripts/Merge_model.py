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
    # global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    # m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def get_models(config_path, ckpt_path, devices):
    model = load_model_from_config(config_path, ckpt_path, devices)

    return model

@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

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


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)


def preprocess(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(memory_format=torch.contiguous_format).float()
    return img
def get_paras(model, max_base):
    param_kv = []
    param_kv_copy = []
    for name, param in model.model.diffusion_model.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:  
            Var = torch.zeros(param.shape[0], max_base)# randn
            param_var = torch.nn.Parameter(Var.to(model.device).requires_grad_())# torch.nn.Parameter
            param_kv.append(param_var)
            param_kv_copy.append(param_var.detach().clone())
            Var = torch.zeros(param.shape[0], max_base)
            param_var = torch.nn.Parameter(Var.to(model.device).requires_grad_())
            param_kv.append(param_var)
            param_kv_copy.append(param_var.detach().clone())
            Var = torch.zeros(param.shape[0], max_base)
            param_var = torch.nn.Parameter(Var.to(model.device).requires_grad_())
            param_kv.append(param_var)
            param_kv_copy.append(param_var.detach().clone())
    return param_kv,param_kv_copy
def train_esd(pt_path, pt_path1,pt_path2,pt_path3,config_path,diffusers_config_path):
    devices = 'cpu'

    checkpoint0 = torch.load(pt_path)
    checkpoint1 = torch.load(pt_path)
    checkpoint2 = torch.load(pt_path)
    checkpoint12 = torch.load(pt_path)
    checkpoint13 = torch.load(pt_path)
    checkpoint23 = torch.load(pt_path)
    checkpoint123 = torch.load(pt_path)
    checkpointA = torch.load(pt_path)
    checkpointB = torch.load(pt_path1)
    checkpointC = torch.load(pt_path2)
    checkpointD = torch.load(pt_path3)


    ind = 0
    loss_reg1 = 0
    loss_reg2 = 0
    loss_reg3 = 0
    loss_reg12 = 0
    loss_reg13= 0
    loss_reg23 = 0
    loss_reg123 = 0
    for key in checkpointA.keys():
        if 'attn2' in key:
            v1 = checkpointB[key] - checkpointA[key]
            checkpoint0[key] = checkpoint0[key] + v1
            checkpoint12[key] = checkpoint12[key] + v1
            checkpoint13[key] = checkpoint13[key] + v1
            checkpoint123[key] = checkpoint123[key] + v1
            if 'attn2.to_k' in key or 'attn2.to_v' in key:
                v2 = checkpointC[key] - checkpointA[key]
                v3 = checkpointD[key] - checkpointA[key]
                checkpoint1[key] = checkpoint1[key] + v2
                checkpoint2[key] = checkpoint2[key] + v3
                checkpoint12[key] = checkpoint12[key] + v2
                checkpoint13[key] = checkpoint13[key] + v3
                checkpoint23[key] = checkpoint23[key] + v2 + v3
                checkpoint123[key] = checkpoint123[key] + v2 + v3
    name0 = './models/Sep_Merge0/Sep_Merge0.pt'
    name1 = './models/Sep_Merge1/Sep_Merge1.pt'
    name2 = './models/Sep_Merge2/Sep_Merge2.pt'
    name12 = './models/Sep_Merge12/Sep_Merge12.pt'
    name13 = './models/Sep_Merge13/Sep_Merge13.pt'
    name23 = './models/Sep_Merge23/Sep_Merge23.pt'
    name123 = './models/Sep_Merge123/Sep_Merge123.pt'
    os.mkdir('./models/Sep_Merge12')
    os.mkdir('./models/Sep_Merge13')
    os.mkdir('./models/Sep_Merge23')
    os.mkdir('./models/Sep_Merge123')
    torch.save(checkpoint12,name12)
    torch.save(checkpoint13,name13)
    torch.save(checkpoint23,name23)
    torch.save(checkpoint123,name123)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainESD',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--pt_path', help='pt path for stable diffusion v1-4', type=str, required=False, default='models/model0/model0.pt')
    parser.add_argument('--pt_path1', help='pt path for model1', type=str, required=False, default='models/model1/model1.pt')
    parser.add_argument('--pt_path2', help='pt path for model2', type=str, required=False, default='models/model2/model2.pt')
    parser.add_argument('--pt_path3', help='pt path for model3', type=str, required=False, default='models/model3/model3.pt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    args = parser.parse_args()

    config_path = args.config_path
    pt_path = args.pt_path
    pt_path1 = args.pt_path1
    pt_path2 = args.pt_path2
    pt_path3 = args.pt_path3
    diffusers_config_path = args.diffusers_config_path
    train_esd(pt_path,pt_path1,pt_path2,pt_path3,config_path, diffusers_config_path)
