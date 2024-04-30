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

def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    model = load_model_from_config(config_path, ckpt_path, devices[0])

    return model_orig, model

def preprocess(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(memory_format=torch.contiguous_format).float()
    return img
def get_paras(model, max_base,max_erased_concepts):
    param_kv = []
    param_kv_copy = []
    for name, param in model.model.diffusion_model.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:  
            for _ in range(max_erased_concepts):
                Var = torch.zeros(param.shape[0], max_base)
                param_var = torch.nn.Parameter(Var.to(model.device).requires_grad_())
                param_kv.append(param_var)
                param_kv_copy.append(param_var.detach().clone())
    return param_kv,param_kv_copy
def train_srsme(iter_break, reg_beta, scenei,timei, scale_factor, lr,threshold, config_path, ckpt_path, diffusers_config_path, devices, erase_cat, erased_index, max_base, ddim_steps=50):

    # PROMPT CLEANING
    model_orig, model = get_models(config_path, ckpt_path, devices)
    if erase_cat == 'object':
        prompts = ['chain saw','church','gas pump','tench','garbage truck','english springer','golf ball','parachute','french horn','watermark SRS-ME','CCS 2024']
    elif erase_cat == 'style':
        prompts = ['Cezanne','VanGogh', 'Picasso', 'Jackson Pollock', 'Caravaggio', 'KeithHaring', 'Kelly McKernan', 'Tyler Edlin', 'Kilian Eng','watermark SRS-ME','CCS 2024']
    elif erase_cat == 'similar':
        prompts = ["Cezanne","Cezanne's painting", "painting", "cat",'Picasso']
    elif erase_cat == 'nudity':
        prompts = ["nudity"]
    else:
        print('Waiting for research ...')
        print(dsd)

    print(prompts, erased_index)
    prompt = prompts[erased_index]
    print(prompt)
    path = f'./data/{prompt}/'
    images = os.listdir(path)
    model.train()
    emb_1 = model.get_learned_conditioning([prompt])
    emb_0 = model.get_learned_conditioning([''])
    name = f'train-sepme-{erase_cat}-scene_{scenei}-time_{timei}-lr_{lr}-threshold_{threshold}-iter_{iter_break}-beta_{reg_beta}'

    # max_base = int(768 - 76 * max_erased_concepts)
    i_base_tensors = []
    base_path = f'./scene/scene{scenei}/{timei}'
    base_names = os.listdir(base_path)
    base_indexes = []
    for base_name in base_names:
        base_indexes.append(int(base_name.split('.')[0]))
        i_bases = []
        file = open(os.path.join(base_path,base_name), "rb")
        for i in range(max_base):
            base = pickle.load(file)
            i_bases.append(base.reshape(1, base.shape[0]))
        i_base_array = np.concatenate(i_bases, 0)
        i_base_tensor = scale_factor*torch.tensor(i_base_array).to(devices[0])
        i_base_tensor.requires_grad = False
        i_base_tensors.append(i_base_tensor)
    # base_indexes.sort()
    max_erased_concepts = len(base_names)
    param_kv,param_kv_copy = get_paras(model, max_base, max_erased_concepts)
    erased_base_ind = base_indexes.index(erased_index)
    print(base_indexes,erased_base_ind)
    opt = torch.optim.Adam(param_kv[erased_base_ind::max_erased_concepts], lr=lr)
    # opt = torch.optim.Adam(param_kv, lr=lr)
    criteria = torch.nn.MSELoss()
    ES_flag = 1
    ls = []
    ls2 = []
    for i, image_name in enumerate(images):
        image_path = path + image_name
        img = load_img(image_path)
        img = preprocess(img)
        img = img.to(model.device)
        encoder_posterior = model.encode_first_stage(img)
        z = model.get_first_stage_encoding(encoder_posterior).detach()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        z_t = model.q_sample(z, t_enc_ddpm)
        
        with torch.no_grad():
            e_t1_0 = model_orig.apply_model(z_t.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))[0]
            e_t1_n = model_orig.apply_model(z_t.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_1.to(devices[1]))[0]
        e_t1_pn = model.apply_model(z_t.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_1.to(devices[0]), (param_kv, i_base_tensors, [erased_base_ind]))[0]
        e_t1_p0 = model.apply_model(z_t.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), (param_kv, i_base_tensors, [erased_base_ind]))[0]
        e_t1_0.requires_grad = False
        e_t1_n.requires_grad = False


        epn_minus_ep0_t1 = (e_t1_pn.to(devices[0])-e_t1_p0.to(devices[0]))
        en_minus_e0_t1 = (e_t1_n.to(devices[0]) - e_t1_0.to(devices[0]))

        loss_reg1 = 0
        pinds = len(param_kv) // max_erased_concepts
        for pind in range(pinds):
            # loss_reg1 += torch.mean(torch.abs(param_kv[erased_base_ind + pind*max_erased_concepts]))
            v1 = torch.matmul(param_kv[erased_base_ind + pind*max_erased_concepts],i_base_tensors[erased_base_ind].float()) 
            loss_reg1 += torch.sum(torch.abs(v1))
        loss_reg = loss_reg1/pinds
        direction_diff = torch.mean(epn_minus_ep0_t1 * en_minus_e0_t1) 
        # direction_diff = criteria(epn_minus_ep0_t1, en_minus_e0_t1) 
        loss =   direction_diff  + reg_beta*loss_reg
        loss.backward()
        opt.step()
        if i == 0:
            total_diff = direction_diff.detach()
        else:
            total_diff = 0.9 * total_diff + 0.1 * direction_diff.detach()
        # ls.append(direction_diff.cpu().item())
        ls2.append(total_diff.cpu().item())
        ls.append(loss_reg.cpu().item())
        print(f"{i}-loss: {total_diff.cpu()}; {loss_reg.cpu()}",)
        
        if i > 10:
            if (total_diff < threshold or i == iter_break) and ES_flag == 1:
                ES_flag = 0
                for pind in range(pinds):
                    param_kv_copy[erased_base_ind + pind * max_erased_concepts] = param_kv[erased_base_ind + pind * max_erased_concepts].detach().clone()
            if ES_flag == 0:
                break
    # rounded_list = [round(num, 6) for num in ls]
    rounded_list = [round(num, 2) for num in ls]
    rounded_list2 = [round(num, 6) for num in ls2]
    print(rounded_list[10:])   
    print(rounded_list2[10:])   
    if ES_flag !=0:
        for pind in range(pinds):
            param_kv_copy[erased_base_ind + pind * max_erased_concepts] = param_kv[erased_base_ind + pind * max_erased_concepts].detach().clone()
    
    # model.eval()
    # save_model(model, name, erased_index, erased_base_ind,compvis_config_file=config_path, param_kv = param_kv_copy, i_base_tensors=i_base_tensors, diffusers_config_file=diffusers_config_path)
    
def save_model(model, name, erased_index, erased_base_ind,compvis_config_file=None, param_kv=None, i_base_tensors=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    namei = f'{name}-{erased_index}'

    folder_path = f'models/{namei}'
    os.makedirs(folder_path, exist_ok=True)
    model_path = f'{folder_path}/{namei}.pt'
    model_state_dict = model.state_dict()

    ind = 0
    max_erased_concepts = len(i_base_tensors)
    loss_reg1 = 0

    for key in model_state_dict.keys():
        if 'attn2.to_k' in key or 'attn2.to_v' in key:
            v1 = torch.matmul(param_kv[erased_base_ind + ind*max_erased_concepts],i_base_tensors[erased_base_ind].float()) 
            model_state_dict[key] = model_state_dict[key] + v1
            loss_reg1 += torch.sum(torch.abs(v1))
            ind+=1
    print("loss_reg:", loss_reg1/ind, ind)
    torch.save(model_state_dict, model_path)

    print('Saving Model in Diffusers Format')
    savemodelDiffusers(namei, compvis_config_file, diffusers_config_file, device=device )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainSRS_ME',
                    description = 'Finetuning stable diffusion model to erase concepts using SepME method')
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-6)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--erase_cat', help='category for erasure', type=str, required=False, default='object') # style
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    # parser.add_argument('--em_indexes', help='em_indexes for erasing the forgotten concept', type=str, required=True)
    parser.add_argument('--iter_break', help='iterations used to train', type=int, required=False, default=1000)
    # parser.add_argument('--max_erased_concepts', help='max_erased_concepts', type=int, required=True)
    parser.add_argument('--erased_index', help='index of the forgotten concept', type=int, required=True)
    parser.add_argument('--scenei', help='index of the scene', type=int, required=True)
    parser.add_argument('--timei', help='index of the time', type=int, required=True)
    parser.add_argument('--max_base', help='max_base', type=int, required=True)
    parser.add_argument('--reg_beta', help='reg norm for weight diff', type=float, required=False, default=1e-5)
    parser.add_argument('--threshold', help='threshold for sepme', type=float, required=False, default=5e-5)
    parser.add_argument('--scale_factor', help='scale factor', type=float, required=False, default=1e-2)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    args = parser.parse_args()

    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    # em_indexes = [int(d.strip()) for d in args.em_indexes.split(',')]
    train_srsme(args.iter_break, args.reg_beta, args.scenei, args.timei,args.scale_factor, args.lr, args.threshold, args.config_path, args.ckpt_path, args.diffusers_config_path, devices, args.erase_cat,args.erased_index, args.max_base, args.ddim_steps)
