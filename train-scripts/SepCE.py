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

#####################
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    model = load_model_from_config(config_path, ckpt_path, devices[0])

    return model_orig, model

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
def train_sepce(iter_break, reg_beta, scale_factor, lr, config_path, ckpt_path, diffusers_config_path, devices):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    iterations : int
        Number of iterations to train.
    scale_factor : float
        scale_factor for fine tuning.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    model_orig, model = get_models(config_path, ckpt_path, devices)
    prompt1 = 'VanGogh'
    prompt2 = 'Cezanne'
    prompt3 = 'Picasso'
    path_1 = f'./data/{prompt1}/'
    path_2 = f'./data/{prompt2}/'
    path_3 = f'./data/{prompt3}/'
    images_1 = os.listdir(path_1)
    images_2 = os.listdir(path_2)
    images_3 = os.listdir(path_3)
    threshold = [0, 0.00015, 0]
    model.train()

    losses = []
    history = []
    
    name = f'train-joint-SepCE-threshold0_{threshold[0]}-threshold1_{threshold[1]}-threshold1_{threshold[2]}-iter_{iter_break}_beta-{reg_beta}-onlykv'


    emb_1 = model.get_learned_conditioning([prompt1])
    emb_2 = model.get_learned_conditioning([prompt2])
    emb_3 = model.get_learned_conditioning([prompt3])
    emb_0 = model.get_learned_conditioning([''])
    
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
    Picasso_base = np.concatenate(Picasso_bases, 0)
    VanGogh_tensor = scale_factor*torch.tensor(VanGogh_base).to(devices[0])
    Cezanne_tensor = scale_factor*torch.tensor(Cezanne_base).to(devices[0])
    Picasso_tensor = scale_factor*torch.tensor(Picasso_base).to(devices[0])

    VanGogh_tensor.requires_grad = False
    Cezanne_tensor.requires_grad = False
    Picasso_tensor.requires_grad = False
    param_kv,param_kv_copy = get_paras(model, max_base)
    opt = torch.optim.Adam(param_kv, lr=lr)
    weight_flag1 = 1
    weight_flag2 = 1
    weight_flag3 = 1
    ratio = 3
    sum_w1= 0
    sum_w2 = 0
    sum_w3 = 0
    for i, image_name_1 in enumerate(images_1):
        image_path_1 = path_1 + image_name_1
        img_1 = load_img(image_path_1)
        img_1 = preprocess(img_1)
        img_1 = img_1.to(model.device)
        encoder_posterior_1 = model.encode_first_stage(img_1)
        z_1 = model.get_first_stage_encoding(encoder_posterior_1).detach()

        image_path_2 = path_2 + images_2[i]
        img_2 = load_img(image_path_2)
        img_2 = preprocess(img_2)
        img_2 = img_2.to(model.device)
        encoder_posterior_2 = model.encode_first_stage(img_2)
        z_2 = model.get_first_stage_encoding(encoder_posterior_2).detach()

        image_path_3 = path_3 + images_3[i]
        img_3 = load_img(image_path_3)
        img_3 = preprocess(img_3)
        img_3 = img_3.to(model.device)
        encoder_posterior_3 = model.encode_first_stage(img_3)
        z_3 = model.get_first_stage_encoding(encoder_posterior_3).detach()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        z_t1 = model.q_sample(z_1, t_enc_ddpm)
        z_t2 = model.q_sample(z_2, t_enc_ddpm)
        z_t3 = model.q_sample(z_3, t_enc_ddpm)
        
        with torch.no_grad():
            e_t1_0 = model_orig.apply_model(z_t1.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))[0]
            e_t2_0 = model_orig.apply_model(z_t2.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))[0]
            e_t3_0 = model_orig.apply_model(z_t3.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))[0]
            e_t1_n = model_orig.apply_model(z_t1.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_1.to(devices[1]))[0]
            e_t2_n = model_orig.apply_model(z_t2.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_2.to(devices[1]))[0]
            e_t3_n = model_orig.apply_model(z_t3.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_3.to(devices[1]))[0]
        e_t1_pn = model.apply_model(z_t1.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_1.to(devices[0]), (param_kv, VanGogh_tensor, Cezanne_tensor, Picasso_tensor))[0]
        e_t2_pn = model.apply_model(z_t2.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_2.to(devices[0]), (param_kv, VanGogh_tensor, Cezanne_tensor, Picasso_tensor))[0]
        e_t3_pn = model.apply_model(z_t3.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_3.to(devices[0]), (param_kv, VanGogh_tensor, Cezanne_tensor, Picasso_tensor))[0]
        e_t1_p0 = model.apply_model(z_t1.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), (param_kv, VanGogh_tensor, Cezanne_tensor, Picasso_tensor))[0]
        e_t2_p0 = model.apply_model(z_t2.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), (param_kv, VanGogh_tensor, Cezanne_tensor, Picasso_tensor))[0]
        e_t3_p0 = model.apply_model(z_t3.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_0.to(devices[0]), (param_kv, VanGogh_tensor, Cezanne_tensor, Picasso_tensor))[0]
        e_t1_0.requires_grad = False
        e_t2_0.requires_grad = False
        e_t3_0.requires_grad = False
        e_t1_n.requires_grad = False
        e_t2_n.requires_grad = False
        e_t3_n.requires_grad = False


        epn_minus_ep0_t1 = (e_t1_pn.to(devices[0])-e_t1_p0.to(devices[0]))
        epn_minus_ep0_t2 = (e_t2_pn.to(devices[0])-e_t2_p0.to(devices[0]))
        epn_minus_ep0_t3 = (e_t3_pn.to(devices[0])-e_t3_p0.to(devices[0]))
        en_minus_e0_t1 = (e_t1_n.to(devices[0]) - e_t1_0.to(devices[0]))
        en_minus_e0_t2 = (e_t2_n.to(devices[0]) - e_t2_0.to(devices[0]))
        en_minus_e0_t3 = (e_t3_n.to(devices[0]) - e_t3_0.to(devices[0]))
        """
        t1_norm = torch.norm(epn_minus_ep0_t1 * en_minus_e0_t1,2).detach()
        t2_norm = torch.norm(epn_minus_ep0_t2 * en_minus_e0_t2,2).detach()
        t3_norm = torch.norm(epn_minus_ep0_t3 * en_minus_e0_t3,2).detach()
        
        w1 = 1.
        w2 = t1_norm/t2_norm
        w3 = t1_norm/t3_norm
        sum_w1 += w1
        sum_w2 += w2
        sum_w3 += w3
        if i ==  100:
            print("weights list:", 1., sum_w2/sum_w1, sum_w3/sum_w1)
            print(dsd) # end 

        w1 = 1.
        w2 = 9.98
        w3 = 7.63
        """
        t1_norm = torch.norm(epn_minus_ep0_t1 * en_minus_e0_t1,2).detach()
        t2_norm = torch.norm(epn_minus_ep0_t2 * en_minus_e0_t2,2).detach()
        t3_norm = torch.norm(epn_minus_ep0_t3 * en_minus_e0_t3,2).detach()
        
        w1 = 1.
        w2 = t1_norm/t2_norm
        w3 = t1_norm/t3_norm
        
        loss_reg1 = 0
        loss_reg2 = 0
        loss_reg3 = 0
        for pind in range(len(param_kv)//3):
            loss_reg1 += torch.mean(torch.abs(param_kv[pind*3]))
            loss_reg2 += torch.mean(torch.abs(param_kv[pind*3+1]))
            loss_reg3 += torch.mean(torch.abs(param_kv[pind*3+2]))
        direction_diff = torch.mean(epn_minus_ep0_t1 * en_minus_e0_t1) 
        direction_diff1 = torch.mean(epn_minus_ep0_t2 * en_minus_e0_t2)
        direction_diff2 = torch.mean(epn_minus_ep0_t3 * en_minus_e0_t3)
        loss1 =   weight_flag1 * (w1 * direction_diff  + reg_beta * loss_reg1)
        loss2 =   weight_flag2 * (w2 * direction_diff1 + reg_beta * loss_reg2)
        loss3 =   weight_flag3 * (w3 * direction_diff2 + reg_beta * loss_reg3)
        loss =  (loss1 + loss2 + loss3)/(weight_flag1 + weight_flag2 + weight_flag3)
        print("loss:",weight_flag1,direction_diff.detach().cpu().numpy(), weight_flag2, direction_diff1.detach().cpu().numpy(), weight_flag3, direction_diff2.detach().cpu().numpy())
        print("loss_reg",loss_reg1.detach().cpu().numpy(),loss_reg2.detach().cpu().numpy(),loss_reg3.detach().cpu().numpy())
        loss.backward()
        losses.append(loss.item())
        history.append(loss.item())
        opt.step()
        if i == 0:
            total_diff = direction_diff.detach()
            total_diff1 = direction_diff1.detach()
            total_diff2 = direction_diff2.detach()
        else:
            total_diff = 0.9 * total_diff + 0.1 * direction_diff.detach()
            total_diff1 = 0.9 * total_diff1 + 0.1 * direction_diff1.detach()
            total_diff2 = 0.9 * total_diff2 + 0.1 * direction_diff2.detach()
            # selectable
            # total_diff =  0.9 * total_diff  + 0.1 * torch.clamp(direction_diff.detach(),  -0.001, 0.001)
            # total_diff1 = 0.9 * total_diff1 + 0.1 * torch.clamp(direction_diff1.detach(), -0.001, 0.001)
            # total_diff2 = 0.9 * total_diff2 + 0.1 * torch.clamp(direction_diff2.detach(), -0.001, 0.001)
        if i > 10:
            if (total_diff < threshold[0] or i == iter_break) and weight_flag1 == 1:
                weight_flag1 = 0
                for pind in range(len(param_kv_copy)//ratio):
                    param_kv_copy[ratio * pind] = param_kv[ratio * pind].detach().clone()
            if (total_diff1 < threshold[1] or i == iter_break) and weight_flag2 == 1:
                weight_flag2 = 0
                for pind in range(len(param_kv_copy)//ratio):
                    param_kv_copy[ratio * pind+1] = param_kv[ratio * pind+1].detach().clone()
            if (total_diff2 < threshold[2] or i == iter_break) and weight_flag3 == 1:
                weight_flag3 = 0
                for pind in range(len(param_kv_copy)//ratio):
                    param_kv_copy[ratio * pind+2] = param_kv[ratio * pind+2].detach().clone()
            if weight_flag1+weight_flag2+weight_flag3 == 0:
                break

    model.eval()
    bases = [VanGogh_tensor, Cezanne_tensor, Picasso_tensor]
    save_model(model, name, None, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, param_kv = param_kv_copy, bases=bases, diffusers_config_file=diffusers_config_path)

def save_model(model, name, num, compvis_config_file=None, param_kv=None, bases=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'
    name_1 = f'{name}1'
    name_2 = f'{name}2'
    name_3 = f'{name}3'
    name_12 = f'{name}12'
    name_13 = f'{name}13'
    name_23 = f'{name}23'
    name_123 = f'{name}123'
    folder_path1 = f'models/{name_1}'
    folder_path2 = f'models/{name_2}'
    folder_path3 = f'models/{name_3}'
    folder_path12 = f'models/{name_12}'
    folder_path13 = f'models/{name_13}'
    folder_path23 = f'models/{name_23}'
    folder_path123 = f'models/{name_123}'

    os.makedirs(folder_path1, exist_ok=True)
    os.makedirs(folder_path2, exist_ok=True)
    os.makedirs(folder_path3, exist_ok=True)
    os.makedirs(folder_path12, exist_ok=True)
    os.makedirs(folder_path13, exist_ok=True)
    os.makedirs(folder_path23, exist_ok=True)
    os.makedirs(folder_path123, exist_ok=True)
    if num is not None:
        path1 = f'{folder_path1}/{name_1}-epoch_{name_1}.pt'
        path2 = f'{folder_path2}/{name_2}-epoch_{name_2}.pt'
        path3 = f'{folder_path3}/{name_3}-epoch_{name_3}.pt'
        path12 = f'{folder_path12}/{name_12}-epoch_{name_12}.pt'
        path13 = f'{folder_path13}/{name_13}-epoch_{name_13}.pt'
        path23 = f'{folder_path23}/{name_23}-epoch_{name_23}.pt'
        path123 = f'{folder_path123}/{name_123}-epoch_{name_123}.pt'
    else:
        path1 = f'{folder_path1}/{name_1}.pt'
        path2 = f'{folder_path2}/{name_2}.pt'
        path3 = f'{folder_path3}/{name_3}.pt'
        path12 = f'{folder_path12}/{name_12}.pt'
        path13 = f'{folder_path13}/{name_13}.pt'
        path23 = f'{folder_path23}/{name_23}.pt'
        path123 = f'{folder_path123}/{name_123}.pt'
    if save_compvis:
        model_state_dict1 = model.state_dict()
        model_state_dict2 = model.state_dict()
        model_state_dict3 = model.state_dict()
        model_state_dict12 = model.state_dict()
        model_state_dict13 = model.state_dict()
        model_state_dict23 = model.state_dict()
        model_state_dict123 = model.state_dict()
        ind = 0
        ratio = len(bases)
        loss_reg1 = 0
        loss_reg2 = 0
        loss_reg3 = 0
        loss_reg12 = 0
        loss_reg13= 0
        loss_reg23 = 0
        loss_reg123 = 0
        for key in model_state_dict1.keys():
            if 'attn2.to_k' in key or 'attn2.to_v' in key:
                v1 = torch.matmul(param_kv[ind*ratio],bases[0].float()) 
                v2 = torch.matmul(param_kv[ind*ratio+1],bases[1].float()) 
                v3 = torch.matmul(param_kv[ind*ratio+2],bases[2].float())
                model_state_dict1[key] = model_state_dict1[key] + v1
                model_state_dict2[key] = model_state_dict2[key] + v2
                model_state_dict3[key] = model_state_dict3[key] + v3
                model_state_dict12[key] = model_state_dict12[key] + v1 + v2
                model_state_dict13[key] = model_state_dict13[key] + v1 + v3
                model_state_dict23[key] = model_state_dict23[key] + v2 + v3
                model_state_dict123[key] = model_state_dict123[key] + v1 + v2 + v3
                loss_reg1 += torch.sum(torch.abs(v1))
                loss_reg2 += torch.sum(torch.abs(v2))
                loss_reg3 += torch.sum(torch.abs(v3))
                loss_reg12 += torch.sum(torch.abs(v1+v2))
                loss_reg13 += torch.sum(torch.abs(v1+v3))
                loss_reg23 += torch.sum(torch.abs(v2+v3))
                loss_reg123 += torch.sum(torch.abs(v1+v2+v3))
                ind+=1
        print("loss_reg:", loss_reg1/ind, loss_reg2/ind, loss_reg3/ind, loss_reg12/ind, loss_reg13/ind, loss_reg23/ind, loss_reg123/ind)
        torch.save(model_state_dict1, path1)
        torch.save(model_state_dict2, path2)
        torch.save(model_state_dict3, path3)
        torch.save(model_state_dict12, path12)
        torch.save(model_state_dict13, path13)
        torch.save(model_state_dict23, path23)
        torch.save(model_state_dict123, path123)
    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name_1, compvis_config_file, diffusers_config_file, device=device )
        savemodelDiffusers(name_2, compvis_config_file, diffusers_config_file, device=device )
        savemodelDiffusers(name_3, compvis_config_file, diffusers_config_file, device=device )
        savemodelDiffusers(name_12, compvis_config_file, diffusers_config_file, device=device )
        savemodelDiffusers(name_13, compvis_config_file, diffusers_config_file, device=device )
        savemodelDiffusers(name_23, compvis_config_file, diffusers_config_file, device=device )
        savemodelDiffusers(name_123, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainSpeCE',
                    description = 'Finetuning stable diffusion model to erase concepts using TrainSpeCE method')
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-6)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--iter_break', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--reg_beta', help='reg norm for weight diff', type=float, required=False, default=1e-5)
    args = parser.parse_args()

    scale_factor = args.scale_factor
    reg_beta = args.reg_beta
    iter_break = args.iter_break
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]


    train_sepce(iter_break= iter_break, reg_beta = reg_beta, scale_factor=scale_factor, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices)
