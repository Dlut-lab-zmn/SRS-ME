from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import argparse
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

##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler

def preprocess(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(memory_format=torch.contiguous_format).float()
    return img
def train_fmn(iter_break,erase_cat,erased_index, lr, config_path, ckpt_path, diffusers_config_path, devices,ddim_steps=50):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
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
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)
    if erase_cat == 'object':
        prompts = ['chain saw','church','gas pump','tench','garbage truck','english springer','golf ball','parachute','french horn']
    elif erase_cat == 'style':
        prompts = ['Cezanne', 'VanGogh', 'Picasso', 'Jackson Pollock', 'Caravaggio', 'KeithHaring', 'Kelly McKernan', 'Tyler Edlin', 'Kilian Eng']
    elif erase_cat == 'nudity':
        prompts = ['nudity']
    else:
        print('Waiting for research ...')
        print(dsd)
    prompt = prompts[erased_index]
    path_vangogh = f'./data/{prompt}/'
    images_vangogh = os.listdir(path_vangogh)

    model.train()

    losses = []

    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if 'attn2' in name:
        # if 'attn2.to_k' in name or 'attn2.to_v' in name:  
            parameters.append(param)
    parameters_gt = []
    for name, param in model_orig.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if 'attn2' in name:
        # if 'attn2.to_k' in name or 'attn2.to_v' in name:  
            parameters_gt.append(param)
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = f'FMN-{erase_cat}-erased_index_{erased_index}-iter_{iter_break}'

    for i, image_name_vangogh in enumerate(images_vangogh):
        image_path_vangogh = path_vangogh + image_name_vangogh
        img_vangogh = load_img(image_path_vangogh)
        img_vangogh = preprocess(img_vangogh)

        img_vangogh = img_vangogh.to(model.device)

        encoder_posterior_vangogh = model.encode_first_stage(img_vangogh)
        z_vangogh = model.get_first_stage_encoding(encoder_posterior_vangogh).detach()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        emb_vangogh = model.get_learned_conditioning([f'a painting of {prompt}'])
        emb_wovg = model.get_learned_conditioning([f'a painting of'])

        loss_sum = 0
        for j, params in enumerate(parameters):
            loss_j = torch.sum(torch.abs(params.to(devices[0]) - parameters_gt[j].to(devices[0])))
            loss_sum += loss_j
        loss_reg = loss_sum / len(parameters)

        z_n = model.q_sample(z_vangogh, t_enc_ddpm)
        _, attn_set = model.apply_model(z_n.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_vangogh.to(devices[0]), context2 = emb_wovg.to(devices[0]))
        loss = 0
        for attn in attn_set:
            loss += criteria(attn,torch.zeros_like(attn).to(attn.device))
        # update weights to erase the concept
        print('loss:', loss_reg, loss)
        loss.backward()
        losses.append(loss.item())
        history.append(loss.item())
        opt.step()
        if i == iter_break:
            break
    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    path = f'{folder_path}/{name}.pt'
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print):
    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainFMN',
                    description = 'Finetuning stable diffusion model to erase concepts using ESD method')
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--iter_break', help='iter_break used to break train', type=int, required=False, default=20)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--erase_cat', help='category for erasure', type=str, required=False, default='object') # style
    parser.add_argument('--erased_index', help='index of the forgotten concept', type=int, required=True)

    args = parser.parse_args()

    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]

    train_fmn(args.iter_break, args.erase_cat, args.erased_index,args.lr, args.config_path, args.ckpt_path, args.diffusers_config_path, devices, args.ddim_steps)
