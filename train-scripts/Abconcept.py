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

def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    model = load_model_from_config(config_path, ckpt_path, devices[0])
    return model_orig, model

def preprocess(img):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.to(memory_format=torch.contiguous_format).float()
    return img

def train_abconcept(prompt, iter_break, lr, config_path, ckpt_path, diffusers_config_path, devices):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    iter_break : int
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
    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    model_orig, model = get_models(config_path, ckpt_path, devices)

    path = f'./data/{prompt}/'
    images = os.listdir(path)

    model.train()

    losses = []

    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        if 'attn2' in name:  
            parameters.append(param)
    parameters_gt = []
    for name, param in model_orig.model.diffusion_model.named_parameters():
        if 'attn2' in name:  
            parameters_gt.append(param)
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = f'abconcept-{prompt}-iter_{iter_break}'

    for i, image_name in enumerate(images):
        image_path = path + image_name
        img = load_img(image_path)
        img = preprocess(img)

        img = img.to(model.device)

        encoder_posterior = model.encode_first_stage(img)
        z = model.get_first_stage_encoding(encoder_posterior).detach()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        emb = model.get_learned_conditioning([f'{prompt} painting'])
        emb_wovg = model.get_learned_conditioning(['painting'])

        loss_sum = 0
        for j, params in enumerate(parameters):
            loss_j = torch.sum(torch.abs(params.to(devices[0]) - parameters_gt[j].to(devices[0])))
            loss_sum += loss_j
        loss_reg = loss_sum / len(parameters)

        z_n = model.q_sample(z, t_enc_ddpm)
        with torch.no_grad():
            e_n = model_orig.apply_model(z_n.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_wovg.to(devices[1]))[0]
        e_pn = model.apply_model(z_n.to(devices[0]), t_enc_ddpm.to(devices[0]), emb.to(devices[0]))[0]
        e_p0 = model.apply_model(z_n.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_wovg.to(devices[0]))[0]
        e_n.requires_grad = False
        loss1 = criteria(e_n.to(devices[0]), e_p0.to(devices[0]))
        loss = criteria(e_pn.to(devices[0]), e_n.to(devices[0])) + 10*loss1
        print("loss:", loss_reg, 10*loss1, loss - 10*loss1)
        loss.backward()
        losses.append(loss.item())
        history.append(loss.item())
        opt.step()
        if i == iter_break:
            break
    model.eval()

    save_model(model, name, None, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

def save_model(model, name, num, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):

    folder_path = f'models/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
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
                    prog = 'Abconcept',
                    description = 'Finetuning stable diffusion model to erase concepts using Abconcept method')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-6)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--iter_break', help='iter_break used to break train', type=int, required=False, default=50)
    args = parser.parse_args()

    prompt = args.prompt
    iter_break = args.iter_break
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]

    train_abconcept(prompt=prompt, iter_break= iter_break, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices)
