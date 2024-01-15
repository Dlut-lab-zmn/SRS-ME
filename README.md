# Separable Diffusion Model Unlearning
 
<div align='center'>
<img src = 'images/1.png'>
</div>

Issues: 
* affect the overall model performance when erasing concepts;
* overlook the challenges associated with iterative concept erasure and subsequent restoration.

We propose 
* Concept-irrelevant Erasure (CiE);
* Separable Concept Erasure (SepCE).

CiE can preserve overall model performance and effectively eliminate concepts, which includes a direction term and momentum statistics to guide the generation of concept-irrelevant representations.
Building upon CiE, SepCE separates optimizable model weights, with each weight term corresponding to a specific concept erasure without affecting generative performance on other concepts. 
To this end, distinct basic solution sets are constructed for different concepts, and each weight is reconstructed as a linear combination of its corresponding set.

Efficacy of our approaches in 
* eliminating concepts;
* preserving model performance;
* offering flexibility in the erasure or recovery of various concepts.

## Fine-tuned Weights

We do not offer the fine-tuned weights for download, as the training process is faster.

## Installation Guide

* To get started clone the following repository of Original Stable Diffusion [Link](https://github.com/CompVis/stable-diffusion)
* Then replace the files from our repository (ldm-replace.txt) to `stable-diffusion` main directory of stable diffusion. 
* Download the weights from [here](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt) and move them to `stable-diffusion/models/ldm/` (This will be `ckpt_path` variable in `train-scripts/train-esd.py`)
* [Only for training] To convert your trained models to diffusers download the diffusers Unet config from [here](https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json)  (This will be `diffusers_config_path` variable in `train-scripts/train-esd.py`)

## Training Guide

After installation, follow these instructions to train a machine unlearning model:

1. Generate data and then put these samples to ./data/train/{0,1,2,3,4,5,6,7,8,9} or ./data/eval/{0,1,2,3,4,5,6,7,8,9}

* `python eval-scripts/generate-data.py --prompt 'Van Gogh' --model_name '0' --save_path './data/' --num_samples 1 --device 'cuda:5'`

prompt can be `Van Gogh`, `Picasso`, `Cezanne`, `Jackson Pollock`, `Caravaggio`, `Keith Haring`, `Kelly McKernan`, `Tyler Edlin`, and `Kilian Eng`.

2. Train classification model 
* python train-scripts/artist-cls-train.py --device 'cuda:0'

3. Train FMN, Abconcept, Esd, CiE, SepCE
* `python train-scripts/FMN.py --lr 1e-5 --iter_break 10 --prompt 'Van Gogh' --devices '0,1'`
* `python train-scripts/Abconcept.py --lr 1e-5 --iter_break 50 --prompt 'Van Gogh' --devices '0,1'`
* `python train-scripts/Esd.py --prompt 'Van Gogh' --train_method 'xattn' --iterations 1000 --devices '0,1'`
* `python train-scripts/CiE.py --threshold 0 --lr 1e-6 --reg_beta 3e-5 --prompt 'Van Gogh' --devices '0,1'`
* `python train-scripts/SepCE.py --lr 1e-2 --scale_factor 1e-4 --reg_beta 3e-5 --devices '0,1'`

## Generating Images

To generate images from one of the custom models use the following instructions:

* To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
* To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
* `python eval-scripts/generate-images.py --model_name='?' --prompts_path './data/?.csv' --save_path 'evaluation_folder' --num_samples 5` 

## Citing our work
The preprint can be cited as follows
```

```
