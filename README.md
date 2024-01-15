# Separable Diffusion Model Unlearning
 
<div align='center'>
<img src = 'images/fig1_m.pdf'>
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

Generate data and then put these samples to ./data/train/{0,1,2,3,4,5,6,7,8,9} or ./data/eval/{0,1,2,3,4,5,6,7,8,9}

* python eval-scripts/generate-data.py --model_name='?' --save_path './data/' --num_samples 1 --device 'cuda:5'

model_name can be `Van Gogh`, `Picasso`, `Cezanne`, `Jackson Pollock`, `Caravaggio`, `Keith Haring`, `Kelly McKernan`, `Tyler Edlin`, and `Kilian Eng`.


The optimization process for erasing undesired visual concepts from pre-trained diffusion model weights involves using a short text description of the concept as guidance. The ESD model is fine-tuned with the conditioned and unconditioned scores obtained from frozen SD model to guide the output away from the concept being erased. The model learns from it's own knowledge to steer the diffusion process away from the undesired concept.
<div align='center'>
<img src = 'images/ESD.png'>
</div>

## Generating Images

To generate images from one of the custom models use the following instructions:

* To use `eval-scripts/generate-images.py` you would need a csv file with columns `prompt`, `evaluation_seed` and `case_number`. (Sample data in `data/`)
* To generate multiple images per prompt use the argument `num_samples`. It is default to 10.
* The path to model can be customised in the script.
* It is to be noted that the current version requires the model to be in saved in `stable-diffusion/compvis-<based on hyperparameters>/diffusers-<based on hyperparameters>.pt`
* `python eval-scripts/generate-images.py --model_name='compvis-word_VanGogh-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05' --prompts_path 'stable-diffusion/art_prompts.csv' --save_path 'evaluation_folder' --num_samples 10` 

### UPDATE (NudeNet)
If you want to recreate the results from our paper on NSFW task - please use this https://drive.google.com/file/d/1J_O-yZMabmS9gBA2qsFSrmFoCl7tbFgK/view?usp=sharing

* Untar this file and save it in the homedirectory '~/.NudeNet'
* This should enable the right results as we use this checkpoint for our analysis.
  
## Citing our work
The preprint can be cited as follows
```
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
  year={2023}
}
```
