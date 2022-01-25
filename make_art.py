




import argparse
import dataclasses
import math
import io
import os
from pathlib import Path
import sys
from typing import Any
import string

sys.path.append('./taming-transformers')

from IPython import display
from omegaconf import OmegaConf
from PIL import Image
import requests
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import vqgan_clip_zquantize
import vqganclip
#from vqgan_clip_zquantize import *


def convert_to_name(text: str):
    '''Convert text prompt to url name.
    '''
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = '_'.join(text.split())
    text = text.replace('__', '_')
    return text



if __name__ == '__main__':

    prompts = [
        'star wars',
        'lightsaber luke skywalker',
        'lightsaber on spaceship',
        'scarlet witch',
        'batman begins',
        'batman wearing a cape',
        'batman and robin',
        'green lantern',
        'superman strength',
        'super hero wearing a cape',
        'spiderman fighting',
        'spiderman punching villian',
        'spiderman swinging on web',
        'incredible hulk',
        'super powered lazer',
        'yoda lightsaber',
        'captain america',
        'Captain America',
        'thor god of thunder',
        'thor hammer thunder',
        'loki god of mischief',
        'Thor Norse mythology',
        'Loki Norse Mythology',
        #'black sky with stars',
        #'black sky city',
        #'space ship with aliens',
        #'desert sun',
        #'flying in the clouds',
        #'airplane in the clouds',
        #'bird in the sky',
        #'birds flying through clouds in the sky',
    ]

    post_prompts = [
        ('', ''),
        ('flickr', ' from Flickr'),
        ('deviantart', ' from Deviantart'),
        ('artstation', ' from Artstation'),
        ('vray', ' from vray'),
        ('ghibli', ' in the style of Studio Ghibli'),
        ('unreal', ', rendered by Unreal Engine'),
    ]

    #for prompt in path_prompts:
    #    print(f'{prompt}: {convert_to_name(prompt)}')
    #exit()
    

    from itertools import product
    #full_paths = path_prompts.copy()
    seeds = list(range(10))

    #full_paths = [(*p, 0) for p in full_paths]
    #prompt_params = [(prompt, post, )]

    #full_paths = [(f'{n}_{sn}_{seed}',f'{d}{sd}', seed) for (n,d),(sn,sd), seed in product(path_prompts, post_scripts, seeds)]
    #print(full_paths[-1])

    # start the main loop
    for prompt, (post_name, post_text), seed in product(prompts, post_prompts, seeds):

        prompt_full = f'{prompt}{post_text}'
        
        #args = argparse.Namespace(
        #    prompts = [prompt_full],
        #    
        #    image_prompts=[],
        #    noise_prompt_seeds=[],
        #    noise_prompt_weights=[],
        #    size=[600, 400],
        #    init_image=None,
        #    #init_image='start_images/snowy_mountain_forest.png',
        #    init_weight=0.,
        #    clip_model='ViT-B/32',
        #    vqgan_config='vqgan_clip_zquantize/vqgan_imagenet_f16_1024.yaml',
        #    vqgan_checkpoint='vqgan_clip_zquantize/vqgan_imagenet_f16_1024.ckpt',
        #    step_size=0.05,
        #    cutn=64,
        #    cut_pow=1.,
        #    seed=seed,
        #)

        display_freq = 10
        save_freq = 10
        run_name = 'heroes'
        training_folder = f'images/training/{run_name}/'
        final_results_folder = f'images/final/final_{run_name}/'
        
        try:
            os.mkdir(training_folder)
        except:
            pass
        
        try:
            os.mkdir(final_results_folder)
        except:
            pass

        prompt_name = f'{convert_to_name(prompt)}_{post_name}_{seed}'
        results_folder = f'{folder}/{prompt_name}/'

        print(f'starting prompt: {prompt_full} ({prompt_name})')

        #init_image='start_images/snowy_mountain_forest.png',
        trainer = vqganclip.VQGANCLIP(args, prompts=[prompt_full], size=[600, 400])
        while True:
            trainer.epoch()
            
            image_fname = f'{prompt_name}_iter{trainer.i}.png'

            # display output if needed
            if display_freq is not None and (trainer.i % display_freq == 0):
                losses_str = ', '.join(f'{loss.item():g}' for loss in trainer.lossAll)
                tqdm.write(f'{prompt_name}: i={trainer.i}, loss={sum(trainer.lossAll).item():g}, losses={losses_str}')
            
            # save image if required
            if save_freq is not None and (trainer.i % save_freq == 0):
                trainer.save_current_image(f'{intermediate_folder}/{image_fname}')
                
            # check convergence
            if trainer.is_converged():
                trainer.save_current_image(f'{final_results_folder}/{image_fname}')
                break



