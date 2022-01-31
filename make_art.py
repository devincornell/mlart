



import pathlib
import argparse
import dataclasses
import math
import io
import os
from pathlib import Path
import sys
from typing import Any
import string
from itertools import product

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
#from tqdm.notebook import tqdm
import tqdm
import imageio

import vqgan_clip_zquantize
import vqganclip
#from vqgan_clip_zquantize import *

def text_to_name(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = '_'.join(text.split())
    text = text.replace('__', '_')
    return f'{text}'

if __name__ == '__main__':
    run_name = 'sujaya08_angel'
    max_iter = 300
    step_size = 0.05
    display_freq = None
    save_freq = 2
    image_path = pathlib.Path(f'images')
    final_folder = image_path.joinpath(f'final/{run_name}/')
    training_folder = image_path.joinpath(f'training/{run_name}/')
    gif_folder = image_path.joinpath(f'tmp/{run_name}/')

    final_folder.mkdir(parents=True, exist_ok=True)
    training_folder.mkdir(parents=True, exist_ok=True)
    gif_folder.mkdir(parents=True, exist_ok=True)

    text_prompts = [
        'dark angel',
        'dark angel wings',
        'thor god of thunder',
        'ares god of war',
        'devil satan',
        'fiery hell',
    ]

    text_prompts_post = [
        ('', ''),
        ('ghibli', ' in the style of Studio Ghibli'),
        ('deviantart', ' from Deviantart'),
        ('artstation', ' from Artstation'),
        ('vray', ' from vray'),
        #('flickr', ' from Flickr'),
        #('unreal', ' rendered by Unreal Engine'),
    ]


    size = [500, 600]
    image_init = pathlib.Path('images/stock_images/sujaya_angel.jpg')# None
    image_prompts = [
        image_init,
        pathlib.Path('images/stock_images/dark_angel.png'),
    ]
    
    # other params
    params = list(product(list(range(3)), text_prompts_post, text_prompts))
    print(f'running {len(params)} param combinations')

    # start the outer loop
    for seed, (post_name, post_text), text_prompt in tqdm.tqdm(params):
        
        # parse params to filenames
        text_prompt_name = text_to_name(text_prompt)
        image_init_name = image_init.stem if image_init is not None else None
        image_prompt_name = ".".join([p.stem for p in image_prompts])
        fname_base = f'{run_name}-{text_prompt_name}-{post_name}-{image_init_name}-{image_prompt_name}-{seed}'
        print(f'{run_name=}\n{text_prompt_name=}\n{image_init_name=}\n{image_prompt_name=}')
        print(f'{fname_base=}')

        # make subfolder for storing each iteration for gif
        tmp_folder = gif_folder.joinpath(f'{fname_base}/')
        tmp_folder.mkdir(parents=True, exist_ok=True)

        # add paramaters to trainer
        trainer = vqganclip.VQGANCLIP(
            size=size,
            init_image=str(image_init),
            text_prompts=[f'{text_prompt}{post_text}'],
            image_prompts=[str(fn) for fn in image_prompts],
            seed=seed,
            step_size=step_size,
        )

        # save original image for some number of frames before changing
        for i in range(5):
            trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter.{i:05d}.png'))


        with tqdm.tqdm() as train_progress_bar:
            while True:
                
                # display output if needed
                if display_freq is not None and (trainer.i % display_freq == 0):
                    losses_str = ', '.join(f'{loss.item():g}' for loss in trainer.lossAll)
                    tqdm.tqdm.write(f'{fname_base}: i={trainer.i}, loss={sum(trainer.lossAll).item():g}, losses={losses_str}')
                
                # save image if required
                if save_freq is not None and (trainer.i % save_freq == 0):
                    trainer.save_current_image(training_folder.joinpath(f'{fname_base}_training.png'))
                    trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter{trainer.i:05d}.png'))

                # run epoch
                trainer.epoch()
                    
                # check convergence
                if trainer.i > max_iter or trainer.is_converged(thresh=0.001, quit_after=10):
                    trainer.save_current_image(final_folder.joinpath(f'{fname_base}_final.png'))

                    # save final result for a few frames
                    for i in range(trainer.i+1, trainer.i+6):
                        trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter.{i:05d}.png'))
                    
                    # save a gif image
                    images = [imageio.imread(fn) for fn in sorted(map(str, tmp_folder.glob('*.png')))]
                    imageio.mimsave(final_folder.joinpath(f'{fname_base}_final.gif'), images, duration=0.2)
                    
                    #convert -delay 100 -loop 0 images/gifs/gifs_sunset7_starrynight/bright_stars_in_the_sky__step0.05_0/*.png
                    break

                # update progress
                train_progress_bar.update()
