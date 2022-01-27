



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


@dataclasses.dataclass
class Prompt:
    base_text: str
    post_text: str
    post_name: str

    @property
    def text(self):
        return f'{self.base_text} {self.post_text}'

    @property
    def name(self):
        text = self.base_text
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = '_'.join(text.split())
        text = text.replace('__', '_')
        return f'{text}_{self.post_name}'


if __name__ == '__main__':
    run_name = 'bridge2'
    max_iter = 300
    display_freq = None
    save_freq = 5
    image_path = pathlib.Path(f'images')
    final_folder = image_path.joinpath(f'final/{run_name}/')
    training_folder = image_path.joinpath(f'training/{run_name}/')
    gif_folder = image_path.joinpath(f'tmp/{run_name}/')

    final_folder.mkdir(parents=True, exist_ok=True)
    training_folder.mkdir(parents=True, exist_ok=True)
    gif_folder.mkdir(parents=True, exist_ok=True)

    prompt_texts = [
        'bright red flowers in the forest',
        'red flowers in the jungle',
        'red forest flowers'
        #'trancendance',
        #'trancendance monk',
        #'buddhist monk',
        #'rainfall',
    ]

    post_prompts = [
        ('', ''),
        #('flickr', 'from Flickr'),
        #('deviantart', 'from Deviantart'),
        #('artstation', 'from Artstation'),
        #('vray', 'from vray'),
        ('ghibli', 'in the style of Studio Ghibli'),
        #('unreal', 'rendered by Unreal Engine'),
    ]

    # generate prompts
    all_prompts = [Prompt(t, pt, pn) for (t, (pn, pt)) in product(prompt_texts, post_prompts)]
    
    # other params
    seeds = list(range(3))
    step_sizes = [0.05, 0.025, 0.01]
    params = list(product(seeds, step_sizes, all_prompts))
    print(f'running {len(params)} param combinations')

    # start the outer loop
    for seed, step_size, prompt in tqdm.tqdm(params):
        
        base_name = f'{prompt.name}_step{step_size}_{seed}'
        print(f'{base_name=}')

        # make subfolder for storing each iteration
        tmp_folder = gif_folder.joinpath(f'{base_name}/')
        tmp_folder.mkdir(parents=True, exist_ok=True)

        trainer = vqganclip.VQGANCLIP(
            init_image='images/start_images/forest_bridge.png',
            size=[600, 400],
            text_prompts=[prompt.text],
            image_prompts=['images/start_images/big_red_flower.avif'],
            
            seed=seed,
            step_size=step_size,
            #device_name='cpu',
        )

        with tqdm.tqdm() as train_progress_bar:
            while True:
                
                # display output if needed
                if display_freq is not None and (trainer.i % display_freq == 0):
                    losses_str = ', '.join(f'{loss.item():g}' for loss in trainer.lossAll)
                    tqdm.tqdm.write(f'{base_name}: i={trainer.i}, loss={sum(trainer.lossAll).item():g}, losses={losses_str}')
                
                # save image if required
                if save_freq is not None and (trainer.i % save_freq == 0):
                    trainer.save_current_image(training_folder.joinpath(f'{base_name}_training.png'))
                    trainer.save_current_image(tmp_folder.joinpath(f'{base_name}_iter{trainer.i:05d}.png'))

                # run epoch
                trainer.epoch()
                    
                # check convergence
                if trainer.i > max_iter or trainer.is_converged(thresh=0.001, quit_after=10):
                    trainer.save_current_image(final_folder.joinpath(f'{base_name}_final.png'))
                    
                    # save a gif image
                    images = [imageio.imread(fn) for fn in sorted(map(str, tmp_folder.glob('*.png')))]
                    imageio.mimsave(final_folder.joinpath(f'{base_name}_final.gif'), images, duration=0.5)
                    
                    #convert -delay 100 -loop 0 images/gifs/gifs_sunset7_starrynight/bright_stars_in_the_sky__step0.05_0/*.png
                    break

                # update progress
                train_progress_bar.update()
