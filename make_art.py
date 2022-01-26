



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
from itertools import product

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

    display_freq = None
    save_freq = 5
    run_name = 'freedom4'
    image_path = pathlib.Path(f'images')
    final_folder = image_path.joinpath(f'final/final_{run_name}/')
    training_folder = image_path.joinpath(f'training/training_{run_name}/')
    gif_folder = image_path.joinpath(f'gifs/gifs_{run_name}/')

    final_folder.mkdir(parents=True, exist_ok=True)
    training_folder.mkdir(parents=True, exist_ok=True)
    gif_folder.mkdir(parents=True, exist_ok=True)

    prompt_texts = [
        #'freedom',
        #'american flag freedom',
        #'struggle for freedom',
        'power and freedom',
        'power of freedom',
        'freedom from power',
        'freedom is beautiful',
        #'mountain scene at sunset',
        #'sunset',
        #'sunset view from a mountain'
        #'sunset mountain scene',
        #'beautiful sunset',
    ]

    post_prompts = [
        ('', ''),
        ('flickr', 'from Flickr'),
        ('deviantart', 'from Deviantart'),
        ('artstation', 'from Artstation'),
        ('vray', 'from vray'),
        ('ghibli', 'in the style of Studio Ghibli'),
        ('unreal', 'rendered by Unreal Engine'),
    ]
    
    

    # generate prompts
    prompts = [Prompt(t, pt, pn) for (t, (pn, pt)) in product(prompt_texts, post_prompts)]
    
    # other params
    seeds = list(range(3))
    step_sizes = [0.05, 0.025, 0.01]
    threshes = [0.01, 0.03, 0.05]
    params = list(product(step_sizes, threshes, seeds, prompts))
    print(f'running {len(params)} param combinations')



    # start the outer loop
    for step_size, thresh, seed, prompt in tqdm.tqdm(params):
        
        base_name = f'{prompt.name}_step{step_size}_thresh{thresh}_{seed}'
        print(f'{base_name=}')

        trainer = vqganclip.VQGANCLIP(
            text_prompts=[prompt.text],
            init_image='images/start_images/kiersten_freedom_painting_cropped.jpeg',
            size=[400, 600],
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
                    trainer.save_current_image(gif_folder.joinpath(f'{base_name}_iter{trainer.i:04d}.png'))

                # run epoch
                trainer.epoch()
                    
                # check convergence
                if trainer.is_converged(thresh=thresh, quit_after=5):
                    trainer.save_current_image(final_folder.joinpath(f'{base_name}_final.png'))
                    break

                # update progress
                train_progress_bar.update()
