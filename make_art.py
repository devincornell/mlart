




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
        text = self.text
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = '_'.join(text.split())
        text = text.replace('__', '_')
        return f'{text}_{self.post_name}'


if __name__ == '__main__':

    display_freq = None
    save_freq = 5
    run_name = 'tester'
    training_folder = f'images/training/training_{run_name}/'
    final_results_folder = f'images/final/final_{run_name}/'

    try:
        os.mkdir(training_folder)
    except:
        pass
    
    try:
        os.mkdir(final_results_folder)
    except:
        pass

    prompt_texts = [
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
    seeds = list(range(5))


    params = list(product(prompts, seeds))
    print(f'running {len(params)} param combinations')



    # start the main loop
    
    for prompt, seed in tqdm.tqdm(params):
        
        base_name = f'{prompt.name}_{seed}'
        print(f'{base_name=}')


        trainer = vqganclip.VQGANCLIP(
            text_prompts=[prompt.text],
            init_image='images/start_images/mountain_sunset_1.png',
            size=[600, 400],
            seed=seed,
            step_size=0.01,
        )

        with tqdm.tqdm() as train_progress_bar:
            while True:
                trainer.epoch()

                # update progress
                train_progress_bar.update()

                # display output if needed
                if display_freq is not None and (trainer.i % display_freq == 0):
                    losses_str = ', '.join(f'{loss.item():g}' for loss in trainer.lossAll)
                    tqdm.tqdm.write(f'{base_name}: i={trainer.i}, loss={sum(trainer.lossAll).item():g}, losses={losses_str}')
                
                # save image if required
                if save_freq is not None and (trainer.i % save_freq == 0):
                    trainer.save_current_image(f'{training_folder}/{base_name}_training.png')
                    
                # check convergence
                if trainer.is_converged(thresh=0.01, quit_after=5):
                    trainer.save_current_image(f'{final_results_folder}/{base_name}_final.png')
                    break
