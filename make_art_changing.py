



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


if __name__ == '__main__':

    display_freq = None
    save_freq = 5
    run_name = 'dante_change6'
    image_path = pathlib.Path(f'images')
    final_folder = image_path.joinpath(f'final/final_{run_name}/')
    training_folder = image_path.joinpath(f'training/training_{run_name}/')
    gif_folder = image_path.joinpath(f'gifs/gifs_{run_name}/')

    final_folder.mkdir(parents=True, exist_ok=True)
    training_folder.mkdir(parents=True, exist_ok=True)
    gif_folder.mkdir(parents=True, exist_ok=True)

    # these will be changed in order provided
    change_every = 100
    prompt_texts = ['Limbo', 'Lust', 'Gluttony', 'Wrath', 'Heresy', 'Violence', 'Fraud', 'Treachery']
    
    # in order provided
    prefix = ''
    suffix = ''
    prompt_texts = [f'{prefix}{pt}{suffix}' for pt in prompt_texts]
    print(f'{prompt_texts=}')


    #base_name = f'{prompt.name}_step{step_size}_thresh{thresh}_{seed}'
    base_name = f'dantes_inferno1'
    print(f'{base_name=}')

    image_fname = 'images/start_images/dantes_inferno1.png'

    trainer = vqganclip.VQGANCLIP(
        init_image=image_fname,
        text_prompts=[prompt_texts[0]],
        image_prompts=['images/start_images/sad_girl_red1.png'],
        size=[600, 400],
        seed=0,
        #step_size=step_size,
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

            # update text if needed
            print(f'\n{prompt_texts[trainer.i // change_every]}')
            trainer.set_text_prompts([prompt_texts[trainer.i // change_every]])

            # kill condition
            if trainer.i > change_every*len(prompt_texts):
                trainer.save_current_image(final_folder.joinpath(f'{base_name}_final.png'))
                break

            # update progress
            train_progress_bar.update()
