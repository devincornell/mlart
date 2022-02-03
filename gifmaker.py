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
import tqdm
import imageio
import typing
from typing import List, Dict
import vqganclip


def text_to_name(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = '_'.join(text.split())
    text = text.replace('__', '_')
    return f'{text}'

def make_gif(
        run_name: str, 
        output_path: pathlib.Path, 
        init_image: pathlib.Path = None, 
        image_prompt_times: Dict[int,List[pathlib.Path]] = [], 
        text_prompt_times: Dict[int, List[str]] = [], 
        size: list = [600, 500], 
        save_freq: int = 2, 
        step_size: float = 0.05, 
        max_iter: int = 300, 
        seed: int = 0, 
        still_frames: int = 0, 
        display_freq: int = None
    ):

    final_folder = output_path.joinpath(f'final/{run_name}/')
    training_folder = output_path.joinpath(f'training/{run_name}/')
    gif_folder = output_path.joinpath(f'tmp/{run_name}/')

    final_folder.mkdir(parents=True, exist_ok=True)
    training_folder.mkdir(parents=True, exist_ok=True)
    gif_folder.mkdir(parents=True, exist_ok=True)

    # parse params for filenames
    init_image_name = init_image.stem if init_image is not None else None
    image_prompt_name = "+".join([f'{t}={".".join([p.stem for p in ips])}' for t, ips in image_prompt_times.items()])
    text_prompt_name = "+".join([f'{t}={".".join([text_to_name(tp) for tp in tps])}' for t, tps in text_prompt_times.items()])
    fname_base = f'{run_name}-{init_image_name}-im{image_prompt_name}-text{text_prompt_name}-{seed}'
    print(f'{run_name=}\n{text_prompt_name=}\n{init_image_name=}\n{image_prompt_name=}')
    print(f'{fname_base=}')

    # make subfolder for storing each iteration for gif
    tmp_folder = gif_folder.joinpath(f'{fname_base}/')
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # add paramaters to trainer
    trainer = vqganclip.VQGANCLIP(
        size=size,
        init_image=str(init_image),
        seed=seed,
        step_size=step_size,

        # these two are added during training since they have timesteps now
        #text_prompts=text_prompt_times.get(0, []),
        #image_prompts=[str(fn) for fn in image_prompts.get(0), []],
    )

    # save original image for some number of frames before changing
    for i in range(still_frames):
        trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter.{i:05d}.png'))

    with tqdm.tqdm() as train_progress_bar:
        while True:

            if trainer.i in text_prompt_times:
                print(f'setting t={trainer.i} text prompts: {text_prompt_times[trainer.i]}')
                trainer.set_text_prompts(text_prompt_times[trainer.i])

            if trainer.i in image_prompt_times:
                prompt_strings = [str(p) for p in image_prompt_times[trainer.i]]
                print(f'setting t={trainer.i} image prompts: {prompt_strings}')
                trainer.set_image_prompts(prompt_strings)
            
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
                break

            # update progress
            train_progress_bar.update()

    # cleanup after
    trainer.save_current_image(final_folder.joinpath(f'{fname_base}_final.png'))

    # save final result for a few frames
    for i in range(trainer.i+1, trainer.i+still_frames+1):
        trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter{i:05d}.png'))
    
    # save a gif image
    images = [imageio.imread(fn) for fn in sorted(map(str, tmp_folder.glob('*.png')))]
    imageio.mimsave(final_folder.joinpath(f'{fname_base}_final.gif'), images, duration=0.2)


