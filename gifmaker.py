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
import PIL
import copy

@dataclasses.dataclass
class AnimateParams:
    init_image: pathlib.Path = None
    img_prompts: Dict[int, List[pathlib.Path]] = dataclasses.field(default_factory=dict)
    txt_prompts: Dict[int, List[str]] = dataclasses.field(default_factory=dict)
    init_image_as_prompt: bool = True
    size: typing.Tuple = None
    still_frames_start: int = 5
    still_frames_end: int = 30
    save_freq: int = 2
    step_size: float = 0.05
    max_iter: int = 300
    seed: int = 0

    def __post_init__(self):

        # make copies of prompts
        self.img_prompts = copy.deepcopy(self.img_prompts)
        self.txt_prompts = copy.deepcopy(self.txt_prompts)

        if self.size is None and self.init_image is not None:
            w,h = PIL.Image.open(self.init_image).size
            aspect_ratio = w/h

            if w > h:
                wb = 600
                hb = int(wb/aspect_ratio)
            else:
                hb = 600
                wb = int(hb*aspect_ratio)
                
            self.size = (wb, hb)

            print(f'calculated size from init image: {self.size}')
        
        # add init image to every image prompt
        if self.init_image_as_prompt and self.init_image is not None:
            for fps in self.img_prompts.values():
                if self.init_image not in fps:
                    fps.append(self.init_image)

        # sort prompts to be consistent every time
        self.img_prompts = {t:[fp for fp in sorted(fps)] for t,fps in sorted(self.img_prompts.items())}
        self.txt_prompts = {t:[txt for txt in sorted(txts)] for t,txts in sorted(self.txt_prompts.items())}

        # check that init image exists
        if self.init_image is not None and not self.init_image.exists():
            raise ValueError(f'Init image {self.init_image} does not exist.')

        # check that prompt images exist
        for fps in self.img_prompts.values():
            for fp in fps:
                if not fp.exists():
                    raise ValueError(f'Prompt image does not exist: {str(fp)}')

    def img_fnames(self, i: int) -> List[str]:
        '''Sorted list of image filenames.
        '''
        return [str(fp) for fp in self.img_prompts[i]]
    
    @property
    def init_image_name(self) -> str:
        '''Get stem of init image for filenames.
        '''
        return self.init_image.stem if self.init_image is not None else None

    @property
    def img_prompt_name(self) -> str:
        '''Combined name of image prompts.
        '''
        return "+".join(f'{t}={".".join([fp.stem for fp in fps])}' for t,fps in self.img_prompts.items())
    
    @property
    def txt_prompt_name(self) -> str:
        '''Return single string representing text prompts for file naming.
        '''
        return '+'.join(f'{t}={".".join(map(self.text_to_name, tps))}' for t,tps in self.txt_prompts.items())

    @staticmethod
    def text_to_name(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = '_'.join(text.split())
        text = text.replace('__', '_')
        return f'{text}'

def make_gif(
        run_name: str, 
        output_path: pathlib.Path, 
        params: AnimateParams,
        display_freq: int = None
    ):

    final_folder = output_path.joinpath(f'final/{run_name}/')
    training_folder = output_path.joinpath(f'training/{run_name}/')
    gif_folder = output_path.joinpath(f'tmp/{run_name}/')

    final_folder.mkdir(parents=True, exist_ok=True)
    training_folder.mkdir(parents=True, exist_ok=True)
    gif_folder.mkdir(parents=True, exist_ok=True)

    # parse params for filenames
    #init_image_name = init_image.stem if init_image is not None else None
    #image_prompt_name = "+".join([f'{t}={".".join([p.stem for p in ips])}' for t, ips in image_prompt_times.items()])
    #text_prompt_name = "+".join([f'{t}={".".join([text_to_name(tp) for tp in tps])}' for t, tps in text_prompt_times.items()])
    fname_base = f'{run_name}-{params.init_image_name}-im{params.img_prompt_name}-text{params.txt_prompt_name}-{params.seed}'
    fname_base = fname_base[:200]
    print(f'{run_name=}\n{params.init_image_name=}\n{params.img_prompt_name=}\n{params.txt_prompt_name=}')
    print(f'{fname_base=}')

    # make subfolder for storing each iteration for gif
    tmp_folder = gif_folder.joinpath(f'{fname_base}/')
    tmp_folder.mkdir(parents=True, exist_ok=True)

    # add paramaters to trainer
    trainer = vqganclip.VQGANCLIP(
        size=params.size,
        init_image=str(params.init_image) if params.init_image is not None else None,
        seed=params.seed,
        step_size=params.step_size,

        # these two are added during training since they have timesteps now
        #text_prompts=text_prompt_times.get(0, []),
        #image_prompts=[str(fn) for fn in image_prompts.get(0), []],
    )

    # save original image for some number of frames before changing
    for i in range(params.still_frames_start):
        trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter.{i:05d}.png'))

    with tqdm.tqdm() as train_progress_bar:
        while True:

            if trainer.i in params.txt_prompts:
                print(f'setting t={trainer.i} text prompts: {params.txt_prompts[trainer.i]}')
                trainer.set_text_prompts(params.txt_prompts[trainer.i])

            if trainer.i in params.img_prompts:
                fnames = params.img_fnames(trainer.i)
                print(f'setting t={trainer.i} image prompts: {fnames}')
                trainer.set_image_prompts(fnames)
            
            # display output if needed
            if display_freq is not None and (trainer.i % display_freq == 0):
                losses_str = ', '.join(f'{loss.item():g}' for loss in trainer.lossAll)
                tqdm.tqdm.write(f'{fname_base}: i={trainer.i}, loss={sum(trainer.lossAll).item():g}, losses={losses_str}')
            
            # save image if required
            if params.save_freq is not None and (trainer.i % params.save_freq == 0):
                trainer.save_current_image(training_folder.joinpath(f'{fname_base}_training.png'))
                trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter{trainer.i:05d}.png'))

            # run epoch
            trainer.epoch()
                
            # check convergence
            if trainer.i > params.max_iter or trainer.is_converged(thresh=0.001, quit_after=10):
                break

            # update progress
            train_progress_bar.update()

    # cleanup after
    trainer.save_current_image(final_folder.joinpath(f'{fname_base}_final.png'))

    # save final result for a few frames
    for i in range(trainer.i+1, trainer.i+params.still_frames_end+1):
        trainer.save_current_image(tmp_folder.joinpath(f'{fname_base}_iter{i:05d}.png'))
    
    # save a gif image
    images = [imageio.imread(fn) for fn in sorted(map(str, tmp_folder.glob('*.png')))]
    imageio.mimsave(final_folder.joinpath(f'{fname_base}_final.gif'), images, duration=0.2)


