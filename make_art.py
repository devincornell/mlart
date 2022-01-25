




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
        
        args = argparse.Namespace(
            prompts = [prompt_full],
            
            image_prompts=[],
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[600, 400],
            init_image=None,
            #init_image='start_images/snowy_mountain_forest.png',
            init_weight=0.,
            clip_model='ViT-B/32',
            vqgan_config='vqgan_clip_zquantize/vqgan_imagenet_f16_1024.yaml',
            vqgan_checkpoint='vqgan_clip_zquantize/vqgan_imagenet_f16_1024.ckpt',
            step_size=0.05,
            cutn=64,
            cut_pow=1.,
            seed=seed,
        )

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        model = vqgan_clip_zquantize.load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
        perceptor = vqgan_clip_zquantize.clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

        cut_size = perceptor.visual.input_resolution
        e_dim = model.quantize.e_dim
        f = 2**(model.decoder.num_resolutions - 1)
        make_cutouts = vqgan_clip_zquantize.MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = model.quantize.n_e
        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.seed is not None:
            torch.manual_seed(args.seed)

        if args.init_image:
            pil_image = Image.open(vqgan_clip_zquantize.fetch(args.init_image)).convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
        else:
            one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
            z = one_hot @ model.quantize.embedding.weight
            z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        z_orig = z.clone()
        z.requires_grad_(True)
        opt = optim.Adam([z], lr=args.step_size)

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        pMs = []

        # add text prompts
        for prompt in args.prompts:
            txt, weight, stop = vqgan_clip_zquantize.parse_prompt(prompt)
            embed = perceptor.encode_text(vqgan_clip_zquantize.clip.tokenize(txt).to(device)).float()
            pMs.append(vqgan_clip_zquantize.Prompt(embed, weight, stop).to(device))

        # add image prompts
        for prompt in args.image_prompts:
            path, weight, stop = vqgan_clip_zquantize.parse_prompt(prompt)
            img = vqgan_clip_zquantize.resize_image(Image.open(vqgan_clip_zquantize.fetch(path)).convert('RGB'), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = perceptor.encode_image(normalize(batch)).float()
            pMs.append(vqgan_clip_zquantize.Prompt(embed, weight, stop).to(device))

        # add noise prompts
        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(vqgan_clip_zquantize.Prompt(embed, weight).to(device))


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


        trainer = vqganclip.VQGANCLIP(args, opt, z, z_min, z_max)
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



