




import argparse
import dataclasses
import math
import io
import os
from pathlib import Path
import sys
from typing import Any

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
from vqgan_clip_zquantize import *

base_path = 'vqgan_clip_zquantize/'

if __name__ == '__main__':

    path_prompts = [
        #'Use computational methods to study cultural processes through which organizations produce and are shaped by meaning'
        #Apply human-centered design and systems thinking in the rural South in pursuit of truth, justice, and equity, unreal engine
        #Accomplished designer of experiments and principle resource on experimental best-practices
        #('results/sujaya_hobbies', 'My hobbies include dancing, acting, painting, hiking.'),
        #('results/castle_unreal', 'Castle on a hill in the stormy sea, unreal engine.'),

        #('zelda_ocarina', 'zelda in the forest playing an ocarina'),
        #('zelda_ocarina_unreal', 'zelda in the forest playing an ocarina | unreal engine'),
        #('babes', 'hot babe wearing a bikini'),
        #('babes_unreal', 'hot babe wearing a bikini | unreal engine'),
        #('castle_storm', 'Castle in the stormy sea.'),
        #('castle_storm_unreal', 'Castle in the stormy sea, unreal engine'),
        #('castle_sky', 'Castle in the Sky'),
        #('castle_sky_unreal', 'Castle in the Sky, unreal engine'),
        #('neopet_picknick', 'Neopet Picnic'),
        #('children_of_time_1', 'the messenger is trying to help them, but its people are unworthy, so preaches the Temple - why else would they fail their God so often? They must improve and become what god has planned for them, but their manner of life and building and invention is wholly at odds with the vission that the messenger relates to them'),

        #('evan_miles', 'miles davis and herbie hancock play ghetto music'),
        #('evan_transpose', 'transpose your thoughts into another axis of evil'),
        #('evan_trumpets', 'trumpets are the devil\'s instrument'),

        #('hungry_shark', 'Hungry and Angry Shark'),
        #('wild_animal', 'Wild Wolf Eating a Rabbit'),
        #('essence_soul', 'The Essence of the Human Soul'),
        #('house_red', 'red house on a hill overlooking a green field in the night time'),
        #('house_green', 'green house on a hill overlooking a green field in the night time'),
        #('house_pink', 'pink house on a hill overlooking a green field in the night time'),
        #('house_orange', 'orange house on a hill overlooking a green field in the night time'),
        #('house_purple', 'purple house on a hill overlooking a green field in the night time'),
        #('house_blue', 'blue house on a hill overlooking a green field in the night time'),
        #('house_yellow', 'yellow house on a hill overlooking a green field in the night time'),
        
        #('bright_fire', 'Bright Fire in the Dark on a Snowy Mountain'),
        #('campfire', 'campfire in the dark on a snowy mountain overlooking a green field'),
        #('knight', 'knight with a sword'),
        
        #('landscape_night_storm', 'view from the top of a mountains where you can see a lights below at night'),
        
        ('culture_algorithms', 'culture and algorithms'),
        ('culture_code', 'culture and code'),
        ('culture_morality', 'culture and morality'),
        ('water_bottle', 'water bottle'),
        ('water_bottle_ocean', 'water bottle in the ocean'),
        ('water_bottle_sea', 'water bottle in the sea'),
    ]

    post_scripts = [
        ('', ''),
        ('unreal', ', rendered by Unreal Engine'),
        ('deviantart', ', Deviantart'),
        ('artstation', ', Artstation'),
        ('vray', ', vray'),
        ('ghibli', ', style of Studio Ghibli'),
        ('painting', ', painted'),
    ]

    from itertools import product
    full_paths = path_prompts.copy()
    seeds = list(range(3))
    #full_paths = [(*p, 0) for p in full_paths]
    full_paths = [(f'{n}_{sn}_{seed}',f'{d}{sd}', seed) for (n,d),(sn,sd), seed in product(path_prompts, post_scripts, seeds)]
    print(full_paths[-1])

    # start the main loop
    for name, prompt, seed in full_paths:
        
        results_folder = f'wash_course_images/{name}/'
        try:
            os.mkdir(results_folder)
        except:
            pass


        args = argparse.Namespace(
            #prompts=['Use computational methods to study cultural processes through which organizations produce and are shaped by meaning'],
            #prompts = ['Apply human-centered design and systems thinking in the rural South in pursuit of truth, justice, and equity, unreal engine'],
            #prompts = ['Accomplished designer of experiments and principle resource on experimental best-practices'],
            #prompts = ['Castle on a hill in the stormy sea, unreal engine'],
            prompts = [prompt],
            
            image_prompts=[],
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[400, 400],
            init_image=None,
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

        model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
        perceptor = clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

        cut_size = perceptor.visual.input_resolution
        e_dim = model.quantize.e_dim
        f = 2**(model.decoder.num_resolutions - 1)
        make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        n_toks = model.quantize.n_e
        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

        if args.seed is not None:
            torch.manual_seed(args.seed)

        if args.init_image:
            pil_image = Image.open(fetch(args.init_image)).convert('RGB')
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

        for prompt in args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in args.image_prompts:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(Image.open(fetch(path)).convert('RGB'), (sideX, sideY))
            batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = perceptor.encode_image(normalize(batch)).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(device))

        def synth(z):
            z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
            return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

        @torch.no_grad()
        def checkin(i, losses):
            out = synth(z)
            fname = f'{results_folder}/{name}_{i}.png'
            TF.to_pil_image(out[0].cpu()).save(fname)
            #display.display(display.Image(fname))

        def ascend_txt():
            out = synth(z)
            iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

            result = []

            if args.init_weight:
                result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)

            for prompt in pMs:
                result.append(prompt(iii))

            return result

        @dataclasses.dataclass
        class Trainer:
            args: Any
            opt: Any
            z: Any
            z_min: float
            z_max: float
            i: int = 0
            prev_losses: Any = dataclasses.field(default_factory=list)
            display_freq: int = 1
            save_freq: int = 100

            def train(self):
                self.opt.zero_grad()
                self.lossAll = ascend_txt()
                loss = sum(self.lossAll)
                
                self.prev_losses.append(loss.item())
                if self.i % self.display_freq == 0:
                    losses_str = ', '.join(f'{loss.item():g}' for loss in self.lossAll)
                    tqdm.write(f'{name}: i={self.i}, loss={sum(self.lossAll).item():g}, losses={losses_str}')
                
                if self.i % self.save_freq == 0:
                    checkin(self.i, self.lossAll)
                
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

                self.i += 1
            
            def is_converged(self, thresh=0.01, quit_after=10):
                ls = self.prev_losses
                if len(self.prev_losses) > 5:
                    if len([i for i in range(len(ls)-1) if (ls[i+1]-ls[i])>thresh]) > quit_after:
                        #if self.prev_losses[-1] - self.prev_losses[-2] > 0.01:
                        checkin(self.i, self.lossAll)
                        return True
                False

        with tqdm() as pbar:
            trainer = Trainer(args, opt, z, z_min, z_max)
            while True:
                trainer.train()
                if trainer.is_converged():
                    print('image converged')
                    break
                pbar.update()



